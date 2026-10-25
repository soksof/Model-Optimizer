# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Staging every module of a local checkpoint's remote code, and keeping a patch on it."""

import shutil
import sys
from types import SimpleNamespace

import pytest

dmu = pytest.importorskip("transformers.dynamic_module_utils")

from modelopt.torch.models import _remote_code
from modelopt.torch.models._remote_code import keep_patched, stage_full_remote_closure


def test_every_module_of_a_local_checkpoint_is_staged(monkeypatch, tmp_path):
    """transformers stages a remote module and its direct imports only, so the third module of
    a -> b -> c is missing until the wrap copies the checkpoint's other modules. Installing again
    wraps the loader once."""
    ckpt, stage = tmp_path / "ckpt", tmp_path / "stage"
    ckpt.mkdir()
    stage.mkdir()
    for name, body in (
        ("a.py", "from .b import B\n"),
        ("b.py", "from .c import C\n"),
        ("c.py", ""),
    ):
        (ckpt / name).write_text(body)

    def stage_one_hop(path, module_file, *args, **kwargs):
        for name in (module_file, "b.py"):
            shutil.copy(ckpt / name, stage / name)
        return str(stage / module_file)

    monkeypatch.setattr(dmu, "get_cached_module_file", stage_one_hop)
    stage_full_remote_closure()
    wrapped = dmu.get_cached_module_file
    stage_full_remote_closure()

    assert dmu.get_cached_module_file is wrapped
    assert dmu.get_cached_module_file(str(ckpt), "a.py") == str(stage / "a.py")
    assert sorted(p.name for p in stage.iterdir()) == ["a.py", "b.py", "c.py"]


def test_a_module_is_staged_whole_or_not_at_all(monkeypatch, tmp_path):
    """Every rank stages into the same dir and imports what it finds there, so a module must not
    appear before its copy completes: a copy cut short leaves nothing behind for a rank to take as
    staged, and the next load stages the whole module."""
    ckpt, stage = tmp_path / "ckpt", tmp_path / "stage"
    ckpt.mkdir()
    stage.mkdir()
    (ckpt / "a.py").write_text("from .b import B\n")
    (ckpt / "b.py").write_text("B = 1\n" * 100)

    def stage_only_the_module(path, module_file, *args, **kwargs):
        shutil.copy(ckpt / module_file, stage / module_file)
        return str(stage / module_file)

    def copy_cut_short(src, dst):
        with open(src) as f, open(dst, "w") as out:
            out.write(f.read()[:10])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(dmu, "get_cached_module_file", stage_only_the_module)
    stage_full_remote_closure()
    with monkeypatch.context() as m:
        m.setattr(_remote_code, "shutil", SimpleNamespace(copy=copy_cut_short))
        with pytest.warns(UserWarning, match="No space left"):
            dmu.get_cached_module_file(str(ckpt), "a.py")
    assert not (stage / "b.py").exists()

    dmu.get_cached_module_file(str(ckpt), "a.py")
    assert (stage / "b.py").read_text() == (ckpt / "b.py").read_text()


def test_a_patch_stays_on_a_module_transformers_loads_again(monkeypatch, tmp_path):
    """transformers re-executes a remote module it loads again (when the files it hashes change, or
    forced as here), rebinding the module's globals. The patch must be back on the module the class
    is resolved from, which is the one from_pretrained builds from."""
    monkeypatch.setattr(dmu, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    monkeypatch.setattr(dmu, "get_class_in_module", dmu.get_class_in_module)
    monkeypatch.setattr(_remote_code, "_KEPT_PATCHES", {})
    loaded = set(sys.modules)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "modeling_kimi_k25.py").write_text(
        "class DeepseekV3ForCausalLM:\n    pass\n\n\nclass KimiK25ForConditionalGeneration:\n    pass\n"
    )
    native = type("DeepseekV3ForCausalLM", (), {})
    ref = "modeling_kimi_k25.KimiK25ForConditionalGeneration"
    try:
        first = dmu.get_class_from_dynamic_module(ref, str(ckpt))
        patched = keep_patched(
            ".modeling_kimi_k25", lambda module: setattr(module, "DeepseekV3ForCausalLM", native)
        )
        again = dmu.get_class_from_dynamic_module(ref, str(ckpt), force_download=True)

        assert patched == [first.__module__]
        assert again is not first  # the module was executed again
        assert sys.modules[again.__module__].DeepseekV3ForCausalLM is native
    finally:
        for name in set(sys.modules) - loaded:
            del sys.modules[name]
