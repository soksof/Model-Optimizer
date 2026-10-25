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

"""Staging every module of a local checkpoint's remote code."""

import shutil
from types import SimpleNamespace

import pytest

dmu = pytest.importorskip("transformers.dynamic_module_utils")

from modelopt.torch.models import _remote_code
from modelopt.torch.models._remote_code import stage_full_remote_closure


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
