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

import pytest

dmu = pytest.importorskip("transformers.dynamic_module_utils")

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
