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

"""What the expert-parallel loader reads off the loaded model and its config.

The loader takes an expert-parallel plan and keeps its experts out of ModelOpt's per-expert quant
wrappers, ``_seed_parallel_states`` gives every module the parallel state its amax syncs over,
``expert_parallel_mesh`` hands calibration the mesh an expert-parallel plan's experts are sharded
over, and ``_ckpt_quant_method`` reads the checkpoint's native quantization. Everything runs in one
CPU process.
"""

import dataclasses
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from transformers import FineGrainedFP8Config

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import moe
from modelopt.torch.utils.distributed import DistributionPlan, set_distribution_plan
from modelopt.torch.utils.moe import expert_parallel_mesh
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import (
    _ckpt_quant_method,
    _experts_sharded_over,
    _refuse_unquantizable_experts,
    _seed_parallel_states,
    _transformers_mesh_builder,
    load_and_prepare_ep,
)

# ----------------------------------------------------------------------------------- tiny MoE


class _Experts(nn.Module):
    """A layer's routed experts as one grouped-GEMM container (every expert in fused tensors)."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 4, 2))
        self.down_proj = nn.Parameter(torch.zeros(4, 2, 2))
        self.weight_quantizer = nn.Identity()  # quantizer insertion adds children like this

    def forward(self, hidden_states, top_k_index, top_k_weights, process_group=None):
        return hidden_states


class _SharedExperts(nn.Module):
    """The always-on shared expert MLP: dense, replicated on every rank."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(2, 4, bias=False)

    def forward(self, hidden_states):
        return hidden_states


class _MoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(2, 2)
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(2, 4, bias=False)  # the router
        self.mlp.experts = _Experts()
        self.mlp.shared_experts = _SharedExperts()


def _model_with_decoder_at(prefix):
    """A two-layer MoE whose decoder (``layers.<i>``) sits under ``prefix`` (``""``: the decoder)."""
    decoder = nn.Module()
    decoder.embed_tokens = nn.Embedding(8, 2)
    decoder.layers = nn.ModuleList(_MoeLayer() for _ in range(2))
    if not prefix:
        return decoder
    root = parent = nn.Module()
    *path, leaf = prefix.rstrip(".").split(".")
    for name in path:
        parent.add_module(name, nn.Module())
        parent = getattr(parent, name)
    parent.add_module(leaf, decoder)
    return root


# ------------------------------------------------------------------------------- the plan

_EP_PLAN = {"layers.*.mlp.gate": "ep_router", "layers.*.mlp.experts": "moe_tp_experts"}
_BLOCK_FP8 = {"quant_method": "fp8", "weight_block_size": [128, 128]}  # DeepSeek-V3's format


class _Mesh:
    """A device mesh, as far as the loader reads one before it seeds the experts' parallel state."""

    device_type = "cpu"

    def __getitem__(self, name):
        return self

    def size(self):
        return 2


@pytest.fixture
def ep_world_of_two(monkeypatch):
    """A two-rank expert-parallel world as far as the loader's checks see, and the checkpoint paths
    it resolves; nothing else is set up."""
    resolved = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(
        model_load_utils, "_resolve_checkpoint_dir", lambda p, r: resolved.append(p) or p
    )
    return resolved


@pytest.mark.parametrize(
    ("plan", "error", "match"),
    [
        pytest.param(DistributionPlan(2), ValueError, "shards no routed experts", id="no-experts"),
        # FSDP2 within each of several expert-parallel groups: a layout the export cannot write.
        pytest.param(
            DistributionPlan(4, 2, "fsdp"),
            NotImplementedError,
            "expert-data-parallel",
            id="dense-hsdp",
        ),
    ],
)
def test_the_loader_refuses_a_plan_it_cannot_carry_out(plan, error, match):
    """Before anything else: no process group, transformers release or checkpoint is looked at."""
    with pytest.raises(error, match=match):
        load_and_prepare_ep("unused", plan)


def test_the_loader_refuses_a_plan_for_another_world(gloo_world_of_one):
    with pytest.raises(ValueError, match="is for 2 ranks, but this run has 1"):
        load_and_prepare_ep("unused", DistributionPlan(2, ep_size=2))


@pytest.mark.parametrize(
    "quantization_config",
    [None, _BLOCK_FP8, {"quant_method": "mxfp4"}],
    ids=["unquantized", "block-fp8", "mxfp4"],
)
def test_a_hub_id_is_made_local_before_the_weights_load(monkeypatch, quantization_config):
    """The export reads the tensors the load could not place back from the checkpoint it records,
    which must be a local directory: a Hub ID there leaves the export nothing to read. The load
    resolves it once the plan and the config pass, before anything reads a weight: a checkpoint
    unquantized or in a format the load dequantizes (block FP8, MXFP4) passes the format check."""

    class _ResolvedError(Exception):
        """Stops the load where it resolves the checkpoint."""

    resolved = []

    def resolve(path, rank):
        resolved.append(path)
        raise _ResolvedError

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "_resolve_checkpoint_dir", resolve)

    config = SimpleNamespace(base_model_ep_plan=_EP_PLAN)
    if quantization_config is not None:
        config.quantization_config = quantization_config

    with pytest.raises(_ResolvedError):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)

    assert resolved == ["org/model"]


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        *(
            (
                SimpleNamespace(quantization_config={"quant_method": method}),
                NotImplementedError,
                f"cannot dequantize a '{method}'",
            )
            for method in ("gptq", "modelopt", "compressed-tensors")
        ),
        (
            SimpleNamespace(
                quantization_config={"quant_method": "fp8", "activation_scheme": "static"}
            ),
            NotImplementedError,
            "cannot dequantize a 'fp8-per-tensor'",
        ),
        (SimpleNamespace(), ValueError, "declare base_model_ep_plan"),
        (
            SimpleNamespace(base_model_ep_plan={"layers.*.self_attn.q_proj": "colwise"}),
            ValueError,
            "styles no routed experts",
        ),
    ],
    ids=[
        "gptq",
        "modelopt",
        "compressed-tensors",
        "per-tensor-fp8",
        "no-ep-plan",
        "no-routed-experts",
    ],
)
def test_a_config_the_load_cannot_take_is_refused_before_anything_loads(
    ep_world_of_two, config, error, match
):
    """A format transformers' own quantizer would load as stored, for PTQ to treat as plain weights,
    or a plan that leaves the experts whole on every rank; the load refuses either before it reads
    anything."""
    with pytest.raises(error, match=match):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)
    assert ep_world_of_two == []


def test_an_ep_size_that_does_not_divide_the_experts_is_refused_before_anything_loads(
    ep_world_of_two,
):
    """The all-to-all transport finds an expert's rank by dividing its id by this rank's share of
    the experts: with 3 experts over 2 ranks the ranks would disagree on who holds which, and send
    tokens to the wrong experts."""
    config = SimpleNamespace(base_model_ep_plan=_EP_PLAN, num_experts=3)

    with pytest.raises(ValueError, match="ep_size=2 must divide the 3 routed experts"):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)
    assert ep_world_of_two == []


def test_a_plan_whose_expert_containers_match_no_module_is_refused(ep_world_of_two, monkeypatch):
    """A plan that styles the grouped-GEMM parameters a level below the experts names a container no
    module is. Loaded anyway, every module would get the dense state, and the experts' amax would
    reduce over ranks holding different experts."""

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _model_with_decoder_at("model.")

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: None)
    config = SimpleNamespace(
        base_model_ep_plan={"layers.*.mlp.experts.*.gate_up_proj": "grouped_gemm"}
    )

    with pytest.raises(ValueError, match="that no module of"):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)


def test_routed_experts_without_experts_in_their_names_are_refused(ep_world_of_two, monkeypatch):
    """Calibration tells routed experts by an ``experts`` in their names, and reduces the weight amax
    of any other module over the expert-parallel group, merging the scales of different experts."""
    model = _model_with_decoder_at("model.")
    for layer in model.model.layers:
        layer.mlp.routed = _Experts()

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return model

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: None)
    config = SimpleNamespace(
        base_model_ep_plan={"layers.*.mlp.routed": "moe_tp_experts"}, num_experts=4
    )

    with pytest.raises(
        NotImplementedError, match=r"2 routed-expert .*model\.layers\.0\.mlp\.routed:"
    ):
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)


@pytest.mark.parametrize("loaded", [4, 3, None], ids=["dividing", "not-dividing", "not-shown"])
def test_an_expert_count_the_config_does_not_name_is_read_off_the_loaded_experts(
    ep_world_of_two, gloo_world_of_one, monkeypatch, loaded
):
    """A config that names its expert count none of the ways the loader reads is checked against the
    loaded experts' weights: transformers keeps the whole count on the DTensors it shards them into,
    and 5.16.0 splits a count ep_size does not divide without an error. A count neither shows is
    refused, as the distribution plan's decision refuses it."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    class _PastError(Exception):
        """Raised by the load's next check: the experts' count passed."""

    model = _model_with_decoder_at("model.")
    if loaded is not None:
        mesh = init_device_mesh("cpu", (1,))
        for layer in model.model.layers:
            experts = layer.mlp.experts
            for name, param in list(experts.named_parameters(recurse=False)):
                whole = torch.zeros(loaded, *param.shape[1:])
                setattr(experts, name, nn.Parameter(distribute_tensor(whole, mesh, [Shard(0)])))

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return model

    def next_check(experts):
        raise _PastError

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)
    monkeypatch.setattr(model_load_utils, "record_unplaced_source_keys", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "_refuse_unquantizable_experts", next_check)
    config = SimpleNamespace(base_model_ep_plan=_EP_PLAN)
    shown = r"\[3\]" if loaded == 3 else "none"
    outcome = (
        pytest.raises(_PastError)
        if loaded == 4
        else pytest.raises(ValueError, match=f"weights show {shown}")
    )

    with outcome:
        load_and_prepare_ep("org/model", DistributionPlan(2, ep_size=2), hf_config=config)


def test_the_weights_load_in_the_checkpoints_dtype_through_its_class(ep_world_of_two, monkeypatch):
    """The checkpoint's ``dtype`` reaches ``from_pretrained``, and the class that builds it is chosen
    with the trust the caller gave its code."""

    class _LoadedError(Exception):
        """Stops the load at from_pretrained."""

    loaded, classes = {}, []

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded.update(kwargs)
            raise _LoadedError

    def auto_class(hf_config, require_lm_head=True, trust_remote_code=False):
        classes.append(trust_remote_code)
        return _Loader

    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", auto_class)
    config = SimpleNamespace(dtype=torch.float16, base_model_ep_plan=_EP_PLAN)

    with pytest.raises(_LoadedError):
        load_and_prepare_ep(
            "org/model", DistributionPlan(2, ep_size=2), trust_remote_code=True, hf_config=config
        )
    assert loaded["dtype"] is torch.float16
    assert classes == [True]


def test_a_compressed_checkpoint_a_spec_decompresses_as_it_loads_passes(monkeypatch):
    """A model spec that installs the on-the-fly decompressor (Kimi-K2.5's) marks the config; that
    checkpoint passes the format check, and the same checkpoint unmarked does not."""
    from modelopt.torch.models import _onfly_decompress

    class _ResolvedPastFormatError(Exception):
        """Stops the load where it resolves the checkpoint, past the format check."""

    def resolve(path, rank):
        raise _ResolvedPastFormatError

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(moe, "require_transformers", lambda *args: None)
    monkeypatch.setattr(model_load_utils, "_resolve_checkpoint_dir", resolve)
    config = SimpleNamespace(
        quantization_config={"quant_method": "compressed-tensors"}, base_model_ep_plan=_EP_PLAN
    )
    plan = DistributionPlan(2, ep_size=2)

    with pytest.raises(NotImplementedError, match="cannot dequantize"):
        load_and_prepare_ep("org/model", plan, hf_config=config)
    _onfly_decompress._mark(config, "pack-quantized")
    with pytest.raises(_ResolvedPastFormatError):
        load_and_prepare_ep("org/model", plan, hf_config=config)


# ------------------------------------------------------------------------ quant wrappers


@pytest.fixture
def quant_wrappers():
    """ModelOpt quant wrappers registered for the experts' and the shared experts' classes."""
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModule, QuantModuleRegistry

    class _PerExpertQuantWrapper(QuantModule):
        pass

    classes = (_Experts, _SharedExperts)
    QuantModuleRegistry.register({cls: f"test.{cls.__name__}" for cls in classes})(
        _PerExpertQuantWrapper
    )
    yield QuantModuleRegistry
    for cls in classes:
        QuantModuleRegistry.unregister(cls)


def test_routed_experts_with_a_quant_wrapper_of_their_own_are_refused(quant_wrappers):
    """Quantization would convert them to it, whose forward drops the process group the all-to-all
    transport passes in. The registry serves every model in the process, so no registration is
    added or removed; the shared experts, which the transport does not manage, may keep a wrapper."""
    model = _model_with_decoder_at("model.")

    with pytest.raises(NotImplementedError, match=r"cannot quantize _Experts: ModelOpt quantizes"):
        _refuse_unquantizable_experts([layer.mlp.experts for layer in model.model.layers])

    assert _Experts in quant_wrappers
    assert _SharedExperts in quant_wrappers


@pytest.mark.parametrize(
    ("wrapper", "outcome"),
    [
        ("_QuantFusedExperts", nullcontext()),
        ("_QuantFP8Experts", pytest.raises(NotImplementedError, match="on their eager forward")),
    ],
    ids=["grouped-gemm", "eager-only"],
)
def test_routed_experts_under_a_fused_experts_wrapper(wrapper, outcome):
    """mtq.quantize registers the generic fused-experts wrapper, for the whole process, for every
    fused-experts class it meets, and that wrapper quantizes through the grouped GEMM: a second
    expert-parallel load of the same class is not refused for it. FP8's applies its block scales on
    the eager forward only, which the all-to-all transport does not run: refused."""
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry
    from modelopt.torch.quantization.plugins import huggingface

    model = _model_with_decoder_at("model.")
    QuantModuleRegistry.register({_Experts: "test._Experts"})(getattr(huggingface, wrapper))
    try:
        with outcome:
            _refuse_unquantizable_experts([layer.mlp.experts for layer in model.model.layers])
    finally:
        QuantModuleRegistry.unregister(_Experts)


@pytest.mark.parametrize("layout", ["says-so", "by-its-shapes", "standard"])
def test_transposed_experts_are_refused(layout):
    """The grouped quantization leaves experts stored transposed (GPT-OSS's) unquantized: the model
    would be quantized everywhere but its experts. Quantization tells them by their flag or, for
    gated experts that do not say so, by their shapes; so does the load."""
    model = _model_with_decoder_at("model.")
    experts = [layer.mlp.experts for layer in model.model.layers]
    for module in experts:
        if layout == "says-so":
            module.is_transposed = True
        else:  # a fused-experts container: 4 experts, gate and up 2 x 2 wide, on hidden size 2
            module.num_experts = 4
            if layout == "by-its-shapes":  # (num_experts, in_dim, out_dim)
                module.gate_up_proj = nn.Parameter(torch.zeros(4, 2, 4))
    refused = pytest.raises(NotImplementedError, match="cannot quantize _Experts: these experts")

    with refused if layout != "standard" else nullcontext():
        _refuse_unquantizable_experts(experts)


# ----------------------------------------------------------------------- _seed_parallel_states


def test_expert_containers_and_their_children_get_the_expert_state():
    """Everything else, the root and the containers' own parents included, gets the dense state; so
    does a sibling whose name merely starts with a container's (Qwen2-MoE keeps
    ``shared_expert_gate`` beside ``shared_expert``)."""
    model = _model_with_decoder_at("model.")
    model.model.layers[0].mlp.experts_gate = nn.Linear(2, 1)
    experts = {f"model.layers.{i}.mlp.experts": model.model.layers[i].mlp.experts for i in (0, 1)}
    dense, expert = object(), object()

    _seed_parallel_states(model, experts, dense_state=dense, expert_state=expert)

    states = {name: module._parallel_state for name, module in model.named_modules()}
    assert {name for name, state in states.items() if state is expert} == {
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.experts.weight_quantizer",
        "model.layers.1.mlp.experts",
        "model.layers.1.mlp.experts.weight_quantizer",
    }
    assert all(state is dense for state in states.values() if state is not expert)


# ------------------------------------------------------------------------ expert_parallel_mesh


@pytest.mark.parametrize(
    ("plan", "is_expert_parallel"),
    [(DistributionPlan(2, ep_size=2), True), (None, False), (DistributionPlan(2), False)],
    ids=["expert-parallel", "no-plan", "replicated"],
)
def test_the_expert_parallel_mesh_comes_with_an_expert_parallel_plan_only(plan, is_expert_parallel):
    """An expert-parallel plan: the mesh transformers kept on the model. Any other model has none,
    even with a device mesh on it, as a load through from_pretrained's own TP can leave."""
    model = _model_with_decoder_at("model.")
    if plan is not None:
        set_distribution_plan(model, plan)
    model._device_mesh = mesh = object()

    assert expert_parallel_mesh(model) is (mesh if is_expert_parallel else None)


def test_an_expert_parallel_model_without_its_mesh_is_refused():
    """A model recorded expert-parallel that load_and_prepare_ep did not load has no mesh for
    calibration to re-install the all-to-all forwards over; it is refused by name."""
    model = _model_with_decoder_at("model.")
    set_distribution_plan(model, DistributionPlan(2, ep_size=2))

    with pytest.raises(RuntimeError, match="carries no device mesh"):
        expert_parallel_mesh(model)


# --------------------------------------------------------------------------- _ckpt_quant_method


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(quantization_config=_BLOCK_FP8), "fp8"),
        (SimpleNamespace(quantization_config=SimpleNamespace(quant_method="mxfp4")), "mxfp4"),
        (SimpleNamespace(text_config=SimpleNamespace(quantization_config=_BLOCK_FP8)), "fp8"),
        (SimpleNamespace(llm_config=SimpleNamespace(quantization_config=_BLOCK_FP8)), "fp8"),
        (
            SimpleNamespace(
                quantization_config={"quant_method": "mxfp4"},
                text_config=SimpleNamespace(quantization_config={"quant_method": "fp8"}),
            ),
            "mxfp4",
        ),
        (SimpleNamespace(quantization_config=FineGrainedFP8Config()), "fp8"),
        (
            SimpleNamespace(
                quantization_config={"quant_method": "fp8", "activation_scheme": "static"}
            ),
            "fp8-per-tensor",
        ),
        (SimpleNamespace(quantization_config={"weight_block_size": [128, 128]}), None),
        (SimpleNamespace(text_config=None), None),
    ],
    ids=[
        "dict",
        "object",
        "vlm-text-config",
        "vlm-llm-config",
        "top-level-wins",
        "transformers-config-object",
        "per-tensor-fp8",
        "no-method",
        "unquantized",
    ],
)
def test_quant_method_comes_from_the_config_or_a_vlms_text_config(config, expected):
    """The top-level config first, else a VLM's text or LLM config. A loaded model's config holds
    transformers' config object, not the JSON dict, and callers compare against ``"fp8"`` /
    ``"mxfp4"``: it reports the plain method name, not its enum. ``"fp8"`` is block FP8 only:
    per-tensor FP8 names the same method without a block size."""
    assert _ckpt_quant_method(config) == expected


def test_from_pretrained_builds_its_mesh_from_the_ep_mesh_it_is_given():
    """Newer transformers ignores ``from_pretrained``'s ``device_mesh`` and builds a world-sized one,
    which shards the experts over every rank when EP is below the world size. During the load it gets
    the EP mesh instead, and its own builder is back afterwards, even when the load raises."""
    pytest.importorskip("transformers")
    mixin = _transformers_mesh_builder()
    if mixin is None:
        pytest.skip("this transformers honors from_pretrained's device_mesh")
    from transformers import PreTrainedModel
    from transformers.distributed import DistributedConfig

    builder = mixin.initialize_distributed_mesh
    ep_mesh = SimpleNamespace(device_type="cpu")
    config = DistributedConfig(enable_expert_parallel=True, tp_plan="auto", tp_size=2)
    with _experts_sharded_over(ep_mesh):
        _, device_map, device_mesh = PreTrainedModel.prepare_distribute_model(config)

    assert device_mesh is ep_mesh
    assert device_map == torch.device("cpu")
    assert mixin.initialize_distributed_mesh is builder
    with pytest.raises(RuntimeError), _experts_sharded_over(ep_mesh):
        raise RuntimeError("the load failed")
    assert mixin.initialize_distributed_mesh is builder


@pytest.mark.parametrize(
    ("fields", "builds_its_mesh", "config", "tp_plan"),
    [
        # transformers 5.6-5.9, 5.11, 5.12
        (["enable_expert_parallel"], False, {"enable_expert_parallel": True}, "auto"),
        # 5.10, 5.13-5.17
        (
            ["tp_size", "tp_plan", "enable_expert_parallel"],
            False,
            {"enable_expert_parallel": True, "tp_plan": "auto"},
            None,
        ),
        # 5.18 on
        (
            ["tp_size", "tp_plan", "enable_expert_parallel"],
            True,
            {"enable_expert_parallel": True, "tp_plan": "auto", "tp_size": 2},
            None,
        ),
        # no release so far
        (
            ["tp_size", "enable_expert_parallel"],
            True,
            {"enable_expert_parallel": True, "tp_size": 2},
            "auto",
        ),
    ],
    ids=["no-tp-plan-field", "tp-plan-field", "own-mesh-builder", "own-mesh-builder-no-tp-plan"],
)
def test_each_load_argument_goes_where_this_transformers_takes_it(
    ep_world_of_two, monkeypatch, fields, builds_its_mesh, config, tp_plan
):
    """``tp_plan="auto"`` goes on ``DistributedConfig`` where it has the field, else to
    ``from_pretrained``, never both; a release that builds its own mesh gets it sized to the EP
    degree. Each is decided by its own field, so a release with one and not the other loads too."""

    class _LoadedError(Exception):
        """Stops the load at from_pretrained."""

    loaded = {}

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            loaded.update(kwargs)
            raise _LoadedError

    import transformers.distributed as hf_distributed

    distributed_config = dataclasses.make_dataclass(
        "DistributedConfig", [(name, object, dataclasses.field(default=None)) for name in fields]
    )
    monkeypatch.setattr(hf_distributed, "DistributedConfig", distributed_config)
    monkeypatch.setattr(
        model_load_utils,
        "_transformers_mesh_builder",
        lambda: SimpleNamespace(initialize_distributed_mesh=None) if builds_its_mesh else None,
    )
    monkeypatch.setattr(
        "torch.distributed.device_mesh.init_device_mesh", lambda *args, **kwargs: _Mesh()
    )
    monkeypatch.setattr(model_load_utils, "_auto_class", lambda *args, **kwargs: _Loader)

    with pytest.raises(_LoadedError):
        load_and_prepare_ep(
            "org/model",
            DistributionPlan(2, ep_size=2),
            hf_config=SimpleNamespace(base_model_ep_plan=_EP_PLAN),
        )

    given = {k: v for k, v in vars(loaded["distributed_config"]).items() if v is not None}
    assert given == config
    assert loaded.get("tp_plan") == tp_plan
