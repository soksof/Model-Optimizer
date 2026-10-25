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

"""Expert-parallel PTQ wiring in ``examples/hf_ptq``.

hf_ptq.py decides a distribution plan once, with ``resolve_distribution_plan``, and loads the model
by it through ``load_pretrained``: expert-parallel, FSDP2-sharded under ``--use_fsdp2``, or whole
through ``get_model``. Everything after reads the plan back from the model: every rank of a
distributed plan calibrates on its own shard of the data, an expert-parallel one through the
distributed forward loop, and a whole model is pinned to the rank's own GPU instead of ``"auto"``.

Nothing here needs a checkpoint, and only the test of a VLM's all-to-all forwards needs a process
group, of one gloo rank: elsewhere the process group's rank and size are answered the way a torchrun
launch would, the configs are tiny real transformers configs served from memory, and the loaders are
replaced by stand-ins that record how they were called.
"""

import contextlib
import importlib
import inspect
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoConfig, DeepseekV3Config, LlamaConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.models import get_spec
from modelopt.torch.utils.distributed import DistributionPlan, ParallelState, set_distribution_plan
from modelopt.torch.utils.plugins import model_load_utils
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"

# Small enough that the dense part fits any GPU, so "auto" dense sharding always resolves to
# replication whether or not the machine running the tests has one.
_TINY = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
}

_CKPT = "/ckpt/model"


class _LoaderReachedError(Exception):
    """Raised by a loader stand-in once the call under test has been recorded."""


class _RunFailedError(Exception):
    """Raised by a stand-in for the body of a run that fails."""


@pytest.fixture
def hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


@pytest.fixture(autouse=True)
def _no_distribution_knobs(monkeypatch):
    """A shell that exports the knobs would otherwise steer every decision below."""
    monkeypatch.delenv("MODELOPT_EP_DEGREE", raising=False)
    monkeypatch.delenv("MODELOPT_FSDP_SIZE", raising=False)


@pytest.fixture(autouse=True)
def _checkpoint_is_local(monkeypatch):
    """The checkpoint path names no files; resolving it must not reach the Hub."""
    monkeypatch.setattr(
        model_load_utils, "ensure_local_checkpoint", lambda path, group=None, **_: (None, path)
    )


def _gathered_alike(gathered, mine, group=None):
    """Answer a gather as if every rank handed it what this rank did."""
    gathered[:] = [mine] * len(gathered)


@pytest.fixture
def torchrun(monkeypatch):
    """Run as rank ``rank`` of a ``world_size``-rank torchrun launch; returns its ``dist_state``.

    Every rank acts as its group's first, deciding for itself, and reads the checkpoint as this one
    does; how the ranks agree on what they decide is tested with the collectives.
    """
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "all_gather_object", _gathered_alike)

    def enter(world_size, rank=0):
        monkeypatch.setattr(dist, "is_initialized", lambda: world_size > 1)
        monkeypatch.setattr(dist, "is_distributed_run", lambda: world_size > 1)
        monkeypatch.setattr(dist, "size", lambda group=None: world_size)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world_size)
        monkeypatch.setattr(dist, "is_master", lambda group=None: rank == 0)
        monkeypatch.setattr(dist, "local_rank", lambda: rank)
        return _dist_state(world_size, rank)

    return enter


def _dist_state(world_size, rank=0):
    """What distributed_run yields to ``rank`` of a ``world_size``-rank launch."""
    return dist.DistributedState(
        rank=rank, world_size=world_size, device=torch.device(f"cuda:{rank}")
    )


def _args(dist_state, **overrides):
    """What load_model reads before it picks a loader, as main() leaves it."""
    args = SimpleNamespace(
        dist_state=dist_state,
        use_fsdp2=False,
        recipe=None,
        local_checkpoint_path=_CKPT,
        trust_remote_code=False,
        attn_implementation=None,
        sparsity_fmt="dense",
        qformat="fp8",
        low_memory_mode=False,
        use_seq_device_map=False,
        offload_folder=None,
        vllm_fakequant_export=False,
    )
    vars(args).update(overrides)
    return args


def _serve(monkeypatch, hf_ptq, config):
    """Answer every config read with ``config`` rather than a checkpoint on disk."""
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    return config


def _deepseek_v3(n_routed_experts):
    """DeepSeek-V3 without a plan of its own, so only its model spec can make it EP-capable.

    Its one layer is an MoE layer: DeepSeek-V3's first layers are dense by default.
    """
    config = DeepseekV3Config(
        n_routed_experts=n_routed_experts,
        first_k_dense_replace=0,
        moe_intermediate_size=8,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        **_TINY,
    )
    config.base_model_ep_plan = None  # holds even if a later transformers ships a plan
    return config


# -------------------------------------------------------------------------------- the decision


def test_use_fsdp2_plans_fsdp2_without_expert_parallelism(hf_ptq, monkeypatch, torchrun):
    """--use_fsdp2 names its load, so even an explicit EP degree for an EP-capable MoE must not
    divert the run to the expert-parallel one."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", "2")
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))
    args = _args(torchrun(world_size=2), use_fsdp2=True)

    assert hf_ptq._resolve_distribution(args)[0] == DistributionPlan(2, dense="fsdp")


def test_dense_sharding_without_expert_parallelism_needs_use_fsdp2(hf_ptq, monkeypatch, torchrun):
    """Without expert parallelism a plan that shards the dense part is an FSDP2 run, which this
    script runs only under --use_fsdp2, the flag its FSDP2 checks hang off; so it must fail and point
    there rather than load the model whole on every rank."""
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    _serve(monkeypatch, hf_ptq, LlamaConfig(**_TINY))

    with pytest.raises(RuntimeError, match="--use_fsdp2"):
        hf_ptq._resolve_distribution(_args(torchrun(world_size=2)))


@pytest.mark.parametrize("ep_degree", [2, 4])
def test_dense_fsdp2_within_expert_data_parallel_replicas_is_refused(
    hf_ptq, monkeypatch, torchrun, ep_degree
):
    """Below the world size each expert-parallel group is a replica, and FSDP2 shards the dense part
    within each: a layout the export cannot write, so it must fail before the load and the
    calibration rather than at the end. Expert parallelism over the whole world can still shard it."""
    monkeypatch.setenv("MODELOPT_EP_DEGREE", str(ep_degree))
    monkeypatch.setenv("MODELOPT_FSDP_SIZE", "world")
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))
    args = _args(torchrun(world_size=4))

    if ep_degree == 4:
        assert hf_ptq._resolve_distribution(args)[0] == DistributionPlan(4, ep_size=4, dense="fsdp")
    else:
        with pytest.raises(NotImplementedError, match=r"MODELOPT_EP_DEGREE.*MODELOPT_FSDP_SIZE=1"):
            hf_ptq._resolve_distribution(args)


@pytest.mark.parametrize(
    ("world_size", "options"),
    [
        (2, {"qformat": "int8_smoothquant"}),
        (2, {"sparsity_fmt": "sparsegpt"}),
        (1, {"sparsity_fmt": "sparsegpt"}),
    ],
    ids=["int8_smoothquant", "sparsity", "single_process"],
)
def test_a_tensorrt_llm_checkpoint_export_needs_a_single_process(
    hf_ptq, monkeypatch, torchrun, world_size, options
):
    """That exporter reads the process count as Megatron's training tensor parallelism: launched
    with torchrun, it hangs in a barrier on rank 0 alone, and on every rank it merges replicas as
    shards. So it fails before the load and the calibration; a single process exports as before."""
    _serve(monkeypatch, hf_ptq, LlamaConfig(**_TINY))
    args = _args(torchrun(world_size=world_size), **options)

    if world_size == 1:
        assert hf_ptq._resolve_distribution(args)[0] == DistributionPlan(1)
    else:
        with pytest.raises(NotImplementedError, match="single process"):
            hf_ptq._resolve_distribution(args)


@pytest.mark.parametrize(
    ("world_size", "env", "use_fsdp2"),
    [(2, {"MODELOPT_EP_DEGREE": "2"}, False), (2, {}, False), (2, {}, True), (1, {}, False)],
    ids=["expert_parallel", "replicated", "use_fsdp2", "single_process"],
)
def test_an_auto_quantize_recipe_is_refused_on_the_routes_this_script_adds(
    hf_ptq, monkeypatch, torchrun, world_size, env, use_fsdp2
):
    """AutoQuantize scores each rank's own calibration shard and reduces no score across ranks, so
    the ranks could choose different formats per layer. An expert-parallel or replicated run refuses
    the recipe before the load; --use_fsdp2 keeps main's warning, and a single process runs it."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _serve(monkeypatch, hf_ptq, _deepseek_v3(n_routed_experts=8))
    args = _args(
        torchrun(world_size=world_size),
        use_fsdp2=use_fsdp2,
        recipe="general/auto_quantize/nvfp4_fp8_kl_div_at_5p4bits",
    )

    if world_size == 1 or use_fsdp2:
        hf_ptq._resolve_distribution(args)
    else:
        with pytest.raises(NotImplementedError, match="AutoQuantize"):
            hf_ptq._resolve_distribution(args)


@pytest.mark.parametrize(
    "option",
    [{"low_memory_mode": True}, {"use_seq_device_map": True}, {"offload_folder": "/offload"}],
    ids=["low_memory_mode", "use_seq_device_map", "offload_folder"],
)
@pytest.mark.parametrize(
    ("make_config", "world_size"),
    [
        (lambda: _deepseek_v3(n_routed_experts=8), 2),
        (lambda: LlamaConfig(**_TINY), 2),
        (lambda: LlamaConfig(**_TINY), 1),
    ],
    ids=["expert_parallel", "replicated", "single_process"],
)
def test_the_single_process_load_options_are_refused_on_the_routes_this_script_adds(
    hf_ptq, monkeypatch, torchrun, make_config, world_size, option
):
    """Each rank of an expert-parallel or replicated run loads its own share or copy of the model
    onto its own GPU, where these options of a single process's load would be ignored or would break
    the load. Both routes refuse them before the load; a single process loads by them as before."""
    _serve(monkeypatch, hf_ptq, make_config())
    args = _args(torchrun(world_size=world_size), **option)

    if world_size == 1:
        assert hf_ptq._resolve_distribution(args)[0] == DistributionPlan(1)
    else:
        (name,) = option
        with pytest.raises(NotImplementedError, match=f"--{name}"):
            hf_ptq._resolve_distribution(args)


@pytest.mark.parametrize(
    ("make_config", "world_size", "refused"),
    [
        (lambda: _deepseek_v3(n_routed_experts=8), 2, True),
        (lambda: LlamaConfig(**_TINY), 2, False),
        (lambda: LlamaConfig(**_TINY), 1, False),
    ],
    ids=["expert_parallel", "replicated", "single_process"],
)
def test_the_vllm_fakequant_export_is_refused_for_a_sharded_model(
    hf_ptq, monkeypatch, torchrun, make_config, world_size, refused
):
    """That exporter writes the whole model with save_pretrained from each process it runs in, so
    every rank of an expert-parallel run would write its own experts into the same directory. Such a
    run refuses it before the load; rank 0 alone writes a replicated model, and a single process
    exports as before."""
    _serve(monkeypatch, hf_ptq, make_config())
    args = _args(torchrun(world_size=world_size), vllm_fakequant_export=True)

    if refused:
        with pytest.raises(NotImplementedError, match="--vllm_fakequant_export"):
            hf_ptq._resolve_distribution(args)
    else:
        hf_ptq._resolve_distribution(args)


# ---------------------------------------------------------------------------------- load_model


def _record_load_pretrained(hf_ptq, monkeypatch):
    """Replace hf_ptq's load_pretrained and every other loader; return the recorded call."""
    calls = []

    def load_pretrained_stub(*args, **kwargs):
        calls.append(inspect.signature(load_pretrained).bind(*args, **kwargs).arguments)
        raise _LoaderReachedError

    monkeypatch.setattr(hf_ptq, "load_pretrained", load_pretrained_stub)
    monkeypatch.setattr(hf_ptq, "get_model", lambda *a, **k: pytest.fail("whole-model loader ran"))
    return calls


@pytest.mark.parametrize(
    ("make_config", "knobs", "world_size", "options", "plan", "passed"),
    [
        pytest.param(
            lambda: _deepseek_v3(n_routed_experts=8),
            {"MODELOPT_FSDP_SIZE": "world"},
            4,
            {"trust_remote_code": True, "attn_implementation": "eager"},
            DistributionPlan(4, ep_size=4, dense="fsdp"),
            {"trust_remote_code": True, "attn_implementation": "eager"},
            id="expert_parallel",
        ),
        pytest.param(
            lambda: LlamaConfig(architectures=["LlamaForCausalLM"], **_TINY),
            {},
            2,
            {
                "use_fsdp2": True,
                "cpu_offload": True,
                "specdec_offline_dataset": None,
                "low_memory_mode": False,
            },
            DistributionPlan(2, dense="fsdp"),
            {"trust_remote_code": False, "cpu_offload": True, "attn_implementation": None},
            id="use_fsdp2",
        ),
    ],
)
def test_load_model_hands_load_pretrained_the_plan_and_the_config_it_was_decided_on(
    hf_ptq, monkeypatch, torchrun, make_config, knobs, world_size, options, plan, passed
):
    """With an EP plan, or under --use_fsdp2, load_model hands load_pretrained the plan -- the
    decided degree and dense sharding -- the config it was decided on, spec-prepared for EP, and
    the caller's options, such as --use_fsdp2's CPU-offload choice."""
    for name, value in knobs.items():
        monkeypatch.setenv(name, value)
    served = _serve(monkeypatch, hf_ptq, make_config())
    calls = _record_load_pretrained(hf_ptq, monkeypatch)

    with pytest.raises(_LoaderReachedError):
        hf_ptq.load_model(_args(torchrun(world_size=world_size), **options))

    (call,) = calls
    assert call["ckpt_path"] == _CKPT
    assert call["distribution"] == plan
    hf_config = call["kwargs"].pop("hf_config")
    assert call["kwargs"] == passed
    if plan.is_expert_parallel:
        # The spec's plan goes on the decision's own copy of the config.
        assert hf_config.base_model_ep_plan == get_spec("deepseek_v3").ep_spec.base_model_ep_plan
        assert served.base_model_ep_plan is None
    else:
        assert hf_config is served


def test_a_whole_model_loads_by_the_plan_through_get_model(hf_ptq, monkeypatch, torchrun):
    """Data-parallel, every rank loads the whole model, by the plan, onto its own GPU."""
    _serve(monkeypatch, hf_ptq, LlamaConfig(**_TINY))
    loads = []

    def get_model(ckpt_path, device, **kwargs):
        loads.append((device, kwargs["distribution"]))
        raise _LoaderReachedError

    monkeypatch.setattr(hf_ptq, "get_model", get_model)
    args = _args(
        torchrun(world_size=4, rank=1),
        specdec_offline_dataset=None,
        low_memory_mode=False,
        gpu_max_mem_percentage=0.8,
        use_seq_device_map=False,
        offload_folder=None,
        max_cpu_memory_gb=None,
        max_gpu_memory_gb=None,
    )

    with pytest.raises(_LoaderReachedError):
        hf_ptq.load_model(args)

    assert loads == [(torch.device("cuda:1"), DistributionPlan(4))]


# ------------------------------------------------------------------------- per-rank placement


@pytest.mark.parametrize(
    ("device", "world_size", "device_map"),
    [
        ("cuda:1", 4, {"": "cuda:1"}),
        (torch.device("cuda:1"), 4, {"": "cuda:1"}),
        ("cuda:1", 1, "auto"),
    ],
    ids=["str", "torch_device", "single_process"],
)
def test_a_rank_device_pins_the_whole_model_to_it(
    monkeypatch, tmp_path, device, world_size, device_map
):
    """Under torchrun each rank asks for its own cuda:<local_rank>; "auto" would let every rank's
    accelerate pick the same idle GPU. A single process spreads the model over the GPUs, whatever
    device it names. Only a spread model is held to a memory limit sized over the node's GPUs: the
    pinned map would ignore one."""
    LlamaConfig(architectures=["LlamaForCausalLM"], **_TINY).save_pretrained(tmp_path)
    # No GPU on the node holds the model, so sizing it over the node limits its memory.
    monkeypatch.setattr(example_utils, "get_max_memory", lambda: {0: 0, "cpu": 1 << 40})
    placed = []

    plan = DistributionPlan(world_size)

    def load_pretrained(ckpt_path, auto_class, **kwargs):
        assert kwargs["distribution"] == plan
        placed.append((kwargs["device_map"], "max_memory" in kwargs))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(example_utils, "load_pretrained", load_pretrained)

    example_utils.get_model(str(tmp_path), device=device, distribution=plan)

    assert placed == [(device_map, device_map == "auto")]


@pytest.mark.parametrize(
    "checkpoint",
    ["pack-quantized", "mxfp4", "diffusion-gemma"],
)
def test_every_load_path_keeps_the_rank_pin(monkeypatch, tmp_path, checkpoint):
    """A pack-quantized or MXFP4 checkpoint loads through a path of its own, and DiffusionGemma asks
    for a sequential map on a multi-GPU node; each would otherwise place every rank's copy over the
    same GPUs."""
    quantization_config = {
        "pack-quantized": {"quant_method": "compressed-tensors", "format": "pack-quantized"},
        "mxfp4": {"quant_method": "mxfp4"},
    }.get(checkpoint)
    LlamaConfig(
        architectures=["LlamaForCausalLM"], quantization_config=quantization_config, **_TINY
    ).save_pretrained(tmp_path)
    if checkpoint == "diffusion-gemma":
        monkeypatch.setattr(example_utils, "is_diffusion_gemma", lambda config: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(example_utils, "_unpack_compressed_linear_weights", lambda *args: None)
    placed = []

    def load_pretrained(ckpt_path, auto_class, **kwargs):
        placed.append(kwargs["device_map"])
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(example_utils, "load_pretrained", load_pretrained)

    example_utils.get_model(str(tmp_path), device="cuda:1", distribution=DistributionPlan(4))

    assert placed == [{"": "cuda:1"}]


@pytest.mark.parametrize(
    ("device", "kwargs", "match"),
    [
        ("cuda:1", {"use_seq_device_map": True}, "--use_seq_device_map"),
        # "auto" would let every rank's accelerate pick the same idle GPU.
        ("cuda", {}, "names no GPU"),
    ],
    ids=["sequential_map", "unindexed_device"],
)
def test_a_placement_over_shared_gpus_is_refused_where_each_rank_holds_its_own_copy(
    tmp_path, device, kwargs, match
):
    LlamaConfig(architectures=["LlamaForCausalLM"], **_TINY).save_pretrained(tmp_path)

    with pytest.raises(ValueError, match=match):
        example_utils.get_model(
            str(tmp_path), device=device, distribution=DistributionPlan(4), **kwargs
        )


# ------------------------------------------------------------------ process group and dist_state


def test_main_owns_the_process_group_around_the_whole_run(hf_ptq, monkeypatch):
    """main() runs inside distributed_run, with a collective timeout above NCCL's 30-minute
    default, and leaves it even when the run raises -- or the other ranks would wait in their next
    collective until that timeout. The run reads its rank, size and device from the state
    distributed_run yields, which puts a single process on --device."""
    events, calls, seen = [], [], []
    state = dist.DistributedState(rank=0, world_size=1, device="cpu")

    @contextlib.contextmanager
    def distributed_run(timeout=None, device="cuda"):
        calls.append((timeout, device))
        events.append("enter")
        try:
            yield state
        finally:
            events.append("exit")

    def main_inner(args):
        events.append("run")
        seen.append(args.dist_state)
        raise _RunFailedError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hf_ptq, "distributed_run", distributed_run)
    monkeypatch.setattr(hf_ptq, "_main_inner", main_inner)

    with pytest.raises(_RunFailedError):
        hf_ptq.main(SimpleNamespace(use_fsdp2=False, device="cpu"))

    assert events == ["enter", "run", "exit"]
    timeout, device = calls[0]
    assert timeout > timedelta(minutes=30)
    assert device == "cpu"
    assert len(seen) == 1
    assert seen[0] is state


# --------------------------------------------------------------------------------- calibration


def test_distributed_calibration_gives_each_rank_its_own_shard(hf_ptq, monkeypatch):
    """Every distributed path calibrates data-parallel: the ranks get equal, disjoint shards that
    together cover the calibration set. A single process keeps the loader as built."""
    loader = DataLoader(list(range(8)), batch_size=1)
    monkeypatch.setattr(hf_ptq, "get_dataset_dataloader", lambda **kwargs: loader)
    args = SimpleNamespace(
        specdec_offline_dataset=None,
        calib_with_images=False,
        dataset=["cnn_dailymail"],
        batch_size=1,
        calib_size=[8],
        calib_seq=16,
    )
    tokenizer = get_tiny_tokenizer()

    def calib_loader(world_size, rank=0):
        args.dist_state = _dist_state(world_size, rank)
        model = torch.nn.Linear(2, 2)
        set_distribution_plan(model, DistributionPlan(world_size))
        built, _ = hf_ptq.make_calib_dataloader(
            args, model, None, tokenizer, torch.device("cpu"), "llama"
        )
        return built

    shards = [[int(b) for b in calib_loader(world_size=2, rank=r)] for r in range(2)]

    assert len(shards[0]) == len(shards[1]) == 4
    assert sorted(shards[0] + shards[1]) == list(range(8))
    assert calib_loader(world_size=1) is loader


@pytest.mark.parametrize("images", [False, True], ids=["text", "images"])
@pytest.mark.parametrize(
    ("plan", "loop"),
    [(DistributionPlan(2, ep_size=2), "distributed"), (DistributionPlan(2, dense="fsdp"), "plain")],
    ids=["expert_parallel", "fsdp2"],
)
def test_only_an_expert_parallel_run_calibrates_through_the_distributed_forward_loop(
    hf_ptq, monkeypatch, plan, loop, images
):
    """Only distributed_forward_loop re-plants the all-to-all forwards that quantizer insertion
    clobbers, so an expert-parallel run must calibrate through it, over the mesh load_and_prepare_ep
    loaded the experts over, image batches included: they run through the whole VLM's own forward
    there (Nemotron-H Omni calibrates on images by default). The loop re-plants over the model it is
    handed, the whole model, and text batches run through the language model. Without expert
    parallelism there is nothing to re-plant: FSDP2 keeps the plain loop and its split-on-OOM retry,
    as on main."""
    ep_mesh, vlm_forward = object(), object()
    full_model, language_model = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    set_distribution_plan(full_model, plan)
    chosen, quantized = [], []
    monkeypatch.setattr(hf_ptq, "is_quantized", lambda m: False)
    monkeypatch.setattr(hf_ptq, "need_calibration", lambda cfg: True)
    monkeypatch.setattr(hf_ptq, "expert_parallel_mesh", lambda full_model: ep_mesh)
    monkeypatch.setattr(
        hf_ptq,
        "distributed_forward_loop",
        lambda model, loader, **k: chosen.append((model, k["ep_mesh"], k["forward"])) or object(),
    )
    monkeypatch.setattr(hf_ptq, "create_vlm_calibration_forward", lambda model: vlm_forward)
    monkeypatch.setattr(
        hf_ptq,
        "create_vlm_calibration_loop",
        lambda model, loader: chosen.append("vlm") or object(),
    )
    monkeypatch.setattr(
        hf_ptq, "create_forward_loop", lambda **k: chosen.append("plain") or object()
    )
    monkeypatch.setattr(
        hf_ptq.mtq, "quantize", lambda m, cfg, forward_loop: quantized.append(forward_loop) or m
    )
    args = SimpleNamespace(
        qformat="fp8",
        calib_with_images=images,
        dist_state=_dist_state(world_size=2),
        layerwise_export=False,
        specdec_offline_dataset=None,
    )

    hf_ptq.mono_quantize(
        args,
        {"algorithm": "max"},
        full_model,
        language_model,
        model_type="qwen3_moe",
        calibration_only=False,
        calib_dataloader=object(),
        is_nemotron_vl_model=False,
    )

    assert len(quantized) == 1
    if loop == "plain":
        assert chosen == ["vlm" if images else "plain"]
    elif images:
        assert chosen == [(full_model, ep_mesh, vlm_forward)]
    else:
        ((model, mesh, forward),) = chosen
        assert (model, mesh) == (full_model, ep_mesh)
        x = torch.ones(1, 2)
        assert torch.equal(forward({"input": x}), language_model(x))


@pytest.mark.parametrize(
    "loop_handed", [False, True], ids=["calibration_free", "loop_never_called"]
)
@pytest.mark.parametrize(
    ("plan", "replanted"),
    [(DistributionPlan(2, ep_size=2), True), (DistributionPlan(2, dense="fsdp"), False)],
    ids=["expert_parallel", "fsdp2"],
)
def test_an_expert_parallel_run_replants_the_all_to_all_forwards_after_quantizing(
    hf_ptq, monkeypatch, plan, replanted, loop_handed
):
    """The calibration loop re-plants the all-to-all forwards that quantizer insertion drops only
    when the calibration runs it: a config without static activation scales calibrates without the
    loop, and algorithm None (the MX formats) never calls the loop it is handed. The preview and the
    export forward through the experts either way, so an expert-parallel run re-plants them once
    quantized; without expert parallelism there is nothing to re-plant."""
    ep_mesh, full_model = object(), torch.nn.Linear(2, 2)
    set_distribution_plan(full_model, plan)
    events = []
    monkeypatch.setattr(hf_ptq, "is_quantized", lambda m: False)
    monkeypatch.setattr(hf_ptq, "need_calibration", lambda cfg: loop_handed)
    monkeypatch.setattr(hf_ptq, "expert_parallel_mesh", lambda model: ep_mesh)
    monkeypatch.setattr(hf_ptq, "distributed_forward_loop", lambda *args, **kwargs: object())
    monkeypatch.setattr(hf_ptq, "create_forward_loop", lambda **kwargs: object())
    monkeypatch.setattr(
        hf_ptq.mtq, "quantize", lambda m, cfg, forward_loop: events.append("quantize") or m
    )
    monkeypatch.setattr(
        hf_ptq,
        "_reinstall_a2a_forwards",
        lambda model, mesh: events.append(("replant", model, mesh)),
    )
    args = SimpleNamespace(
        qformat="mxfp8",
        calib_with_images=False,
        dist_state=_dist_state(world_size=2),
        layerwise_export=False,
        specdec_offline_dataset=None,
    )

    hf_ptq.mono_quantize(
        args,
        {"algorithm": None},
        full_model,
        full_model,
        model_type="qwen3_moe",
        calibration_only=False,
        calib_dataloader=object(),
        is_nemotron_vl_model=False,
    )

    assert events == ["quantize"] + ([("replant", full_model, ep_mesh)] if replanted else [])


@pytest.fixture
def gloo_world_of_one(tmp_path):
    """A one-rank gloo process group, for the all-to-all forwards to run on."""
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1
    )
    yield torch.distributed.group.WORLD
    torch.distributed.destroy_process_group()


class _EpMesh:
    """The expert-parallel submesh of a two-rank plan, over this process's one-rank group."""

    def size(self):
        return 2

    def get_group(self):
        return torch.distributed.group.WORLD


def test_a_vlm_calibrated_on_text_gets_its_experts_all_to_all_forwards_back(
    hf_ptq, gloo_world_of_one
):
    """Text batches run through a VLM's language model, but transformers keeps the expert-parallel
    plan on the model from_pretrained built, and the language model's own plan names no experts.
    Re-planted over the language model, the experts' all-to-all forward that quantizer insertion
    drops (transformers 5.16 and later) stayed dropped, and the first batch raised for the missing
    EP process group."""
    pytest.importorskip("transformers", minversion="5.16")
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration
    from transformers.distributed.configuration_utils import DistributedConfig
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, _get_parameter_tp_plan

    from modelopt.torch.utils.moe import register_torch_a2a

    register_torch_a2a()
    config = Qwen3VLMoeConfig(
        text_config={
            **_TINY,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "decoder_sparse_step": 1,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "deepstack_visual_indexes": [],
        },
    )
    # The model as load_and_prepare_ep leaves it: its plan's all-to-all styles on the language
    # model's config and expert parallelism on the top one, the forwards from_pretrained installs
    # for them, the experts dispatching through the all-to-all, and their seeded parallel state.
    config.text_config.base_model_ep_plan = {
        "layers.*.mlp.gate": "torch_a2a_router",
        "layers.*.mlp.experts": "torch_a2a_experts",
    }
    config.distributed_config = DistributedConfig(enable_expert_parallel=True)
    model = Qwen3VLMoeForConditionalGeneration(config).eval()
    mesh = _EpMesh()
    for name, module in model.named_modules():
        style = _get_parameter_tp_plan(parameter_name=name, tp_plan=model.tp_plan, is_weight=False)
        if style in ("torch_a2a_router", "torch_a2a_experts"):
            ALL_PARALLEL_STYLES[style].install_forward(module, mesh)
    model.config.text_config._experts_implementation = "torch_a2a"
    model._device_mesh = mesh
    set_distribution_plan(model, DistributionPlan(2, ep_size=2))
    model_load_utils._seed_parallel_states(
        model,
        [name for name, _ in model.named_modules() if name.endswith("mlp.experts")],
        dense_state=ParallelState(data_parallel_group=-1),
        expert_state=ParallelState(
            data_parallel_group=-1, expert_model_parallel_group=gloo_world_of_one
        ),
    )
    language_model = model.model.language_model
    args = SimpleNamespace(
        qformat="int8",
        calib_with_images=False,
        dist_state=dist.DistributedState(rank=0, world_size=2, device=torch.device("cpu")),
        layerwise_export=False,
        specdec_offline_dataset=None,
    )
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4)}

    hf_ptq.mono_quantize(
        args,
        hf_ptq.mtq.INT8_DEFAULT_CFG,
        model,
        language_model,
        model_type="qwen3_vl_moe",
        calibration_only=False,
        calib_dataloader=[batch],
        is_nemotron_vl_model=False,
    )

    assert language_model.layers[0].mlp.experts.down_proj_input_quantizer.amax is not None


class _PreviewModel(torch.nn.Module):
    """A model whose ``generate`` records the kwargs each preview passes it, and how often it entered
    the decode hook that Transformers' ``generate`` enters to switch a MoE's experts."""

    def __init__(self, plan=None):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False, model_type="qwen3_moe")
        self.calls = []
        self.switches = 0
        if plan is not None:
            set_distribution_plan(self, plan)

    @contextlib.contextmanager
    def _optimize_model_for_decode(self):
        self.switches += 1
        yield

    def generate(self, input_ids, **kwargs):
        self.calls.append(kwargs)
        with self._optimize_model_for_decode():
            return torch.cat([input_ids, input_ids[:, :1]], dim=1)


_PREVIEW_PLANS = pytest.mark.parametrize(
    ("plan", "lockstep"),
    [
        (None, False),
        (DistributionPlan(4), False),
        (DistributionPlan(4, ep_size=4), True),
        (DistributionPlan(4, dense="fsdp"), True),
        (DistributionPlan(1, dense="fsdp"), False),
    ],
    ids=["single_process", "replicated", "expert_parallel", "fsdp2", "one_rank_fsdp2"],
)


@_PREVIEW_PLANS
def test_a_sharded_model_previews_in_lockstep(hf_ptq, monkeypatch, plan, lockstep):
    """Each rank previews its own calibration shard's first sample, so the ranks finish decoding at
    different steps. A sharded model runs a collective in every forward, and a rank that stopped
    early left the others blocked in it: Qwen3-30B-A3B NVFP4 at EP4 hung in the post-PTQ preview.
    Both previews keep every rank of a sharded model decoding until all have finished."""
    model = _PreviewModel(plan)
    loader = [{"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}]
    args = SimpleNamespace(
        specdec_offline_dataset=None,
        skip_generate=False,
        verbose=False,
        local_checkpoint_path=_CKPT,
        trust_remote_code=False,
    )
    monkeypatch.setattr(hf_ptq, "export_quantized", lambda *args, **kwargs: None)

    ids, mask, before = hf_ptq.pre_quantize(args, model, "qwen3_moe", None, loader, False)
    hf_ptq.post_quantize(
        args,
        model,
        model,
        "qwen3_moe",
        get_tiny_tokenizer(),
        None,
        ids,
        mask,
        before,
        False,
        None,
        loader,
    )

    synced = [call.get("synced_gpus", False) for call in model.calls]
    assert synced == [lockstep, lockstep]  # before and after quantization
    assert model.switches == 0  # neither preview lets generate move the experts off their backend


@_PREVIEW_PLANS
def test_a_sharded_nemotron_vl_model_previews_in_lockstep(monkeypatch, plan, lockstep):
    """The Nemotron-VL preview's text, fallback and image generations keep a sharded model's ranks
    decoding together too."""
    recorded = {}
    keeps = lambda model: "_optimize_model_for_decode" in vars(model)  # noqa: E731
    vlm_utils = SimpleNamespace(
        run_text_only_generation=lambda model, tokenizer, question, config, *args: recorded.update(
            text=config, text_keeps_the_experts=keeps(model)
        ),
        run_vl_preview_generation=lambda model, tokenizer, path, stage, trust, kwargs: (
            recorded.update(image=kwargs, image_keeps_the_experts=keeps(model))
        ),
    )
    monkeypatch.setitem(sys.modules, "vlm_utils", vlm_utils)
    model = _PreviewModel(plan)

    example_utils.run_nemotron_vl_preview(
        model, get_tiny_tokenizer(), torch.tensor([[1, 2, 3]]), _CKPT, "stage", allow_fallback=True
    )

    expected = {"synced_gpus": True} if lockstep else {}
    assert recorded["text"].get("synced_gpus", False) == lockstep
    assert recorded["image"] == expected
    assert recorded["text_keeps_the_experts"] and recorded["image_keeps_the_experts"]
    assert "_optimize_model_for_decode" not in vars(model)
    assert model.calls[0].get("synced_gpus", False) == lockstep  # the fallback generate


def test_the_offline_specdec_loader_is_split_across_ranks_too(hf_ptq):
    """It wraps its DataLoader to move each batch onto the device; each rank still calibrates on
    its own share, as it would from the bare loader."""

    def loader():
        return DataLoader(list(range(8)), batch_size=1, collate_fn=lambda b: {"x": torch.tensor(b)})

    wrapped = hf_ptq._DeviceDataLoader(loader(), torch.device("cpu"))

    shard = hf_ptq._shard_calib_dataloader(wrapped, 1, 2)

    assert isinstance(shard, hf_ptq._DeviceDataLoader)
    assert [b["x"].tolist() for b in shard] == [
        b["x"].tolist() for b in dist.shard_dataloader(loader(), 1, 2)
    ]
    with pytest.raises(NotImplementedError, match="cannot be split"):
        hf_ptq._shard_calib_dataloader(iter(loader()), 1, 2)


@pytest.mark.parametrize(
    ("plan", "nemotron_vl", "raises"),
    [
        (DistributionPlan(4), False, False),
        (DistributionPlan(4, dense="fsdp"), False, False),
        (DistributionPlan(4, ep_size=4), False, True),
        (DistributionPlan(4, ep_size=4), True, True),
    ],
    ids=["replicated", "fsdp2", "expert_parallel", "expert_parallel_nemotron_vl"],
)
def test_a_failed_preview_of_an_expert_parallel_model_stops_the_run(
    hf_ptq, monkeypatch, plan, nemotron_vl, raises
):
    """A rank that skipped an expert-parallel model's preview would leave the others decoding in
    lockstep, blocked in the next step's collective, so the failure is raised, which aborts the run,
    with the preview's error as its cause. A model every rank holds whole keeps the best-effort
    skip and exports, and so does --use_fsdp2, as on main."""

    def fail(*args, **kwargs):
        raise RuntimeError("the preview failed on this rank")

    class _FailingModel(torch.nn.Module):
        generate = fail

    model = _FailingModel()
    set_distribution_plan(model, plan)
    exported = []
    monkeypatch.setattr(hf_ptq, "export_quantized", lambda *args, **kwargs: exported.append(True))
    monkeypatch.setattr(hf_ptq, "run_nemotron_vl_preview", fail)  # as vlm_utils raises in lockstep
    ids = torch.zeros(1, 4, dtype=torch.long)
    args = SimpleNamespace(
        specdec_offline_dataset=None,
        verbose=False,
        local_checkpoint_path=_CKPT,
        trust_remote_code=False,
    )
    tokenizer = SimpleNamespace()
    run = lambda: hf_ptq.post_quantize(  # noqa: E731
        args, model, model, "llama", tokenizer, None, ids, None, ids, nemotron_vl, None, None
    )

    if raises:
        with pytest.raises(RuntimeError) as stopped:
            run()
        assert "failed on this rank" in str(stopped.value.__cause__)
    else:
        with pytest.warns(UserWarning, match="sanity check failed"):
            run()
    assert exported == ([] if raises else [True])


@pytest.mark.parametrize(("lockstep", "raises"), [({}, False), ({"synced_gpus": True}, True)])
def test_a_failed_text_only_vl_preview_stops_a_sharded_run(monkeypatch, lockstep, raises):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    vlm_utils = importlib.import_module("vlm_utils")

    class _ChatModel:
        def chat(self, *args, **kwargs):
            raise RuntimeError("the preview failed on this rank")

    run = lambda: vlm_utils.run_text_only_generation(  # noqa: E731
        _ChatModel(), None, "question", dict(lockstep), _CKPT
    )

    if raises:
        with pytest.raises(RuntimeError, match="failed on this rank"):
            run()
    else:
        assert run() is None


@pytest.mark.parametrize("generation_kwargs", [None, {"synced_gpus": True}])
def test_the_vl_image_preview_generates_with_the_kwargs_it_is_handed(
    monkeypatch, tmp_path, generation_kwargs
):
    """run_vl_preview_generation hands the kwargs it is given to the model's chat(), in its config."""
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    vlm_utils = importlib.import_module("vlm_utils")
    (tmp_path / "images").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "images" / "example1a.jpeg")
    monkeypatch.setattr(vlm_utils, "__file__", str(tmp_path / "vlm_utils.py"))
    monkeypatch.setattr(
        vlm_utils.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: lambda images: {}
    )
    received = []

    class _ChatModel:
        device = torch.device("cpu")

        def chat(self, tokenizer, question, generation_config, **image_features):
            received.append(generation_config)
            return "a picture"

    vlm_utils.run_vl_preview_generation(
        _ChatModel(), SimpleNamespace(eos_token_id=0), _CKPT, "stage", False, generation_kwargs
    )

    assert len(received) == 1
    assert received[0].get("synced_gpus", False) == (generation_kwargs is not None)


@pytest.mark.parametrize("has_image", [True, False], ids=["a_peer_has_none", "no_rank_has_one"])
def test_the_ranks_of_a_sharded_model_skip_the_image_preview_together(
    monkeypatch, tmp_path, has_image
):
    """Each node reads the sample images from its own checkout, which may have none: the repository
    ships none. A rank that previewed alone would wait for its peers in lockstep, so a rank with an
    image skips when a peer has none, and when no rank has one, they all skip without stopping."""
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    vlm_utils = importlib.import_module("vlm_utils")
    if has_image:
        (tmp_path / "images").mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / "images" / "example1a.jpeg")
    monkeypatch.setattr(vlm_utils, "__file__", str(tmp_path / "vlm_utils.py"))
    monkeypatch.setattr(
        vlm_utils.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: lambda images: {}
    )
    # This rank's answer, and a peer's that found no image.
    monkeypatch.setattr(vlm_utils, "all_reduce", lambda label, fn, combine: combine([fn(), False]))

    class _ChatModel:
        device = torch.device("cpu")

        def chat(self, *args, **kwargs):
            pytest.fail("this rank previewed without its peers")

    preview = vlm_utils.run_vl_preview_generation(
        _ChatModel(), SimpleNamespace(eos_token_id=0), _CKPT, "stage", False, {"synced_gpus": True}
    )

    assert preview is None


# ------------------------------------------------------------------------------------- export


@pytest.mark.parametrize(
    ("plan", "rank", "exports"),
    [
        (DistributionPlan(4), 0, True),
        (DistributionPlan(4), 1, False),
        (DistributionPlan(4, dense="fsdp"), 1, True),
        (DistributionPlan(4, ep_size=4), 1, True),
    ],
    ids=["replicated_rank0", "replicated_rank1", "fsdp2_rank1", "expert_parallel_rank1"],
)
def test_only_rank_0_writes_a_replicated_model(hf_ptq, monkeypatch, plan, rank, exports):
    """A replicated model is whole on every rank, whose exports would write one checkpoint into the
    same directory over each other; a sharded one needs every rank for its part. The source config
    is rank 0's either way: another rank's, written after rank 0's export, would drop the
    quantization config."""
    written = []
    monkeypatch.setattr(hf_ptq, "has_spec_opt", lambda model: False)
    monkeypatch.setattr(hf_ptq, "is_multimodal_model", lambda model: True)
    monkeypatch.setattr(hf_ptq, "save_source_config", lambda args, path: written.append("config"))
    monkeypatch.setattr(
        hf_ptq, "export_hf_checkpoint", lambda model, export_dir: written.append("checkpoint")
    )
    args = SimpleNamespace(
        dist_state=_dist_state(plan.world_size, rank),
        export_path="/export",
        layerwise_export=False,
        sparsity_fmt="dense",
        qformat="fp8",
        inference_tensor_parallel=1,
        inference_pipeline_parallel=1,
        vllm_fakequant_export=False,
    )

    model = _PreviewModel(plan)
    hf_ptq.export_quantized(args, model, model, "qwen3_moe")

    assert written == ["config"] * (rank == 0) + ["checkpoint"] * exports


@pytest.mark.parametrize(
    "plan", [DistributionPlan(4, dense="fsdp"), DistributionPlan(4, ep_size=4)], ids=["fsdp2", "ep"]
)
def test_a_sharded_speculative_decoding_model_is_refused_at_export(hf_ptq, monkeypatch, plan):
    """The speculative-decoding exporter writes the whole checkpoint from one process and has no
    rank gate, so every rank of a sharded model would write it over the others: it is refused on
    every rank instead."""
    written = []
    monkeypatch.setattr(hf_ptq, "has_spec_opt", lambda model: True)
    monkeypatch.setattr(
        hf_ptq, "export_speculative_decoding", lambda model, export_dir: written.append(export_dir)
    )
    args = SimpleNamespace(dist_state=_dist_state(plan.world_size, 1), export_path="/export")

    model = _PreviewModel(plan)
    with pytest.raises(NotImplementedError, match="speculative-decoding model sharded"):
        hf_ptq.export_quantized(args, model, model, "qwen3_moe")
    assert written == []
