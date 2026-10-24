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

"""Decompress a compressed-tensors MoE checkpoint's experts as ``from_pretrained`` loads them.

Each rank decompresses only its own experts into the transformers 5.x fused-experts layout, so
expert-parallel PTQ runs on the original compressed checkpoint, with no decompressed copy on disk.
The math is the ``compressed_tensors`` library's, driven by the checkpoint's ``quantization_config``:
``compressor.compression_param_names(scheme)`` names the stored keys of one weight, and
``compressor.decompress(...)["weight"]`` dequantizes them, so every format the library registers
works. It needs ``compressed_tensors`` importable and transformers 5.8 or newer (earlier releases'
``Concatenate`` looks up the sibling keys the decompression consumed).

:func:`install_onfly_decompress` rewrites the per-expert converters ``get_model_conversion_mapping``
returns, for the model built from the config it marks: each ``.weight`` source becomes the stored
keys, and a ``DecompressExperts`` op ahead of the merge dequantizes each expert.
:func:`to_native_backbone_config` builds a VLM's transformers-native backbone config.
"""

import torch


def to_native_backbone_config(
    vlm_config,
    model_type: str = "deepseek_v3",
    architectures: "tuple[str, ...]" = ("DeepseekV3ForCausalLM",),
):
    """Derive the transformers-native fused backbone config from a VLM config's ``text_config``, so
    ``from_pretrained`` builds the native model directly from the ORIGINAL checkpoint's compressed
    shards — NO preprocessed native-config "view" dir.

    A VLM's ``text_config`` is typically the bundled remote-code config (its ``auto_map`` points at a
    ModuleList modeling file, and it carries the stale compressed-tensors ``quantization_config``).
    Dropping ``auto_map`` + ``quantization_config`` and setting ``model_type`` to the native type makes
    it resolve to the stock transformers native config (which, for the a2a path, supplies
    ``base_model_ep_plan``). ``architectures`` must be set to the native class (NOT left None — hf_ptq's
    VL-detection iterates ``config.architectures``).

    Defaults target the Kimi DeepSeek-V3 backbone; pass ``model_type`` / ``architectures`` for another.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    text_config = getattr(vlm_config, "text_config", None)
    if text_config is None:
        raise ValueError("to_native_backbone_config: config has no .text_config (not a VLM?)")
    if model_type not in CONFIG_MAPPING:
        raise ValueError(f"to_native_backbone_config: unknown native model_type {model_type!r}")
    d = text_config.to_dict() if hasattr(text_config, "to_dict") else dict(text_config)
    for key in ("auto_map", "quantization_config", "_name_or_path"):
        d.pop(key, None)
    d["model_type"] = model_type
    d["architectures"] = list(architectures)
    return CONFIG_MAPPING[model_type].from_dict(d)


def _find_quantization_config(hf_config):
    """Locate the compressed-tensors ``quantization_config`` (top-level or under ``text_config``) and
    return it as a plain dict, or ``None`` for a config without one."""
    qc = getattr(hf_config, "quantization_config", None)
    if qc is None:
        tc = getattr(hf_config, "text_config", None)
        qc = getattr(tc, "quantization_config", None) if tc is not None else None
    if qc is None and isinstance(hf_config, dict):
        qc = hf_config.get("quantization_config") or (hf_config.get("text_config") or {}).get(
            "quantization_config"
        )
    if qc is None:
        return None
    if hasattr(qc, "to_dict"):
        qc = qc.to_dict()
    elif not isinstance(qc, dict):
        qc = dict(qc)
    return qc


def _read_compression(hf_config):
    """Parse the checkpoint's compressed-tensors ``quantization_config`` and return
    ``(compressor, scheme, param_names, format_str)`` via the ``compressed_tensors`` registry.

    ``compressor`` is the registered compressor for ``format`` (its ``decompress`` classmethod is the
    dequant); ``scheme`` is the ``QuantizationScheme`` (supplies ``scheme.weights`` to ``decompress``);
    ``param_names`` is the exact set of stored sibling keys per weight. The experts' group is chosen as
    the sole config group, else the one whose ``targets`` reference "expert", else the first.
    ``None`` for a config with no ``quantization_config``: nothing to decompress."""
    # Look the config up first: with none, compressed_tensors need not even be installed. A config
    # that fails to parse raises pydantic's ValidationError, a ValueError of its own.
    quantization_config = _find_quantization_config(hf_config)
    if quantization_config is None:
        return None
    from compressed_tensors.compressors.base import BaseCompressor
    from compressed_tensors.quantization import QuantizationConfig

    qconfig = QuantizationConfig.model_validate(quantization_config)
    groups = qconfig.config_groups
    if len(groups) == 1:
        scheme = next(iter(groups.values()))
    else:
        scheme = next(
            (
                g
                for g in groups.values()
                if any("expert" in str(t).lower() for t in (getattr(g, "targets", None) or []))
            ),
            next(iter(groups.values())),
        )
    fmt = qconfig.format
    fmt = fmt.value if hasattr(fmt, "value") else str(fmt)
    compressor = BaseCompressor.get_value_from_registry(fmt)
    param_names = tuple(compressor.compression_param_names(scheme))
    return compressor, scheme, param_names, fmt


def _make_op_class(compressor, scheme, param_names):
    """Build the per-expert decompress op, bound to a compressor, its scheme and the stored keys.

    The first stored key is the anchor the decompressed experts are emitted under, in place of the
    ``.weight`` the downstream merge expects. The op subclasses ``MergeModulelist`` only because the
    loader shards experts at load (``tensor_idx``, so non-local experts arrive as ``None``) only when
    a converter's first op is one; the original merge later in the chain still does the stacking.
    """
    from transformers.core_model_loading import MergeModulelist

    anchor = "." + param_names[0]
    others = [(pn, "." + pn) for pn in param_names[1:]]

    class DecompressExperts(MergeModulelist):
        """First op in a fused-experts WeightConverter chain.

        For each anchor source (e.g. ``*.weight_packed``), pair it with its sibling stored params (``*.weight_scale``,
        ``*.weight_shape``, ``*.weight_zero_point``, ``*.weight_global_scale``, ... — whatever the format stores),
        decompress each expert via ``compressed_tensors`` -> bf16, and emit the decompressed list UNDER the anchor key
        (dropping the siblings). The downstream MergeModulelist/Concatenate then fuse as for an uncompressed checkpoint.

        Subclasses ``MergeModulelist`` purely so the loader recognizes ``operations[0]`` and shards experts
        at load (see ``_make_op_class`` docstring); non-local experts arrive as ``None`` (from
        ``GroupedGemmParallel.shard_tensor``) and are skipped, so only this rank's local experts are
        decompressed.
        """

        def __init__(self):
            super().__init__(
                dim=0
            )  # dim unused (convert overridden); needed so isinstance -> shard_index

        @torch.no_grad()
        def convert(self, input_dict, *args, **kwargs):
            out = {}
            for pat in list(input_dict.keys()):
                marker = "$" if pat.endswith("$") else ""
                base = pat[:-1] if marker else pat
                if not base.endswith(anchor):
                    continue
                stem = base[: -len(anchor)]
                lists = {param_names[0]: input_dict.pop(pat)}
                # Pop each sibling if PRESENT; a checkpoint may store only a SUBSET of the scheme's
                # possible params (e.g. mxfp4-pack-quantized declares weight_global_scale via
                # compression_param_names, but the public moonshotai/Kimi-K3 checkpoint stores only
                # weight_packed + weight_scale). Missing optional siblings simply aren't passed to the
                # compressor (whose decompress uses whatever the scheme requires).
                for pn, suf in others:
                    sib = input_dict.pop(stem + suf + marker, None)
                    if sib is not None:
                        lists[pn] = sib
                anchor_list = lists[param_names[0]]
                present = [pn for pn in param_names if pn in lists]
                # Non-local experts were sharded away at load (GroupedGemmParallel -> None); decompress
                # only the local experts this rank owns. Keeps peak memory at the per-rank shard.
                out[pat] = [
                    compressor.decompress({pn: lists[pn][i] for pn in present}, scheme)[
                        "weight"
                    ].to(torch.bfloat16)
                    for i in range(len(anchor_list))
                    if anchor_list[i] is not None
                ]
            out.update(input_dict)  # nothing else expected, but pass through defensively
            return out

    return DecompressExperts


def _compressedify(conv, compressor, scheme, param_names):
    """Rewrite a per-expert WeightConverter (.weight sources) into a compressed-aware one.

    each ``.weight`` -> the checkpoint's stored keys ``[.<pn>$ for pn in param_names]`` (anchor first, preserving
    gate/up order so Concatenate fuses gate-then-up), and prepend DecompressExperts.
    """
    from transformers.core_model_loading import WeightConverter

    DecompressExperts = _make_op_class(compressor, scheme, param_names)  # noqa: N806 (a class, keeping its own name)
    by_pname = {pn: [] for pn in param_names}
    for p in conv.source_patterns:
        base = p.removesuffix("$")
        if base.endswith(".weight"):
            stem = base[: -len(".weight")]
            for pn in param_names:
                by_pname[pn].append(stem + "." + pn + "$")
        else:
            by_pname[param_names[0]].append(p)  # leave non-weight sources as-is (none expected)
    new_sources = [s for pn in param_names for s in by_pname[pn]]
    tgt = getattr(conv, "_original_target_patterns", conv.target_patterns)
    new = WeightConverter(
        source_patterns=new_sources,
        target_patterns=tgt,
        operations=[DecompressExperts(), *conv.operations],
    )
    # What else transformers sets on a converter when it maps a model -- the sub-model scope it
    # matches keys under, whether it loads on the CPU -- whichever of them this release has.
    for attr in ("scope_prefix", "base_model_prefix", "force_cpu"):
        if hasattr(conv, attr):
            setattr(new, attr, getattr(conv, attr))
    return new


def _install_keep_int_native():
    """Patch _materialize_copy so INTEGER source tensors are never cast to a FLOATING dtype during load.

    Without an hf_quantizer the loader casts every source to the target param dtype (bf16); that destroys the integer
    bit pattern of packed weights (`weight_packed`), shapes (`weight_shape`), zero-points and g_idx. Floats (the scale,
    global scale, dense weights) still cast normally.
    """
    import transformers.core_model_loading as cml

    if getattr(cml._materialize_copy, "_onfly_patched", False):
        return
    orig = cml._materialize_copy

    def _materialize_copy(tensor, device=None, dtype=None):
        t = tensor[...]
        if dtype is not None and dtype.is_floating_point and not t.is_floating_point():
            dtype = None  # keep packed/shape/zero-point/g_idx integers native
        return orig(t, device=device, dtype=dtype)

    setattr(_materialize_copy, "_onfly_patched", True)
    cml._materialize_copy = _materialize_copy


# Marks a config whose model's converters the installed hooks rewrite (holds the checkpoint's format).
_MARK = "_modelopt_onfly_decompress"


def _mark(hf_config, fmt):
    """Mark ``hf_config``'s load for the installed hooks; ``from_pretrained``'s copy of it keeps the mark."""
    if isinstance(hf_config, dict):
        hf_config[_MARK] = fmt
    else:
        setattr(hf_config, _MARK, fmt)


def is_marked(hf_config) -> bool:
    """Whether :func:`install_onfly_decompress` marked ``hf_config`` (or its ``text_config``) for its hooks."""
    configs = (hf_config, getattr(hf_config, "text_config", None))
    return any(getattr(c, _MARK, None) is not None for c in configs if c is not None)


def install_onfly_decompress(hf_config):
    """Monkeypatch get_model_conversion_mapping so the fused-MoE per-expert converters decompress the
    checkpoint's compressed-tensors weights on the fly, in whatever ``format`` the checkpoint declares.
    Call before from_pretrained of a compressed checkpoint. ``hf_config`` is the model config carrying
    the compressed-tensors ``quantization_config`` (top-level or under ``text_config``); the format,
    bit-width, group size, strategy and stored param names are all read from it.

    The hooks are process-wide, so a process decompresses one compressed checkpoint: installing again
    for it marks that config too, and installing for a checkpoint compressed differently raises. Only a
    model built from a marked config has its converters rewritten; any other model ``from_pretrained``
    loads in the process, an uncompressed MoE say, keeps its stock converters."""
    import transformers.modeling_utils as mu
    from transformers.core_model_loading import WeightConverter

    from modelopt.torch.utils import print_rank_0
    from modelopt.torch.utils.moe import require_transformers

    installed = getattr(mu.get_model_conversion_mapping, "_onfly_compression", None)
    compression = _read_compression(hf_config)
    if installed is not None:
        if compression is None:
            # The prepare that installed the hooks drops quantization_config from its config, already
            # marked, and may run again on that same config: nothing new to decompress.
            return
        if compression != installed:
            raise RuntimeError(
                f"install_onfly_decompress: this process already decompresses a '{installed[3]}' "
                f"checkpoint (stored params {installed[2]}); one compressed differently ('{compression[3]}', "
                f"stored params {compression[2]}) needs a process of its own"
            )
        # The same checkpoint through a config of its own, which its load must find marked too.
        _mark(hf_config, installed[3])
        return
    if compression is None:
        # Nothing to decompress: refused before anything is patched.
        raise ValueError(
            "install_onfly_decompress: config has no compressed-tensors quantization_config"
        )
    compressor, scheme, param_names, fmt = compression
    require_transformers("5.8", "on-the-fly compressed-tensors decompression")
    _install_keep_int_native()
    orig = mu.get_model_conversion_mapping

    def patched(*args, **kwargs):
        convs = orig(*args, **kwargs)
        config = getattr(args[0] if args else kwargs["model"], "config", None)
        if getattr(config, _MARK, None) is None:
            # Not the compressed checkpoint: rewritten, an uncompressed MoE's experts would match no
            # stored key and load newly initialized without an error.
            return convs
        # from_pretrained loads from a copy of the marked config; the mark must not reach the config
        # a later save writes.
        delattr(config, _MARK)
        out = []
        for c in convs:
            # only the per-expert MoE converters (have `.weight` sources under `experts`)
            if isinstance(c, WeightConverter) and any(
                "experts" in p and p.endswith((".weight", ".weight$")) for p in c.source_patterns
            ):
                out.append(_compressedify(c, compressor, scheme, param_names))
            else:
                out.append(c)
        return out

    setattr(patched, "_onfly_compression", compression)
    mu.get_model_conversion_mapping = patched
    _mark(hf_config, fmt)
    print_rank_0(
        f"[onfly_decompress] installed: compressed_tensors '{fmt}' fused-MoE experts decompress "
        f"(stored params={param_names}) -> bf16 during from_pretrained"
    )
