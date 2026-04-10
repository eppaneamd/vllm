# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup aiter GEMM kernels for ROCm (AMD MI-series GPUs).

aiter JIT-compiles and tunes CK GEMM kernels on first use.  The warmup checks
whether each GEMM shape the model will use at runtime is already tuned for the
live GPU, and optionally triggers the tuner before CUDA graph capture so that
JIT compilation does not happen in the hot path.

Environment variables:
  VLLM_ROCM_USE_AITER_GEMM_AUTOTUNE=1  Run the tuner for any missing shape
                                        (default: warn only).
"""

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import AiterExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
)

logger = init_logger(__name__)

# Maps aiter tune module name → kernel class used by that module.
# FP8 block-scaled linear GEMM (Fp8LinearMethod + block_quant).
_TUNE_MODULE_MAP: list[tuple[str, type]] = [
    ("module_gemm_a8w8_blockscale_tune", AiterFp8BlockScaledMMKernel),
]

# Maps aiter batched tune module name → experts class used by that module.
# MoE batched GEMM: shapes are (B, M, N, K) where B = local_num_experts.
# TODO: add module_batched_gemm_bf16_tune for BF16 MoE when AiterExperts
#   supports a separate bf16 path.
_BATCHED_TUNE_MODULE_MAP: list[tuple[str, type]] = [
    ("module_batched_gemm_a8w8_tune", AiterExperts),
]


def _collect_shapes(
    model: torch.nn.Module,
    kernel_cls: type,
    capture_sizes: list[int],
) -> list[tuple[int, ...]]:
    """
    Collect (M, N, K) shapes for all LinearBase layers that use kernel_cls.

    M values come from cudagraph_capture_sizes — the exact batch sizes vLLM
    captures CUDA graphs for, covering both decode and chunked-prefill steps.
    Only LinearBase layers with block-quantised Fp8LinearMethod and an active
    fp8_linear kernel of type kernel_cls are included.
    """
    seen: set[tuple[int, int]] = set()  # (N, K) — dedup across tied weights
    shapes: list[tuple[int, ...]] = []

    for module in model.modules():
        if not isinstance(module, LinearBase):
            continue
        qm = module.quant_method
        if not (
            isinstance(qm, Fp8LinearMethod)
            and getattr(qm, "block_quant", False)
            and isinstance(getattr(qm, "fp8_linear", None), kernel_cls)
        ):
            continue
        w = module.weight
        if not isinstance(w, torch.Tensor) or w.ndim != 2:
            continue
        n, k = w.shape
        if (n, k) in seen:
            continue
        seen.add((n, k))
        for m in capture_sizes:
            shapes.append((m, n, k))

    return shapes


def _collect_batched_shapes(
    model: torch.nn.Module,
    experts_cls: type,
    capture_sizes: list[int],
) -> list[tuple[int, ...]]:
    """
    Collect (B, M, N, K) shapes for all FusedMoE layers that use experts_cls.

    B = local_num_experts (experts assigned to this GPU after EP/TP split).
    N, K from moe_config: intermediate_size_per_partition and hidden_dim.
    Each MoE layer contributes two shape families:
      - gate/up projection: (B, M, N, K)  where N=intermediate, K=hidden
      - down projection:    (B, M, K, N)  where K=intermediate, N=hidden
    M values come from cudagraph_capture_sizes, same as for regular GEMM.
    """
    seen: set[tuple[int, int, int]] = set()  # (B, N, K) — dedup across layers
    shapes: list[tuple[int, ...]] = []

    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        experts = getattr(module, "experts", None)
        if not isinstance(experts, experts_cls):
            continue
        b = module.local_num_experts
        n = module.moe_config.intermediate_size_per_partition
        k = module.moe_config.hidden_dim
        if (b, n, k) in seen:
            continue
        seen.add((b, n, k))
        for m in capture_sizes:
            shapes.append((b, m, n, k))  # gate / up projection
            shapes.append((b, m, k, n))  # down projection

    return shapes


def aiter_gemm_warmup(model: torch.nn.Module, capture_sizes: list[int]) -> None:
    """
    Check (and optionally run) aiter GEMM tuning for all shapes the model will use.

    capture_sizes: cudagraph_capture_sizes from compilation_config — the exact
    M values (token counts) that CUDA graphs are captured for.  Covers both
    decode steps and chunked-prefill steps that execute as captured graphs.

    Called from kernel_warmup() before CUDA graph capture.
    """
    # aiter tuning writes to a shared CSV on disk and compiles a shared .so.
    # Running from multiple workers simultaneously would race-write the CSV,
    # so only rank 0 checks coverage and runs the tuner; other workers skip.
    if get_world_group().local_rank != 0:
        return

    if not capture_sizes:
        logger.debug("aiter GEMM warmup: no CUDA graph capture sizes — skipping.")
        return

    try:
        from aiter.utility.pretune import warn_if_undertuned, warmup as aiter_warmup
    except ImportError:
        logger.debug("aiter.utility.pretune not available — skipping aiter GEMM warmup.")
        return

    auto_tune = envs.VLLM_ROCM_USE_AITER_GEMM_AUTOTUNE

    # ── CSV coverage check ────────────────────────────────────────────────────
    # Check whether the tuning CSVs have sufficient coverage for the live GPU
    # across all modules this model uses, before checking specific shapes.
    all_module_names = (
        [name for name, _ in _TUNE_MODULE_MAP]
        + [name for name, _ in _BATCHED_TUNE_MODULE_MAP]
    )
    undertuned = warn_if_undertuned(all_module_names)
    if undertuned and not auto_tune:
        logger.warning(
            "aiter GEMM: %d module(s) have low CSV coverage for this GPU. "
            "Set VLLM_ROCM_USE_AITER_GEMM_AUTOTUNE=1 or pre-tune with: "
            "PRETUNE_MODULES=%s python setup.py develop",
            len(undertuned),
            ",".join(d["module"] for d in undertuned),
        )

    # ── Regular (non-batched) GEMM ────────────────────────────────────────────
    for module_name, kernel_cls in _TUNE_MODULE_MAP:
        shapes = _collect_shapes(model, kernel_cls, capture_sizes)
        if not shapes:
            logger.debug(
                "aiter GEMM warmup: no %s layers found, skipping %s.",
                kernel_cls.__name__,
                module_name,
            )
            continue

        unique_nk = len({s[1:] for s in shapes})
        logger.info(
            "aiter GEMM warmup: checking %d (M, N, K) shapes across %d unique (N, K) "
            "for %s.",
            len(shapes),
            unique_nk,
            module_name,
        )
        result = aiter_warmup(shapes, module_name, auto_tune=auto_tune)
        if result["missing"] > 0 and not auto_tune:
            logger.warning(
                "aiter GEMM warmup: %d/%d shapes not tuned for %s. "
                "Set VLLM_ROCM_USE_AITER_GEMM_AUTOTUNE=1 to tune automatically.",
                result["missing"],
                len(shapes),
                module_name,
            )

    # ── Batched MoE GEMM ─────────────────────────────────────────────────────
    for module_name, experts_cls in _BATCHED_TUNE_MODULE_MAP:
        shapes = _collect_batched_shapes(model, experts_cls, capture_sizes)
        if not shapes:
            logger.debug(
                "aiter batched GEMM warmup: no %s experts found, skipping %s.",
                experts_cls.__name__,
                module_name,
            )
            continue

        unique_bnk = len({(s[0], s[2], s[3]) for s in shapes})
        logger.info(
            "aiter batched GEMM warmup: checking %d (B, M, N, K) shapes across "
            "%d unique (B, N, K) for %s.",
            len(shapes),
            unique_bnk,
            module_name,
        )
        result = aiter_warmup(shapes, module_name, auto_tune=auto_tune)
        if result["missing"] > 0 and not auto_tune:
            logger.warning(
                "aiter batched GEMM warmup: %d/%d shapes not tuned for %s. "
                "Set VLLM_ROCM_USE_AITER_GEMM_AUTOTUNE=1 to tune automatically.",
                result["missing"],
                len(shapes),
                module_name,
            )
