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
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
)

logger = init_logger(__name__)

# Maps aiter tune module name → kernel class used by that module.
# FP8 block-scaled linear GEMM (Fp8LinearMethod + block_quant) is covered here.
# TODO: add Int8 linear GEMM (module_gemm_a8w8_tune, AiterInt8ScaledMMLinearKernel,
#   used by compressed_tensors/quark W8A8 INT8 schemes) and MoE batched GEMM
#   (module_batched_gemm_a8w8_tune / module_batched_gemm_bf16_tune, AiterExperts)
#   — both require separate shape collection logic.
_TUNE_MODULE_MAP: list[tuple[str, type]] = [
    ("module_gemm_a8w8_blockscale_tune", AiterFp8BlockScaledMMKernel),
]


def _m_values(max_tokens: int) -> list[int]:
    """Powers of two up to max_tokens — matches CUDA graph capture sizes."""
    ms = []
    m = 1
    while m <= max_tokens:
        ms.append(m)
        m *= 2
    return ms


def _collect_shapes(
    model: torch.nn.Module,
    kernel_cls: type,
    max_tokens: int,
) -> list[tuple[int, ...]]:
    """
    Collect (M, N, K) shapes for all LinearBase layers that use kernel_cls.

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
        for m in _m_values(max_tokens):
            shapes.append((m, n, k))

    return shapes


def aiter_gemm_warmup(model: torch.nn.Module, max_tokens: int) -> None:
    """
    Check (and optionally run) aiter GEMM tuning for all shapes the model will use.

    Called from kernel_warmup() before CUDA graph capture.
    """
    try:
        from aiter.utility.pretune import warmup as aiter_warmup
    except ImportError:
        logger.debug("aiter.utility.pretune not available — skipping aiter GEMM warmup.")
        return

    auto_tune = envs.VLLM_ROCM_USE_AITER_GEMM_AUTOTUNE

    for module_name, kernel_cls in _TUNE_MODULE_MAP:
        shapes = _collect_shapes(model, kernel_cls, max_tokens)
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
