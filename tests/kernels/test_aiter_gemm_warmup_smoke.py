#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Smoke test for aiter_gemm_warmup with a real FP8 model.

Loads Qwen/Qwen3-VL-30B-A3B-Instruct-FP8, calls _collect_shapes and
aiter_gemm_warmup without running inference.  Requires a live ROCm GPU
and model weights in HuggingFace cache.

Prerequisites (Python-only install, no C++ rebuild needed):
    cd /path/to/vllm-fork
    VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

    cd /path/to/aiter-main
    uv pip install -e .

Usage (TP=8 for full model, TP=1 for testing shape collection on a single GPU):
    VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1 \
    .venv/bin/python tests/kernels/test_aiter_gemm_warmup_smoke.py

To also trigger tuning for missing shapes:
    VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1 \
    VLLM_ROCM_USE_AITER_GEMM_AUTOTUNE=1 \
    .venv/bin/python tests/kernels/test_aiter_gemm_warmup_smoke.py
"""

import os
import sys

MODEL = os.getenv("SMOKE_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
# Use TP=8 to shard across all 8 GPUs (matching production config).
# Reduce to TP=1 if running on a single GPU for quick shape-collection check.
TENSOR_PARALLEL_SIZE = int(os.getenv("SMOKE_TP", "8"))
MAX_TOKENS = int(os.getenv("SMOKE_MAX_TOKENS", "128"))


def main():
    print(f"Loading {MODEL} with TP={TENSOR_PARALLEL_SIZE} ...")

    from vllm import LLM

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        # Required for MoE models (e.g. Qwen3-VL-30B-A3B) to distribute experts.
        enable_expert_parallel=True,
        # Limit memory to leave room for the warmup buffers.
        gpu_memory_utilization=0.85,
        # Disable CUDA graph capture — we only want to inspect the loaded model.
        enforce_eager=True,
        # Small context to speed up load.
        max_model_len=512,
        # Limit token budget to match the warmup M-value range.
        max_num_batched_tokens=MAX_TOKENS,
    )

    print("Model loaded. Extracting model runner ...")

    # Access the underlying model (driver worker in v1 engine).
    try:
        # v1 engine path
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        # Fallback for different engine variants
        model = llm.llm_engine.driver_worker.model_runner.model

    print(f"Model class: {type(model).__name__}")

    # --- Shape collection ---
    from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
        AiterFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.warmup.aiter_gemm_warmup import (
        _collect_shapes,
        aiter_gemm_warmup,
    )

    shapes = _collect_shapes(model, AiterFp8BlockScaledMMKernel, MAX_TOKENS)
    unique_nk = {s[1:] for s in shapes}

    print(f"\n--- Shape collection results ---")
    print(f"  Unique (N, K) pairs: {len(unique_nk)}")
    for nk in sorted(unique_nk):
        print(f"    N={nk[0]}, K={nk[1]}")
    print(f"  Total (M, N, K) shapes: {len(shapes)}")
    print(f"  M values: {sorted({s[0] for s in shapes})}")

    if not shapes:
        print(
            "\nWARNING: no AiterFp8BlockScaledMMKernel layers found.\n"
            "  Check that VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_LINEAR=1\n"
            "  are set, and that the model uses block-quantised FP8."
        )
        sys.exit(1)

    # --- Warmup ---
    print(f"\n--- Running aiter_gemm_warmup ---")
    aiter_gemm_warmup(model, MAX_TOKENS)
    print("aiter_gemm_warmup completed successfully.")

    print("\nDone.")
    sys.exit(0)


if __name__ == "__main__":
    main()
