# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MegaMoE-TP for row-gated forwards of a TP MoE layer (ROCm gfx950).

MegaMoeTP (aiter/ops/flydsl/mega_moe_tp.py) runs a TP MoE layer as one persistent
kernel: activation quant, GEMM1 + SiLU, GEMM2 and, in comm_mode "ar", the
all-reduce of the routed output. Routing stays in vLLM. With
VLLM_ROCM_AITER_MEGAMOE_TP=1, forwards of VLLM_ROCM_AITER_MEGAMOE_TP_MIN_TOKENS to
_MAX_TOKENS rows take it, eager or inside a CUDA graph; other row counts keep the
stock path.

Kernel contract (mega_moe_tp.py module docstring): every TP rank makes the same
calls with the same rows, launch configs are prepared before graph capture, and
nothing else runs on the GPU meanwhile. The take / stock decision depends only on
the row count and the layer, which TP ranks share. A shared expert must be fused
into the routed experts: an unfused one would need its own all-reduce and stream.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_LB_KEYS = ("min", "mt", "mt_small", "small_max", "npp", "q")

# One kernel instance per layer shape and setting, shared by the layers of that
# shape (MegaMoeTP.set_weights): their launches form one collective sequence.
_ENGINES: dict[tuple, Any] = {}


def comm_options(comm_dtype: str) -> tuple[str, str]:
    """MegaMoeTPConfig (comm_dtype, ar_gather) for the env value."""
    # fp8: fp8 partial sums, MXFP8 second gather hop from 512 rows (AITER "auto").
    return ("fp8", "auto") if comm_dtype == "fp8" else ("bf16", "bf16")


def parse_lb(spec: str) -> dict[str, int | str]:
    """VLLM_ROCM_AITER_MEGAMOE_TP_LB ("mt=4;npp=2") as MegaMoeTPConfig fields.

    mt may be AITER's row-tile table ("1024:3,4"), so knobs are split on ";".
    """
    fields: dict[str, int | str] = {}
    for item in filter(None, (s.strip() for s in spec.split(";"))):
        key, sep, value = (s.strip() for s in item.partition("="))
        if not sep or key not in _LB_KEYS:
            raise ValueError(
                f"VLLM_ROCM_AITER_MEGAMOE_TP_LB: {item!r} is not one of "
                f"{', '.join(k + '=N' for k in _LB_KEYS)}"
            )
        table = key == "mt" and not value.isdigit()
        fields[f"lb_{key}"] = value if table else int(value)
    return fields


class MegaMoeTPDispatch:
    """Runs one MoERunner's routed experts through MegaMoeTP when it applies."""

    def __init__(self, runner) -> None:
        self._runner = runner
        self._min_tokens = envs.VLLM_ROCM_AITER_MEGAMOE_TP_MIN_TOKENS
        self._max_tokens = envs.VLLM_ROCM_AITER_MEGAMOE_TP_MAX_TOKENS
        if not 1 <= self._min_tokens <= self._max_tokens:
            raise ValueError(
                "VLLM_ROCM_AITER_MEGAMOE_TP needs 1 <= _MIN_TOKENS <= _MAX_TOKENS, got "
                f"{self._min_tokens} and {self._max_tokens}"
            )
        self._comm_dtype = envs.VLLM_ROCM_AITER_MEGAMOE_TP_COMM_DTYPE
        self._lb = parse_lb(envs.VLLM_ROCM_AITER_MEGAMOE_TP_LB)
        # aiter's MegaMoeTP; AITER is an optional dependency, hence Any.
        self._engine: Any = None
        self._declined: str | None = None
        # Watchdog bits of the previous eager launch, copied asynchronously.
        self._err_host: torch.Tensor | None = None

    def _unsupported(self, x: torch.Tensor) -> str | None:
        from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp_config import (
            mega_moe_tp_shape_supported,
        )
        from aiter.ops.flydsl.mega_moe_tp import mega_moe_tp_supported

        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
            ZeroExpertRouter,
        )

        r = self._runner
        mc, qm, rx = r.moe_config, r._quant_method, r.routed_experts
        backend = getattr(getattr(qm, "mxfp4_backend", None), "value", None)
        inter = rx.w13_weight.shape[1] // 2
        checks = (
            (mega_moe_tp_supported(), "needs gfx950"),
            (backend == "AITER_MXFP4_MXFP4", f"MoE backend {backend}"),
            (not qm.is_monolithic, "monolithic MoE kernel"),
            (mc.tp_size > 1, "tensor parallel size 1"),
            (
                mc.dp_size == 1
                and mc.pcp_size == 1
                and not mc.use_ep
                and not mc.is_sequence_parallel,
                "data, context, expert or sequence parallelism",
            ),
            (r.shared_experts is None, "unfused shared expert"),
            (not mc.skip_final_all_reduce, "deferred final all-reduce"),
            (not r.enable_dbo, "dual-batch overlap"),
            (not mc.is_lora_enabled, "LoRA"),
            (not mc.has_hash_routing, "hash routing"),
            (
                not mc.require_decomposed_backend,
                "expert substitution or simulated routing",
            ),
            (mc.activation == MoEActivation.SILU, f"activation {mc.activation}"),
            (mc.swiglu_alpha is None and mc.swiglu_beta is None, "SwiGLU alpha/beta"),
            (not mc.has_bias, "expert bias"),
            (rx.expert_map is None, "expert map"),
            (not rx.apply_router_weight_on_input, "router weight on input"),
            (
                r.routed_input_transform is None and r.routed_output_transform is None,
                "routed input/output transform",
            ),
            (not isinstance(r.router, ZeroExpertRouter), "zero experts"),
            (r.router.eplb_state is None, "EPLB"),
            (x.dtype == torch.bfloat16, f"activations {x.dtype}"),
            (
                mc.hidden_dim_unpadded == mc.hidden_dim
                and mc.intermediate_size_per_partition_unpadded
                == mc.intermediate_size_per_partition
                and not mc.intermediate_pad,
                "padded hidden or intermediate size",
            ),
            (
                mega_moe_tp_shape_supported(x.shape[1], inter, mc.tp_size),
                f"shape h{x.shape[1]} i{inter} tp{mc.tp_size}",
            ),
        )
        return next((why for ok, why in checks if not ok), None)

    def _build(self, x: torch.Tensor, topk_ids: torch.Tensor):
        from aiter import ActivationType
        from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig

        from vllm.distributed import get_tp_group

        rx, mc = self._runner.routed_experts, self._runner.moe_config
        tp = get_tp_group()
        num_experts, n2, _ = rx.w13_weight.shape
        hidden, topk = x.shape[1], topk_ids.shape[1]
        local = -(-self._max_tokens // tp.world_size)
        key = (hidden, n2 // 2, num_experts, topk, mc.swiglu_limit, local)
        eng = _ENGINES.get(key)
        if eng is not None:
            return eng
        comm_dtype, ar_gather = comm_options(self._comm_dtype)
        cfg = MegaMoeTPConfig(
            rank=tp.rank_in_group,
            world_size=tp.world_size,
            model_dim=hidden,
            inter_dim=n2 // 2,
            experts=num_experts,
            topk=topk,
            max_local_tokens=local,
            activation=ActivationType.Silu,
            swiglu_limit=mc.swiglu_limit,
            comm_mode="ar",
            comm_dtype=comm_dtype,
            ar_gather=ar_gather,
            **self._lb,
        )
        eng = MegaMoeTP(
            cfg,
            w1=rx.w13_weight,
            w1_scale=rx.w13_weight_scale,
            w2=rx.w2_weight,
            w2_scale=rx.w2_weight_scale,
            group=tp.cpu_group,
            device=x.device,
        )
        # Arm every launch config the window reaches (launch_config is keyed by
        # local rows), so graph capture never compiles.
        first = -(-self._min_tokens // tp.world_size)
        configs = {eng.launch_config(m): m for m in range(local, first - 1, -1)}
        eng.prepare(sorted(configs.values()))
        _ENGINES[key] = eng
        self._err_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        logger.info(
            "MegaMoeTP enabled: h%d i%d e%d k%d tp%d, %d-%d rows, comm %s, LB %s, "
            "%d launch configs",
            hidden,
            n2 // 2,
            num_experts,
            topk,
            tp.world_size,
            self._min_tokens,
            self._max_tokens,
            comm_dtype,
            self._lb or "AITER default",
            len(configs),
        )
        return eng

    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """The all-reduced routed output, or None for the stock path."""
        if self._declined is not None:
            return None
        capturing = torch.cuda.is_current_stream_capturing()
        eng = self._engine
        if eng is None:
            # Building is collective and allocates, so it happens on the first
            # eager forward whatever its size: the memory profiling run, before
            # KV cache sizing and CUDA graph capture.
            if capturing:
                logger.warning_once(
                    "MegaMoeTP: CUDA graph capture before the first eager forward; "
                    "the captured graphs use the stock MoE path",
                    scope="process",
                )
                return None
            from aiter.ops.flydsl.mega_moe_tp import MegaMoeTPLDSError

            why = self._unsupported(x)
            if why is None:
                try:
                    eng = self._engine = self._build(x, topk_ids)
                except MegaMoeTPLDSError as e:
                    why = str(e)
            if why is not None:
                self._declined = why
                logger.warning_once("MegaMoeTP declined: %s", why, scope="process")
                return None
        rows = x.shape[0]
        if not self._min_tokens <= rows <= self._max_tokens:
            return None
        # The sticky bits copied after the previous eager launch of the layer
        # that built the kernel; that step's output sync completed the copy.
        err = self._err_host
        if err is not None and not capturing and int(err[0]):
            raise RuntimeError(
                "MegaMoeTP: a wait inside the kernel timed out; MoE outputs since the "
                f"previous check are invalid (watchdog bits {int(err[0])})"
            )
        rx = self._runner.routed_experts
        eng.set_weights(
            w1=rx.w13_weight,
            w1_scale=rx.w13_weight_scale,
            w2=rx.w2_weight,
            w2_scale=rx.w2_weight_scale,
        )
        pad = -rows % eng.cfg.world_size
        if pad:  # ar mode takes a multiple of tp rows; id -1 routes nowhere
            x = F.pad(x, (0, 0, 0, pad))
            topk_weights = F.pad(topk_weights, (0, 0, 0, pad))
            topk_ids = F.pad(topk_ids, (0, 0, 0, pad), value=-1)
        out = eng(x, topk_weights, topk_ids, out=torch.empty_like(x))
        if err is not None and not capturing:
            err.copy_(eng.error_flag(), non_blocking=True)
        # info_once dedups on the arguments too, so the messages carry none.
        if capturing:
            logger.info_once(
                "MegaMoeTP: launch captured in a CUDA graph", scope="process"
            )
        else:
            logger.info_once("MegaMoeTP: first eager launch", scope="process")
        return out[:rows]
