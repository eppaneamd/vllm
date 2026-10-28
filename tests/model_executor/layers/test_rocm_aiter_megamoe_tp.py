# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row gating of the AITER MegaMoE-TP MoE path, on CPU with a stand-in kernel."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe import rocm_aiter_megamoe_tp as megamoe

pytestmark = pytest.mark.skip_global_cleanup


class _Kernel:
    """A built TP 2 MegaMoeTP: records the routing ids of each launch."""

    cfg = SimpleNamespace(world_size=2)

    def __init__(self):
        self.launched_ids = []

    def set_weights(self, **weights):
        pass

    def __call__(self, x, topk_weights, topk_ids, out):
        self.launched_ids.append(topk_ids)
        return out.fill_(1.0)


@pytest.mark.parametrize(
    ("rows", "takes"),
    [(255, False), (256, True), (257, True), (2048, True), (2049, False)],
)
def test_row_window_selects_megamoe(monkeypatch, rows, takes):
    monkeypatch.setenv("VLLM_ROCM_AITER_MEGAMOE_TP_MIN_TOKENS", "256")
    monkeypatch.setenv("VLLM_ROCM_AITER_MEGAMOE_TP_MAX_TOKENS", "2048")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    experts = SimpleNamespace(
        w13_weight=None, w13_weight_scale=None, w2_weight=None, w2_weight_scale=None
    )
    dispatch = megamoe.MegaMoeTPDispatch(SimpleNamespace(routed_experts=experts))
    kernel = _Kernel()
    dispatch._engine = kernel

    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    ids = torch.zeros(rows, 9, dtype=torch.int32)
    out = dispatch.forward(x, torch.ones(rows, 9), ids)

    if not takes:
        assert out is None and kernel.launched_ids == []
        return
    assert out is not None and torch.equal(out, torch.ones_like(x))
    (launched,) = kernel.launched_ids
    # ar mode takes a multiple of tp rows: an odd count gets one id -1 row.
    pad = rows % 2
    assert launched.shape == (rows + pad, 9)
    assert torch.equal(launched[:rows], ids)
    assert bool((launched[rows:] == -1).all())


def test_lb_knobs_parse_to_config_fields():
    assert megamoe.parse_lb("") == {}
    assert megamoe.parse_lb("mt=4; npp=2") == {"lb_mt": 4, "lb_npp": 2}
    assert megamoe.parse_lb("mt=1024:3,4") == {"lb_mt": "1024:3,4"}
    with pytest.raises(ValueError, match="VLLM_ROCM_AITER_MEGAMOE_TP_LB"):
        megamoe.parse_lb("mtx=4")
