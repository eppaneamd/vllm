# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Asynchronous Direct Block I/O Weight Loader and Single-Reader Broadcast."""

import mmap
import os
import tempfile
from typing import Any
import pytest
from safetensors.torch import save_file
import torch

from vllm.model_executor.model_loader.direct_block_reader import (
    DirectBlockFileReader,
    calculate_alignment,
)
from vllm.model_executor.model_loader.moe_fast_loader import (
    SafetensorsMoEIndex,
    check_page_cache_warmth,
    fast_bypass_safetensors_iterator,
)
from vllm.model_executor.model_loader.shared_pinned_pool import (
    SharedPinnedBufferPool,
)


def test_calculate_alignment():
    """Verifies that calculate_alignment computes correct 4096-byte boundaries and shifts."""
    # Case 1: Perfectly aligned 4096-byte block
    off, sz, shift = calculate_alignment(0, 4096, 4096)
    assert off == 0
    assert sz == 4096
    assert shift == 0

    # Case 2: Unaligned start (e.g. 816 byte shift from safetensors header)
    off, sz, shift = calculate_alignment(816, 1000, 4096)
    assert off == 0
    assert sz == 4096
    assert shift == 816
    assert off + shift == 816

    # Case 3: Straddling multiple blocks
    off, sz, shift = calculate_alignment(4097, 4096, 4096)
    assert off == 4096
    assert sz == 8192
    assert shift == 1
    assert off + shift == 4097


@pytest.fixture
def synthetic_multishard_checkpoint():
    """Creates a temporary multi-shard safetensors checkpoint for direct I/O testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        num_shards = 2
        num_layers = 2
        num_experts = 4
        hidden_dim = 32
        intermediate_dim = 64

        shard_paths = []
        all_weights = {}

        for s_idx in range(num_shards):
            weights = {}
            if s_idx == 0:
                weights["model.embed_tokens.weight"] = torch.randn(
                    50, hidden_dim, dtype=torch.bfloat16
                )
                weights["model.norm.weight"] = torch.ones(hidden_dim, dtype=torch.bfloat16)

            for l in range(num_layers):
                weights[f"model.layers.{l}.input_layernorm.weight"] = torch.ones(
                    hidden_dim, dtype=torch.bfloat16
                )
                for e in range(num_experts):
                    # Distribute experts across shards
                    if e % num_shards == s_idx:
                        weights[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"] = (
                            torch.randn(intermediate_dim, hidden_dim, dtype=torch.bfloat16)
                        )
                        weights[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"] = (
                            torch.randn(intermediate_dim, hidden_dim, dtype=torch.bfloat16)
                        )
                        weights[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"] = (
                            torch.randn(hidden_dim, intermediate_dim, dtype=torch.bfloat16)
                        )

            shard_path = os.path.join(tmpdir, f"model-{s_idx:05d}-of-{num_shards:05d}.safetensors")
            save_file(weights, shard_path)
            shard_paths.append(shard_path)
            all_weights.update(weights)

        yield shard_paths, all_weights, num_layers, num_experts


def test_shared_pinned_buffer_pool_lifecycle():
    """Verifies creation, memoryview access, and clean teardown of SharedPinnedBufferPool."""
    prefix = f"test_pool_{os.getpid()}"
    slot_size = 1024 * 1024 # 1 MB
    pool = SharedPinnedBufferPool(
        prefix=prefix,
        slot_size=slot_size,
        is_creator=True,
        tp_rank=0,
        tp_size=1,
    )

    try:
        buf0 = pool.get_slot_buffer(0)
        assert len(buf0) == slot_size
        # Test write and read
        test_payload = b"DIRECT_IO_TEST_PAYLOAD"
        buf0[: len(test_payload)] = test_payload
        assert bytes(buf0[: len(test_payload)]) == test_payload
    finally:
        pool.unlink()


def test_check_page_cache_warmth(synthetic_multishard_checkpoint):
    """Verifies that check_page_cache_warmth returns a valid fraction in [0.0, 1.0]."""
    shard_paths, _, _, _ = synthetic_multishard_checkpoint
    warmth = check_page_cache_warmth(shard_paths)
    assert isinstance(warmth, float)
    assert 0.0 <= warmth <= 1.0


def test_direct_block_file_reader(synthetic_multishard_checkpoint):
    """Verifies DirectBlockFileReader correctly reads shard contents into destination buffer."""
    shard_paths, all_weights, _, _ = synthetic_multishard_checkpoint
    shard_0 = shard_paths[0]
    file_size = os.path.getsize(shard_0)

    reader = DirectBlockFileReader(chunk_size=64 * 1024, max_workers=2)
    aligned_buf_len = ((file_size + 4095) // 4096) * 4096
    mm = mmap.mmap(-1, aligned_buf_len)
    mv = memoryview(mm)

    try:
        n_read = reader.read_file_to_buffer(shard_0, mv, file_offset=0, length=file_size)
        assert n_read == file_size

        with open(shard_0, "rb") as f:
            expected_bytes = f.read()
        assert bytes(mv[:file_size]) == expected_bytes
    finally:
        mv.release()
        mm.close()
        reader.close()


def test_direct_io_broadcast_parity(synthetic_multishard_checkpoint):
    """Verifies that Mode 3 (Direct-I/O Broadcast) yields bitwise identical tensors to Mode 2."""
    shard_paths, all_weights, _, _ = synthetic_multishard_checkpoint

    # Run Mode 2 (Direct-to-VRAM)
    tensors_mode_2 = {
        k: v.clone()
        for k, v in fast_bypass_safetensors_iterator(
            shard_paths,
            direct_vram_mode=True,
            direct_io_mode=False,
        )
    }

    # Run Mode 3 (Direct-I/O Broadcast)
    tensors_mode_3 = {
        k: v.clone()
        for k, v in fast_bypass_safetensors_iterator(
            shard_paths,
            direct_io_mode=True,
            tp_rank=0,
            tp_size=1,
        )
    }

    assert set(tensors_mode_2.keys()) == set(tensors_mode_3.keys())
    for k in tensors_mode_2:
        t2 = tensors_mode_2[k]
        t3 = tensors_mode_3[k]
        assert t2.shape == t3.shape, f"Shape mismatch for {k}: {t2.shape} vs {t3.shape}"
        assert t2.dtype == t3.dtype, f"Dtype mismatch for {k}: {t2.dtype} vs {t3.dtype}"
        assert torch.equal(t2, t3), f"Tensor value mismatch for key {k}"


def _mp_worker(rank: int, world_size: int, shard_paths: list[str], res_queue: Any) -> None:
    local_eids = {rank}
    tensors = {
        k: v.clone()
        for k, v in fast_bypass_safetensors_iterator(
            shard_paths,
            local_expert_ids=local_eids,
            direct_io_mode=True,
            tp_rank=rank,
            tp_size=world_size,
        )
    }
    res_queue.put((rank, list(tensors.keys())))


def test_direct_io_multiprocess_broadcast(synthetic_multishard_checkpoint):
    """Verifies that multi-rank broadcast correctly serves disjoint expert sets to concurrent ranks."""
    shard_paths, all_weights, num_layers, num_experts = synthetic_multishard_checkpoint

    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()
    world_size = 2
    procs = []
    for r in range(world_size):
        p = ctx.Process(target=_mp_worker, args=(r, world_size, shard_paths, q))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"Worker process failed with code {p.exitcode}"

    results = {}
    while not q.empty():
        r, keys = q.get()
        results[r] = keys

    assert len(results) == world_size
    # Verify non-MoE keys are received by both ranks
    assert "model.embed_tokens.weight" in results[0]
    assert "model.embed_tokens.weight" in results[1]

    # Verify rank 0 received expert 0 and not expert 1
    r0_has_e0 = any("experts.0." in k for k in results[0])
    r0_has_e1 = any("experts.1." in k for k in results[0])
    assert r0_has_e0 and not r0_has_e1

    # Verify rank 1 received expert 1 and not expert 0
    r1_has_e0 = any("experts.0." in k for k in results[1])
    r1_has_e1 = any("experts.1." in k for k in results[1])
    assert r1_has_e1 and not r1_has_e0


def test_direct_io_broadcast_ep_filtering(synthetic_multishard_checkpoint):
    """Verifies EP expert filtering correctly restricts yielded slices in Mode 3."""
    shard_paths, all_weights, num_layers, num_experts = synthetic_multishard_checkpoint

    # Filter to only local expert 1
    local_eids = {1}
    tensors_ep = {
        k: v.clone()
        for k, v in fast_bypass_safetensors_iterator(
            shard_paths,
            local_expert_ids=local_eids,
            direct_io_mode=True,
            tp_rank=0,
            tp_size=1,
        )
    }

    for k in tensors_ep:
        if "experts." in k:
            import re
            m = re.search(r"experts\.(\d+)\.", k)
            if m:
                eid = int(m.group(1))
                assert eid in local_eids, f"Unassigned expert {eid} found in {k}"

    has_e1 = any("experts.1." in k for k in tensors_ep)
    has_e0 = any("experts.0." in k for k in tensors_ep)
    assert has_e1, "Local expert 1 not found in yielded tensors"
    assert not has_e0, "Non-local expert 0 erroneously found in yielded tensors"

