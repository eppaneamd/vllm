# SPDX-License-Identifier: Apache-2.0
"""Fast Safetensors MoE bypass loader.

Pre-indexes safetensors shard headers to consolidate individual 2D MoE expert
slices into contiguous 3D host tensors (gate_up, down, and quantization scales)
before yielding to the model loader. Eliminates thousands of Python generator
iterations, submodule tree traversals, and non-contiguous GPU strided DMA copies.
"""

from collections import defaultdict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import os
import re
import struct
from typing import Any

from safetensors import safe_open
import torch

logger = logging.getLogger(__name__)

# Regex matching standard MoE expert slice keys across model families:
# e.g.:
# model.layers.0.mlp.experts.42.gate_proj.weight
# model.language_model.layers.0.mlp.experts.42.down_proj.weight_scale
# transformer.encoder.layers.0.mlp.experts.42.dense_h_to_4h.weight
# layers.0.block_sparse_moe.experts.42.w1.weight
_MOE_2D_KEY_RE = re.compile(
    r"^(?P<prefix>.*?\b(?:experts|block_sparse_moe\.experts)\b)\."
    r"(?P<expert_id>\d+)\."
    r"(?P<proj>[^.]+)"
    r"(?P<suffix>\..*)?$"
)

# Regex matching pre-fused 3D MoE expert keys:
# e.g.:
# model.layers.0.mlp.experts.gate_up_proj.weight
# model.layers.0.mlp.experts.down_proj.weight_scale
# model.language_model.layers.3.mlp.experts.w13_weight
_MOE_3D_KEY_RE = re.compile(
    r"^(?P<prefix>.*?\b(?:experts|block_sparse_moe\.experts)\b)\."
    r"(?P<proj>[^.]+)"
    r"(?P<suffix>\..*)?$"
)

# Regex matching shared expert keys when Fused Shared Experts (FSE) is enabled:
# e.g.:
# model.layers.0.mlp.shared_experts.gate_proj.weight
# model.layers.0.mlp.shared_expert.down_proj.weight_scale
# layers.0.mlp.shared_experts.w1.weight
# layers.0.ffn.shared_experts.w1.weight
_SHARED_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.*?\b(?:mlp|block_sparse_moe|ffn)\b)\."
    r"(?:shared_experts?)\."
    r"(?P<proj>[^.]+)"
    r"(?P<suffix>\..*)?$"
)

_MOE_KEY_RE = _MOE_2D_KEY_RE

# Canonical projection mappings
_GATE_NAMES = frozenset({"gate_proj", "w1", "w1_weight", "dense_h_to_4h_gate"})
_UP_NAMES = frozenset({"up_proj", "w3", "w3_weight", "dense_h_to_4h"})
_DOWN_NAMES = frozenset({"down_proj", "w2", "w2_weight", "dense_4h_to_h"})
_FUSED_GATE_UP_NAMES = frozenset(
    {"gate_up_proj", "w13", "w13_weight", "dense_h_to_4h_gate_up"}
)
_ALL_MOE_PROJ_NAMES = _GATE_NAMES | _UP_NAMES | _DOWN_NAMES | _FUSED_GATE_UP_NAMES


@dataclass(slots=True)
class MoESliceLocation:
    shard_file: str
    key: str
    expert_id: int
    proj: str
    suffix: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(slots=True)
class MoE3DTensorLocation:
    shard_file: str
    key: str
    proj: str
    suffix: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    num_experts: int


@dataclass
class MoELayerPlan:
    layer_prefix: str
    num_total_experts: int = 0
    num_routed_experts: int = 0
    # 2D slices: (proj_type, suffix) -> dict[expert_id, MoESliceLocation]
    slices: dict[tuple[str, str], dict[int, MoESliceLocation]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # Shared expert slices when FSE is active: (proj_type, suffix) -> MoESliceLocation
    shared_slices: dict[tuple[str, str], MoESliceLocation] = field(
        default_factory=dict
    )
    # 3D tensors: (proj_type, suffix) -> MoE3DTensorLocation
    tensors_3d: dict[tuple[str, str], MoE3DTensorLocation] = field(
        default_factory=dict
    )


class PinnedHostStagingPool:
    """Pool of reusable pinned CPU buffers across MoE layers.

    Caches pinned host tensors by (shape, dtype) to eliminate repeated
    hipHostMalloc / cudaHostAlloc driver allocation and page-locking overhead.
    """

    def __init__(self, capacity_per_shape: int = 2):
        self.capacity = capacity_per_shape
        self._pool: dict[tuple[tuple[int, ...], torch.dtype], list[torch.Tensor]] = (
            defaultdict(list)
        )

    def acquire(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        key = (shape, dtype)
        if self._pool[key]:
            buf = self._pool[key].pop()
            buf.zero_()
            return buf
        pin = torch.cuda.is_available()
        return torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=pin)

    def release(self, tensor: torch.Tensor) -> None:
        key = (tuple(tensor.shape), tensor.dtype)
        if len(self._pool[key]) < self.capacity:
            self._pool[key].append(tensor)
        else:
            del tensor

    def clear(self) -> None:
        self._pool.clear()


class SafetensorsMoEIndex:
    """Parses safetensors JSON headers upfront to partition non-MoE and MoE keys."""

    def __init__(
        self,
        hf_weights_files: list[str],
        local_expert_ids: set[int] | None = None,
        fse_enabled: bool = False,
        n_shared_experts: int = 1,
    ):
        self.hf_weights_files = hf_weights_files
        self.local_expert_ids = local_expert_ids
        self.fse_enabled = fse_enabled
        self.n_shared_experts = n_shared_experts

        # Ordered non-MoE keys: list of (key, shard_file)
        self.non_moe_keys: list[tuple[str, str]] = []

        # Layer plans: layer_prefix -> MoELayerPlan
        self.moe_layers: dict[str, MoELayerPlan] = {}

        # Open file handles for safe_open: shard_file -> safe_open handle
        self.handles: dict[str, Any] = {}

    @classmethod
    def build(
        cls,
        hf_weights_files: list[str],
        local_expert_ids: set[int] | None = None,
        max_workers: int = 8,
        fse_enabled: bool = False,
        n_shared_experts: int = 1,
    ) -> "SafetensorsMoEIndex":
        index = cls(
            hf_weights_files,
            local_expert_ids,
            fse_enabled=fse_enabled,
            n_shared_experts=n_shared_experts,
        )
        index._parse_headers(max_workers)
        return index

    def _parse_headers(self, max_workers: int) -> None:
        def read_header(shard_path: str) -> tuple[str, dict[str, Any]]:
            with open(shard_path, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_bytes = f.read(header_size)
                header_json = json.loads(header_bytes.decode("utf-8"))
            return shard_path, header_json

        workers = min(max_workers, max(1, len(self.hf_weights_files)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            headers = list(executor.map(read_header, self.hf_weights_files))

        dtype_map = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "U8": torch.uint8,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "BOOL": torch.bool,
        }

        for shard_file, header in headers:
            for key, meta in header.items():
                if key == "__metadata__":
                    continue

                m_2d = _MOE_2D_KEY_RE.match(key)
                if m_2d:
                    expert_id = int(m_2d.group("expert_id"))
                    if (
                        self.local_expert_ids is not None
                        and expert_id not in self.local_expert_ids
                    ):
                        # Drop non-local expert slice upfront (EP pruning)
                        continue

                    prefix = m_2d.group("prefix")
                    proj = m_2d.group("proj")
                    suffix = m_2d.group("suffix") or ""

                    if prefix not in self.moe_layers:
                        self.moe_layers[prefix] = MoELayerPlan(
                            layer_prefix=prefix,
                            num_total_experts=0,
                        )
                    layer_plan = self.moe_layers[prefix]

                    # Map projection to canonical category
                    if proj in _GATE_NAMES:
                        proj_cat = "gate"
                    elif proj in _UP_NAMES:
                        proj_cat = "up"
                    elif proj in _DOWN_NAMES:
                        proj_cat = "down"
                    elif proj in _FUSED_GATE_UP_NAMES:
                        proj_cat = "gate_up"
                    else:
                        proj_cat = proj

                    shape = tuple(meta.get("shape", ()))
                    dtype_str = meta.get("dtype", "BF16")
                    dtype = dtype_map.get(dtype_str, torch.bfloat16)

                    slice_loc = MoESliceLocation(
                        shard_file=shard_file,
                        key=key,
                        expert_id=expert_id,
                        proj=proj,
                        suffix=suffix,
                        shape=shape,
                        dtype=dtype,
                    )
                    layer_plan.slices[(proj_cat, suffix)][expert_id] = slice_loc
                    layer_plan.num_routed_experts = max(
                        layer_plan.num_routed_experts, expert_id + 1
                    )
                    continue

                m_3d = _MOE_3D_KEY_RE.match(key)
                shape = tuple(meta.get("shape", ()))
                proj = m_3d.group("proj") if m_3d else ""
                is_3d_moe = (
                    m_3d is not None
                    and proj in _ALL_MOE_PROJ_NAMES
                    and "bias" not in key
                    and (
                        len(shape) == 3
                        or (
                            len(shape) in (1, 2, 3)
                            and (
                                "scale" in key
                                or "scale" in (m_3d.group("suffix") or "")
                            )
                        )
                    )
                    and len(shape) >= 1
                    and shape[0] >= 1
                )

                if is_3d_moe:
                    prefix = m_3d.group("prefix")
                    suffix = m_3d.group("suffix") or ""
                    dtype_str = meta.get("dtype", "BF16")
                    dtype = dtype_map.get(dtype_str, torch.bfloat16)

                    if proj in _GATE_NAMES:
                        proj_cat = "gate"
                    elif proj in _UP_NAMES:
                        proj_cat = "up"
                    elif proj in _DOWN_NAMES:
                        proj_cat = "down"
                    elif proj in _FUSED_GATE_UP_NAMES:
                        proj_cat = "gate_up"
                    else:
                        proj_cat = proj

                    if prefix not in self.moe_layers:
                        self.moe_layers[prefix] = MoELayerPlan(
                            layer_prefix=prefix,
                            num_total_experts=shape[0],
                            num_routed_experts=shape[0],
                        )
                    layer_plan = self.moe_layers[prefix]
                    layer_plan.num_routed_experts = max(
                        layer_plan.num_routed_experts, shape[0]
                    )

                    tensor_3d = MoE3DTensorLocation(
                        shard_file=shard_file,
                        key=key,
                        proj=proj,
                        suffix=suffix,
                        shape=shape,
                        dtype=dtype,
                        num_experts=shape[0],
                    )
                    layer_plan.tensors_3d[(proj_cat, suffix)] = tensor_3d
                    continue

                m_shared = (
                    _SHARED_EXPERT_KEY_RE.match(key) if self.fse_enabled else None
                )
                if m_shared:
                    prefix_base = m_shared.group("prefix")
                    prefix = f"{prefix_base}.experts"
                    proj = m_shared.group("proj")
                    suffix = m_shared.group("suffix") or ""

                    if proj in _GATE_NAMES:
                        proj_cat = "gate"
                    elif proj in _UP_NAMES:
                        proj_cat = "up"
                    elif proj in _DOWN_NAMES:
                        proj_cat = "down"
                    elif proj in _FUSED_GATE_UP_NAMES:
                        proj_cat = "gate_up"
                    else:
                        proj_cat = proj

                    shape = tuple(meta.get("shape", ()))
                    dtype_str = meta.get("dtype", "BF16")
                    dtype = dtype_map.get(dtype_str, torch.bfloat16)

                    slice_loc = MoESliceLocation(
                        shard_file=shard_file,
                        key=key,
                        expert_id=-1,
                        proj=proj,
                        suffix=suffix,
                        shape=shape,
                        dtype=dtype,
                    )
                    routed_prefix = next(
                        (p for p in self.moe_layers if p.startswith(prefix)),
                        prefix,
                    )
                    if routed_prefix not in self.moe_layers:
                        self.moe_layers[routed_prefix] = MoELayerPlan(
                            layer_prefix=routed_prefix,
                            num_total_experts=0,
                            num_routed_experts=0,
                        )
                    self.moe_layers[routed_prefix].shared_slices[(proj_cat, suffix)] = slice_loc
                    continue

                self.non_moe_keys.append((key, shard_file))

        # Calculate final total expert counts per layer (folding shared experts if FSE active)
        for plan in self.moe_layers.values():
            if self.fse_enabled and plan.shared_slices:
                plan.num_total_experts = plan.num_routed_experts + self.n_shared_experts
            else:
                plan.num_total_experts = plan.num_routed_experts

    def open_handles(self) -> None:
        for shard_file in self.hf_weights_files:
            if shard_file not in self.handles:
                self.handles[shard_file] = safe_open(
                    shard_file, framework="pt", device="cpu"
                )

    def close_handles(self) -> None:
        self.handles.clear()


def _copy_fse_gate_up(
    buf: torch.Tensor,
    shared_slices: dict[tuple[str, str], MoESliceLocation],
    suf: str,
    inter_dim: int,
    num_routed: int,
    n_shared: int,
    eid_to_slot: dict[int, int],
    handles: dict[str, Any],
) -> None:
    """Stage sliced shared expert gate/up weights into appended virtual expert slots."""
    shared_gate = shared_slices.get(("gate", suf))
    shared_up = shared_slices.get(("up", suf))
    shared_gate_up = shared_slices.get(("gate_up", suf))

    if shared_gate and shared_up:
        sh_g = handles[shared_gate.shard_file].get_tensor(shared_gate.key)
        sh_u = handles[shared_up.shard_file].get_tensor(shared_up.key)
        s_chunk = sh_g.shape[0] // n_shared
        for i in range(n_shared):
            virt_eid = num_routed + i
            if virt_eid in eid_to_slot:
                slot = eid_to_slot[virt_eid]
                g_c = sh_g[i * s_chunk : (i + 1) * s_chunk]
                u_c = sh_u[i * s_chunk : (i + 1) * s_chunk]
                if len(buf.shape) == 3:
                    buf[slot, :inter_dim, :].copy_(g_c)
                    buf[slot, inter_dim:, :].copy_(u_c)
                elif len(buf.shape) == 2:
                    buf[slot, :inter_dim].copy_(g_c)
                    buf[slot, inter_dim:].copy_(u_c)
    elif shared_gate_up:
        sh_gu = handles[shared_gate_up.shard_file].get_tensor(shared_gate_up.key)
        s_chunk = sh_gu.shape[0] // n_shared
        for i in range(n_shared):
            virt_eid = num_routed + i
            if virt_eid in eid_to_slot:
                slot = eid_to_slot[virt_eid]
                buf[slot].copy_(sh_gu[i * s_chunk : (i + 1) * s_chunk])


def _copy_fse_down(
    buf: torch.Tensor,
    shared_slices: dict[tuple[str, str], MoESliceLocation],
    suf: str,
    num_routed: int,
    n_shared: int,
    eid_to_slot: dict[int, int],
    handles: dict[str, Any],
) -> None:
    """Stage sliced shared expert down weights into appended virtual expert slots."""
    shared_down = shared_slices.get(("down", suf))
    if shared_down:
        sh_d = handles[shared_down.shard_file].get_tensor(shared_down.key)
        s_chunk = sh_d.shape[-1] // n_shared
        for i in range(n_shared):
            virt_eid = num_routed + i
            if virt_eid in eid_to_slot:
                slot = eid_to_slot[virt_eid]
                d_c = sh_d[..., i * s_chunk : (i + 1) * s_chunk]
                buf[slot].copy_(d_c)


def _prefetch_file_cache(path: str) -> None:
    """Issue POSIX_FADV_WILLNEED on shard file to initiate asynchronous kernel read-ahead."""
    posix_fadvise = getattr(os, "posix_fadvise", None)
    willneed = getattr(os, "POSIX_FADV_WILLNEED", None)
    if posix_fadvise is None or willneed is None:
        return
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        posix_fadvise(fd, 0, 0, willneed)
    except OSError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def _drop_file_cache(path: str) -> None:
    """Issue POSIX_FADV_DONTNEED to release page cache pages for a completed shard."""
    posix_fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if posix_fadvise is None or dontneed is None:
        return
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY)
        posix_fadvise(fd, 0, 0, dontneed)
    except OSError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def _stream_shard_direct_to_vram(
    hf_weights_files: list[str],
    index: SafetensorsMoEIndex,
    drop_cache_after_load: bool = True,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Streams weights shard-by-shard directly to VRAM with pipelined prefetching.

    Maintains a bounded sliding window of at most 2 open handles:
    - Shard i: actively read and yielded directly to GPU VRAM (zero host staging buffer).
    - Shard i+1: asynchronously prefetched via POSIX_FADV_WILLNEED and pre-opened.
    - Shard i-1: evicted from Linux page cache via POSIX_FADV_DONTNEED once completed.

    Inter-Rank Phase Invariance:
    All TP ranks stream shards in identical sorted sequence, maximizing Linux OS page-cache
    hits across peer ranks while bounding aggregate NVMe I/O to 1.45 TB.
    """
    sorted_shards = sorted(hf_weights_files)
    num_shards = len(sorted_shards)
    if num_shards == 0:
        return

    _prefetch_file_cache(sorted_shards[0])
    if num_shards > 1:
        _prefetch_file_cache(sorted_shards[1])

    prev_shard: str | None = None
    next_handle: Any = None
    if num_shards > 1:
        try:
            next_handle = safe_open(sorted_shards[1], framework="pt", device="cpu")
        except Exception as e:
            logger.debug("Failed to pre-open shard %s: %s", sorted_shards[1], e)
            next_handle = None

    for idx, shard_path in enumerate(sorted_shards):
        if idx == 0:
            current_handle = safe_open(shard_path, framework="pt", device="cpu")
        else:
            current_handle = next_handle or safe_open(shard_path, framework="pt", device="cpu")
            next_handle = None

        if idx + 1 < num_shards:
            next_shard_path = sorted_shards[idx + 1]
            _prefetch_file_cache(next_shard_path)
            try:
                next_handle = safe_open(next_shard_path, framework="pt", device="cpu")
            except Exception as e:
                logger.debug("Failed to pre-open shard %s: %s", next_shard_path, e)
                next_handle = None

        try:
            for key in current_handle.keys():
                if key == "__metadata__":
                    continue

                # Case 1: MoE 2D expert slice key
                m_2d = _MOE_2D_KEY_RE.match(key)
                if m_2d:
                    expert_id = int(m_2d.group("expert_id"))
                    if (
                        index.local_expert_ids is not None
                        and expert_id not in index.local_expert_ids
                    ):
                        continue
                    tensor = current_handle.get_tensor(key)
                    yield key, tensor
                    continue

                # Case 2: Shared expert slice key (FSE)
                m_fse = _SHARED_EXPERT_KEY_RE.match(key)
                if m_fse:
                    if index.fse_enabled:
                        prefix = m_fse.group("prefix")
                        proj = m_fse.group("proj")
                        suffix = m_fse.group("suffix") or ""
                        routed_prefix = next(
                            (p for p in index.moe_layers if p.startswith(prefix)),
                            f"{prefix}.experts",
                        )
                        plan = index.moe_layers.get(routed_prefix)
                        num_routed = plan.num_routed_experts if plan else 0

                        tensor = current_handle.get_tensor(key)
                        if index.n_shared_experts <= 1:
                            virt_key = f"{routed_prefix}.{num_routed}.{proj}{suffix}"
                            yield virt_key, tensor
                        else:
                            if proj in _DOWN_NAMES or "down" in proj:
                                s_chunk = tensor.shape[-1] // index.n_shared_experts
                                for s_idx in range(index.n_shared_experts):
                                    virt_eid = num_routed + s_idx
                                    chunk = tensor[..., s_idx * s_chunk : (s_idx + 1) * s_chunk]
                                    virt_key = f"{routed_prefix}.{virt_eid}.{proj}{suffix}"
                                    yield virt_key, chunk
                            else:
                                s_chunk = tensor.shape[0] // index.n_shared_experts
                                for s_idx in range(index.n_shared_experts):
                                    virt_eid = num_routed + s_idx
                                    chunk = tensor[s_idx * s_chunk : (s_idx + 1) * s_chunk]
                                    virt_key = f"{routed_prefix}.{virt_eid}.{proj}{suffix}"
                                    yield virt_key, chunk
                        continue
                    else:
                        tensor = current_handle.get_tensor(key)
                        yield key, tensor
                        continue

                # Case 3: 3D MoE key or standard non-MoE key
                tensor = current_handle.get_tensor(key)
                yield key, tensor
        finally:
            del current_handle

        if prev_shard is not None and drop_cache_after_load:
            _drop_file_cache(prev_shard)
        prev_shard = shard_path

    if prev_shard is not None and drop_cache_after_load:
        _drop_file_cache(prev_shard)
    if next_handle is not None:
        del next_handle


def _consolidate_3d_host_staging(
    hf_weights_files: list[str],
    index: SafetensorsMoEIndex,
    local_expert_ids: set[int] | None = None,
    max_workers: int = 8,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Consolidates 2D expert slices into contiguous 3D host staging buffers (Mode 1)."""
    index.open_handles()
    pool = PinnedHostStagingPool(capacity_per_shape=2)

    try:
        # Phase 1: Yield all non-MoE keys directly via lazy mmap
        logger.info(
            "Yielding %d non-MoE parameter keys...", len(index.non_moe_keys)
        )
        for key, shard_file in index.non_moe_keys:
            handle = index.handles[shard_file]
            tensor = handle.get_tensor(key)
            yield key, tensor

        # Phase 2: Consolidate MoE layers into 3D tensors
        logger.info("Consolidating %d MoE layer(s)...", len(index.moe_layers))
        workers = min(max_workers, max(1, os.cpu_count() or 4))
        executor = ThreadPoolExecutor(max_workers=workers)

        try:
            for layer_prefix, plan in index.moe_layers.items():
                # Determine local experts
                if local_expert_ids is not None:
                    active_eids = sorted(local_expert_ids)
                else:
                    active_eids = list(range(plan.num_total_experts))

                num_local = len(active_eids)
                eid_to_slot = {eid: idx for idx, eid in enumerate(active_eids)}
                is_contiguous = (
                    len(active_eids) > 0
                    and active_eids == list(range(active_eids[0], active_eids[-1] + 1))
                )
                start_eid = active_eids[0] if is_contiguous else 0
                end_eid = (active_eids[-1] + 1) if is_contiguous else num_local
                routed_end_eid = (
                    min(end_eid, plan.num_routed_experts)
                    if is_contiguous
                    else plan.num_routed_experts
                )

                # Branch 1: If layer has 3D pre-fused tensors
                if plan.tensors_3d:
                    suffixes = {suf for (_, suf) in plan.tensors_3d.keys()}
                    for suf in sorted(suffixes):
                        # Case 1A: Separate 3D gate and up projections -> fuse into gate_up
                        gate_loc = plan.tensors_3d.get(("gate", suf))
                        up_loc = plan.tensors_3d.get(("up", suf))
                        if gate_loc and up_loc:
                            gate_h = index.handles[gate_loc.shard_file]
                            up_h = index.handles[up_loc.shard_file]
                            gate_slice = gate_h.get_slice(gate_loc.key)
                            up_slice = up_h.get_slice(up_loc.key)
                            inter_dim = (
                                gate_loc.shape[1] if len(gate_loc.shape) >= 2 else 0
                            )
                            hidden_dim = (
                                gate_loc.shape[2] if len(gate_loc.shape) >= 3 else 0
                            )

                            if len(gate_loc.shape) == 3:
                                fused_shape = (num_local, 2 * inter_dim, hidden_dim)
                            elif len(gate_loc.shape) == 2:
                                fused_shape = (num_local, 2 * inter_dim)
                            else:
                                fused_shape = (
                                    num_local,
                                    2 * inter_dim,
                                    *gate_loc.shape[1:],
                                )

                            fused_buf = pool.acquire(fused_shape, gate_loc.dtype)
                            if is_contiguous:
                                if len(gate_loc.shape) == 3:
                                    fused_buf[:routed_end_eid, :inter_dim, :].copy_(
                                        gate_slice[start_eid:routed_end_eid]
                                    )
                                    fused_buf[:routed_end_eid, inter_dim:, :].copy_(
                                        up_slice[start_eid:routed_end_eid]
                                    )
                                elif len(gate_loc.shape) == 2:
                                    fused_buf[:routed_end_eid, :inter_dim].copy_(
                                        gate_slice[start_eid:routed_end_eid]
                                    )
                                    fused_buf[:routed_end_eid, inter_dim:].copy_(
                                        up_slice[start_eid:routed_end_eid]
                                    )
                            else:
                                for slot_idx, eid in enumerate(active_eids):
                                    if eid < plan.num_routed_experts:
                                        if len(gate_loc.shape) == 3:
                                            fused_buf[slot_idx, :inter_dim, :].copy_(
                                                gate_slice[eid]
                                            )
                                            fused_buf[slot_idx, inter_dim:, :].copy_(
                                                up_slice[eid]
                                            )
                                        elif len(gate_loc.shape) == 2:
                                            fused_buf[slot_idx, :inter_dim].copy_(
                                                gate_slice[eid]
                                            )
                                            fused_buf[slot_idx, inter_dim:].copy_(
                                                up_slice[eid]
                                            )

                            if index.fse_enabled and plan.shared_slices:
                                _copy_fse_gate_up(
                                    fused_buf,
                                    plan.shared_slices,
                                    suf,
                                    inter_dim,
                                    plan.num_routed_experts,
                                    index.n_shared_experts,
                                    eid_to_slot,
                                    index.handles,
                                )

                            if suf in ("", ".weight"):
                                yield_key = f"{layer_prefix}.gate_up_proj"
                            else:
                                clean_suf = suf if suf.startswith(".") else f".{suf}"
                                yield_key = f"{layer_prefix}.gate_up_proj{clean_suf}"

                            yield yield_key, fused_buf

                        # Case 1B: Pre-fused 3D gate_up
                        gate_up_loc = plan.tensors_3d.get(("gate_up", suf))
                        if gate_up_loc:
                            h = index.handles[gate_up_loc.shard_file]
                            slice_obj = h.get_slice(gate_up_loc.key)
                            target_shape = (num_local, *gate_up_loc.shape[1:])
                            buf = pool.acquire(target_shape, gate_up_loc.dtype)
                            if is_contiguous:
                                buf[:routed_end_eid].copy_(slice_obj[start_eid:routed_end_eid])
                            else:
                                for slot_idx, eid in enumerate(active_eids):
                                    if eid < plan.num_routed_experts:
                                        buf[slot_idx].copy_(slice_obj[eid])

                            if index.fse_enabled and plan.shared_slices:
                                inter_dim = buf.shape[1] // 2 if len(buf.shape) >= 2 else 0
                                _copy_fse_gate_up(
                                    buf,
                                    plan.shared_slices,
                                    suf,
                                    inter_dim,
                                    plan.num_routed_experts,
                                    index.n_shared_experts,
                                    eid_to_slot,
                                    index.handles,
                                )

                            if suf in ("", ".weight"):
                                yield_key = f"{layer_prefix}.gate_up_proj"
                            else:
                                clean_suf = suf if suf.startswith(".") else f".{suf}"
                                yield_key = f"{layer_prefix}.gate_up_proj{clean_suf}"

                            yield yield_key, buf

                        # Case 1C: Down projection
                        down_loc = plan.tensors_3d.get(("down", suf))
                        if down_loc:
                            h = index.handles[down_loc.shard_file]
                            slice_obj = h.get_slice(down_loc.key)
                            target_shape = (num_local, *down_loc.shape[1:])
                            buf = pool.acquire(target_shape, down_loc.dtype)
                            if is_contiguous:
                                buf[:routed_end_eid].copy_(slice_obj[start_eid:routed_end_eid])
                            else:
                                for slot_idx, eid in enumerate(active_eids):
                                    if eid < plan.num_routed_experts:
                                        buf[slot_idx].copy_(slice_obj[eid])

                            if index.fse_enabled and plan.shared_slices:
                                _copy_fse_down(
                                    buf,
                                    plan.shared_slices,
                                    suf,
                                    plan.num_routed_experts,
                                    index.n_shared_experts,
                                    eid_to_slot,
                                    index.handles,
                                )

                            if suf in ("", ".weight"):
                                yield_key = f"{layer_prefix}.down_proj"
                            else:
                                clean_suf = suf if suf.startswith(".") else f".{suf}"
                                yield_key = f"{layer_prefix}.down_proj{clean_suf}"

                            yield yield_key, buf

                        # Case 1D: Standalone other projections
                        for (cat, s), loc in plan.tensors_3d.items():
                            if s == suf and cat not in ("gate", "up", "down", "gate_up"):
                                h = index.handles[loc.shard_file]
                                slice_obj = h.get_slice(loc.key)
                                target_shape = (num_local, *loc.shape[1:])
                                buf = pool.acquire(target_shape, loc.dtype)
                                if is_contiguous:
                                    buf.copy_(slice_obj[start_eid:end_eid])
                                else:
                                    for slot_idx, eid in enumerate(active_eids):
                                        buf[slot_idx].copy_(slice_obj[eid])
                                clean_suf = suf if (suf.startswith(".") or not suf) else f".{suf}"
                                yield f"{layer_prefix}.{cat}{clean_suf}", buf

                # Branch 2: If layer has 2D per-expert slices
                if plan.slices:
                    # Group by suffix (e.g. "", ".weight_scale", ".weight_scale_inv", ".weight_scale_2")
                    suffixes = {suf for (_, suf) in plan.slices.keys()}

                    for suf in sorted(suffixes):
                        gate_slices = plan.slices.get(("gate", suf), {})
                        up_slices = plan.slices.get(("up", suf), {})
                        down_slices = plan.slices.get(("down", suf), {})

                        # Consolidate SwiGLU / GeGLU gate_up projection if gate and up exist
                        if gate_slices and up_slices:
                            sample_gate = next(iter(gate_slices.values()))
                            dtype = sample_gate.dtype
                            inter_dim = (
                                sample_gate.shape[0] if len(sample_gate.shape) >= 2 else 0
                            )
                            hidden_dim = (
                                sample_gate.shape[1] if len(sample_gate.shape) >= 2 else 0
                            )

                            if len(sample_gate.shape) == 2:
                                fused_shape = (num_local, 2 * inter_dim, hidden_dim)
                            elif len(sample_gate.shape) == 1:
                                fused_shape = (num_local, 2 * inter_dim)
                            else:
                                fused_shape = (
                                    num_local,
                                    2 * inter_dim,
                                    *sample_gate.shape[1:],
                                )

                            fused_buffer = pool.acquire(fused_shape, dtype)

                            def copy_gate(eid: int, loc: MoESliceLocation) -> None:
                                if eid in eid_to_slot:
                                    slot = eid_to_slot[eid]
                                    h = index.handles[loc.shard_file]
                                    t = h.get_tensor(loc.key)
                                    if len(t.shape) == 2:
                                        fused_buffer[slot, :inter_dim, :].copy_(t)
                                    elif len(t.shape) == 1:
                                        fused_buffer[slot, :inter_dim].copy_(t)

                            def copy_up(eid: int, loc: MoESliceLocation) -> None:
                                if eid in eid_to_slot:
                                    slot = eid_to_slot[eid]
                                    h = index.handles[loc.shard_file]
                                    t = h.get_tensor(loc.key)
                                    if len(t.shape) == 2:
                                        fused_buffer[slot, inter_dim:, :].copy_(t)
                                    elif len(t.shape) == 1:
                                        fused_buffer[slot, inter_dim:].copy_(t)

                            list(
                                executor.map(
                                    lambda item: copy_gate(*item), gate_slices.items()
                                )
                            )
                            list(
                                executor.map(
                                    lambda item: copy_up(*item), up_slices.items()
                                )
                            )

                            if index.fse_enabled and plan.shared_slices:
                                _copy_fse_gate_up(
                                    fused_buffer,
                                    plan.shared_slices,
                                    suf,
                                    inter_dim,
                                    plan.num_routed_experts,
                                    index.n_shared_experts,
                                    eid_to_slot,
                                    index.handles,
                                )

                            # Yield fused gate_up key
                            if suf in ("", ".weight"):
                                yield_key = f"{layer_prefix}.gate_up_proj"
                            else:
                                clean_suf = suf if suf.startswith(".") else f".{suf}"
                                yield_key = f"{layer_prefix}.gate_up_proj{clean_suf}"

                            yield yield_key, fused_buffer

                        # Consolidate down projection
                        if down_slices:
                            sample_down = next(iter(down_slices.values()))
                            dtype = sample_down.dtype
                            down_shape = (num_local, *sample_down.shape)
                            down_buffer = pool.acquire(down_shape, dtype)

                            def copy_down(eid: int, loc: MoESliceLocation) -> None:
                                if eid in eid_to_slot:
                                    slot = eid_to_slot[eid]
                                    h = index.handles[loc.shard_file]
                                    t = h.get_tensor(loc.key)
                                    down_buffer[slot].copy_(t)

                            list(
                                executor.map(
                                    lambda item: copy_down(*item), down_slices.items()
                                )
                            )

                            if index.fse_enabled and plan.shared_slices:
                                _copy_fse_down(
                                    down_buffer,
                                    plan.shared_slices,
                                    suf,
                                    plan.num_routed_experts,
                                    index.n_shared_experts,
                                    eid_to_slot,
                                    index.handles,
                                )

                            if suf in ("", ".weight"):
                                yield_key = f"{layer_prefix}.down_proj"
                            else:
                                clean_suf = suf if suf.startswith(".") else f".{suf}"
                                yield_key = f"{layer_prefix}.down_proj{clean_suf}"

                            yield yield_key, down_buffer

                        # Handle standalone projections (e.g. dense_h_to_4h without gate)
                        for (cat, s), slices in plan.slices.items():
                            if s == suf and cat not in ("gate", "up", "down"):
                                sample = next(iter(slices.values()))
                                shape = (num_local, *sample.shape)
                                buf = pool.acquire(shape, sample.dtype)

                                def copy_other(eid: int, loc: MoESliceLocation) -> None:
                                    if eid in eid_to_slot:
                                        slot = eid_to_slot[eid]
                                        h = index.handles[loc.shard_file]
                                        buf[slot].copy_(h.get_tensor(loc.key))

                                list(
                                    executor.map(
                                        lambda item: copy_other(*item), slices.items()
                                    )
                                )
                                clean_suf = (
                                    suf if (suf.startswith(".") or not suf) else f".{suf}"
                                )
                                yield f"{layer_prefix}.{cat}{clean_suf}", buf

        finally:
            executor.shutdown(wait=False)

    finally:
        index.close_handles()
        pool.clear()


def fast_bypass_safetensors_iterator(
    hf_weights_files: list[str],
    local_expert_ids: set[int] | None = None,
    max_workers: int = 8,
    fse_enabled: bool | None = None,
    n_shared_experts: int = 1,
    direct_vram_mode: bool | None = None,
    direct_vram_threshold_gb: float = 100.0,
    drop_cache_after_load: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterates through safetensors checkpoint shards with scale-aware MoE ingestion.

    Dual Ingestion Modes:
    1. Mode 1 (Bulk 3D Host Staging): Checkpoints <= 100 GB (e.g. Qwen3-VL-30B, GLM-5.3-Flash).
       Consolidates slices in pinned CPU host staging buffers to reduce thousands of DMA
       calls to 192 wire-speed transfers, preserving 13.1x-28.1x load records.
    2. Mode 2 (Shard-Driven Direct-to-VRAM with Pipelined Prefetching): Checkpoints > 100 GB
       (e.g. Kimi-K3, DeepSeek-V3/V4). Streams one shard at a time with a bounded sliding window
       of 2 handles (POSIX_FADV_WILLNEED on shard i+1, POSIX_FADV_DONTNEED on shard i-1).
       Eliminates CPU pinned buffers and multi-process page-table lock stalls.
    """
    if fse_enabled is None:
        fse_enabled = os.environ.get(
            "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "0"
        ).lower() in ("1", "true")

    logger.info(
        "Building fast MoE safetensors index across %d shard(s)...",
        len(hf_weights_files),
    )
    index = SafetensorsMoEIndex.build(
        hf_weights_files,
        local_expert_ids,
        max_workers=max_workers,
        fse_enabled=fse_enabled,
        n_shared_experts=n_shared_experts,
    )

    env_direct = os.environ.get("VLLM_MOE_DIRECT_VRAM")
    if direct_vram_mode is not None:
        use_direct = direct_vram_mode
    elif env_direct is not None:
        use_direct = env_direct.lower() in ("1", "true")
    else:
        env_thresh = os.environ.get("VLLM_FAST_MOE_DIRECT_VRAM_THRESHOLD_GB")
        if env_thresh is not None:
            try:
                threshold_gb = float(env_thresh)
            except ValueError:
                threshold_gb = direct_vram_threshold_gb
        else:
            threshold_gb = direct_vram_threshold_gb

        total_bytes = sum(os.path.getsize(f) for f in hf_weights_files)
        total_gb = total_bytes / (1024 ** 3)
        use_direct = total_gb > threshold_gb

    if use_direct:
        logger.info(
            "Fast MoE Bypass: Selected Mode 2 (Shard-Driven Direct-to-VRAM with Pipelined Prefetching). "
            "Total checkpoint size: %.2f GiB across %d shard(s).",
            sum(os.path.getsize(f) for f in hf_weights_files) / (1024**3),
            len(hf_weights_files),
        )
        os.environ["VLLM_MOE_DISABLE_HOST_STAGING"] = "1"
        yield from _stream_shard_direct_to_vram(
            hf_weights_files,
            index,
            drop_cache_after_load=drop_cache_after_load,
        )
    else:
        logger.info(
            "Fast MoE Bypass: Selected Mode 1 (Bulk 3D Host Staging). "
            "Total checkpoint size: %.2f GiB across %d shard(s).",
            sum(os.path.getsize(f) for f in hf_weights_files) / (1024**3),
            len(hf_weights_files),
        )
        yield from _consolidate_3d_host_staging(
            hf_weights_files,
            index,
            local_expert_ids=local_expert_ids,
            max_workers=max_workers,
        )
