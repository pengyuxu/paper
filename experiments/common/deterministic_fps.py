"""Deterministic replacement for the network's stochastic farthest-point
sampling (newcode/models/riconv2_utils_features.py:88,
`farthest = torch.randint(0, N, (B,))`).

Modes:
  - "first" : D0 is always point index 0 (bitwise-reproducible single pass);
  - "gen"   : Ds draws D0 from a fixed-seed CPU torch.Generator on every FPS
              call (reset to `seed` at the start of each pass / split); this
              is the "fixed-seed generator" deterministic variant of the
              paper's Table 4, distinct from the canonical-index D0;
  - "seeded": D0 is a decorrelated splitmix64 hash of
              (seed, pass_index, fps_call_index, batch_row), giving K
              reproducible-but-distinct MC-FPS passes.

Usage:
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 MUST be exported before torch import.
    patcher = FPSPatcher(mode="seeded", seed=42).install()
    for k in range(K):
        patcher.pass_index = k
        ...forward...
    patcher.restore()

Verified bitwise on A40 / torch 2.x (experiments/tests/fps_determinism_test.py):
"first" gives max|Δlogits|=0 across repeated eval passes; "seeded" reproduces
the same pass_index bit-for-bit while distinct passes differ (~3e-4 mean |Δ|).
"""
from __future__ import annotations
import sys
import types
import numpy as np
import torch

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    x += np.uint64(0x9E3779B97F4A7C15)
    z = x.copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def make_fps(mode: str, seed: int, state: dict):
    def farthest_point_sample(xyz, npoint):
        device = xyz.device
        B, N, _ = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        if mode == "first":
            farthest = torch.zeros(B, dtype=torch.long, device=device)
        elif mode == "gen":
            # Ds: fixed-seed CPU generator, one reproducible random D0 per row
            farthest = torch.randint(
                0, N, (B,), generator=state["gen"]).long().to(device)
        else:
            call = state["call"]
            state["call"] = call + 1
            base = (np.uint64(seed) * np.uint64(1000003)
                    + np.uint64(state["pass_index"]) * np.uint64(7919)
                    + np.uint64(call) * np.uint64(31))
            key = (base * np.uint64(281474976710656)
                   + np.arange(B, dtype=np.uint64)) & _MASK64
            h = _splitmix64(key)
            farthest = torch.from_numpy(
                (h % np.uint64(N)).astype(np.int64)).long().to(device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids
    return farthest_point_sample


def _eigh_det(M):
    """Symmetric eigendecomposition with a deterministic eigenvector sign.

    GPU ``Tensor.symeig`` (used by compute_LRA*) returns eigenvectors whose
    sign flips arbitrarily between calls -- physically degenerate but enough
    to perturb logits. ``linalg.eigh`` + fixing the sign of the
    largest-magnitude component removes the ambiguity.
    """
    ev, vec = torch.linalg.eigh(M, UPLO="U")
    signs = torch.gather(
        vec, -1, vec.abs().argmax(dim=-1, keepdim=True)).sign()
    return ev, vec * signs


def compute_LRA_one_det(group_xyz, weighting=False):
    B, S, N, C = group_xyz.shape
    dists = torch.norm(group_xyz, dim=-1, keepdim=True)
    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3, 2), weights * group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3, 2), group_xyz)
    _, vec = _eigh_det(M)
    LRA = vec[:, :, :, 0]
    return LRA / torch.norm(LRA, dim=-1, keepdim=True)


def compute_LRA_det(xyz, weighting=False, nsample=64):
    B, N, _ = xyz.shape
    dists = torch.cdist(xyz, xyz)
    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)
    bidx = torch.arange(B, device=xyz.device).view(B, 1, 1)
    group_xyz = xyz[bidx, idx]  # == index_points(xyz, idx), [B, N, K, 3]
    group_xyz = group_xyz - xyz.unsqueeze(2)
    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3, 2), weights * group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3, 2), group_xyz)
    _, vec = _eigh_det(M)
    LRA = vec[:, :, :, 0]
    return LRA / torch.norm(LRA, dim=-1, keepdim=True)


class FPSPatcher:
    def __init__(self, mode="first", seed=42):
        assert mode in ("first", "seeded", "gen")
        self.mode, self.seed = mode, seed
        self._state = {"pass_index": 0, "call": 0,
                       "gen": torch.Generator("cpu").manual_seed(seed)}
        self._saved = []

    @property
    def pass_index(self):
        return self._state["pass_index"]

    @pass_index.setter
    def pass_index(self, k):
        self._state["pass_index"] = int(k)
        self._state["call"] = 0
        if self.mode == "gen":
            # each pass / split starts from the same fixed generator state
            self._state["gen"].manual_seed(self.seed + int(k))

    def install(self, *modules):
        if not modules:
            modules = [m for name, m in list(sys.modules.items())
                       if isinstance(m, types.ModuleType)
                       and "riconv2_utils" in name
                       and hasattr(m, "farthest_point_sample")]
        # FPS is not the only stochastic piece: cuDNN picks nondeterministic
        # conv algorithms by default, and symeig returns random-sign
        # eigenvectors. Pin them all for bitwise reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        fps = make_fps(self.mode, self.seed, self._state)
        for mod in modules:
            self._saved.append((mod, "farthest_point_sample",
                                mod.farthest_point_sample))
            mod.farthest_point_sample = fps
            if hasattr(mod, "compute_LRA_one"):
                self._saved.append((mod, "compute_LRA_one",
                                    mod.compute_LRA_one))
                mod.compute_LRA_one = compute_LRA_one_det
            if hasattr(mod, "compute_LRA"):
                self._saved.append((mod, "compute_LRA", mod.compute_LRA))
                mod.compute_LRA = compute_LRA_det
        return self

    def restore(self):
        for mod, name, fn in self._saved:
            setattr(mod, name, fn)
        self._saved = []
