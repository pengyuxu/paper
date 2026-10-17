"""Rotation TTA over the 24 proper rotations of the cube (octahedral group).

Point clouds (B,N,8): channels 0:3 xyz, 3:6 normals, 6:8 potentials.
xyz and normals are rotated by the same R; potentials stay unchanged.
Probabilities from the 24 views are averaged.
"""
from __future__ import annotations
import itertools
import numpy as np
import torch


def cube_rotations(dtype=np.float32):
    rots = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3), dtype)
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for signs in itertools.product([1.0, -1.0], repeat=3):
            R = P * np.array(signs, dtype)[:, None]
            if np.linalg.det(R) > 0:
                rots.append(R)
    assert len(rots) == 24
    return rots


def apply_rotation(pts: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    out = pts.clone()
    out[..., 0:3] = pts[..., 0:3] @ R.T
    out[..., 3:6] = pts[..., 3:6] @ R.T
    return out


@torch.no_grad()
def tta_predict(model, pts: torch.Tensor, rotations=None, device="cuda"):
    """Return mean log-probability over rotations. pts: (B,N,C) tensor."""
    if rotations is None:
        rotations = cube_rotations()
    probs_sum, n = None, 0
    for R in rotations:
        Rt = torch.as_tensor(R, dtype=pts.dtype, device=pts.device)
        o, _ = model(apply_rotation(pts, Rt))
        if o.dim() == 3:
            o = o.mean(dim=1)
        p = torch.softmax(o.float(), dim=1)
        probs_sum = p if probs_sum is None else probs_sum + p
        n += 1
    return torch.log(probs_sum / n + 1e-300)
