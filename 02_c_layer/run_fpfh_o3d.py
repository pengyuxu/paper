"""C2 batch (open3d reimplementation of Tatsuma's vtk2feat).

Faithful port of external/shrec2025_protein/src/{vtk2feat.cpp,utils.hpp}:
 - vertices normalized by mean + max vertex radius
 - 30000 area-weighted points using the OFFICIAL i8_sobol sequence (libsobol.so)
 - PCL-equivalent normals radius 0.1 (oriented toward origin like PCL viewpoint),
   FPFH radius 0.2 via Open3D
 - same 612-float layout: 33 means, 561 upper-tri sample cov, 16-bin potential
   histogram (normalized), potential mean, potential population variance
Output: results_final/fpfh/<key>.dat (612 float32 = 2448 bytes). Resumable.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPEN3D_CPU_RELEASE_TYPE", "Release")
import sys, ctypes
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
VTKDIR = BASE.parent / "data"
SO = BASE / "external/shrec2025_protein/bin/libsobol.so"
sys.path.insert(0, str(BASE / "external/guerra"))

N_POINTS = 30000
N_BINS = 16

_sobol = None
_o3d = None


def _init_worker():
    global _sobol, _o3d
    _sobol = ctypes.CDLL(str(SO))
    _sobol.sobol_reset.argtypes = []
    _sobol.sobol_draws.argtypes = [ctypes.c_longlong, ctypes.POINTER(ctypes.c_double)]
    import open3d
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
    _o3d = open3d


def _draws(n):
    buf = np.empty(n * 2, dtype=np.float64)
    _sobol.sobol_reset()
    _sobol.sobol_draws(ctypes.c_longlong(n),
                       buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    return buf.reshape(n, 2)


def sample_points(V, F, P, n_points):
    # normalize: mean-center then divide by max vertex radius (utils.hpp)
    P = np.asarray(P, dtype=np.float32).reshape(-1)
    V = V.astype(np.float64).copy()
    V -= V.mean(axis=0, keepdims=True)
    maxd = np.sqrt((V ** 2).sum(axis=1)).max()
    V /= maxd

    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    a = (v1 - v0)
    b = (v2 - v0)
    cr = np.cross(a, b)
    areas = 0.5 * np.sqrt((cr ** 2).sum(axis=1))
    areas = np.where(np.isfinite(areas), areas, 0.0)
    mesh_area = areas.sum()
    if mesh_area <= 0:
        raise RuntimeError("zero mesh area")

    # polygons visited in ascending (area, index) order (std::sort of pairs)
    order = np.lexsort((np.arange(len(areas)), areas))
    ratio = areas[order] / mesh_area
    raw = n_points * ratio
    base = np.floor(raw)
    cum = np.cumsum(raw - base)
    carry = np.floor(cum)
    extra = carry - np.floor(np.concatenate(([0.0], cum[:-1])))
    counts = (base + extra).astype(np.int64)

    # total may be < n_points: official repeats the whole polygon loop; every
    # pass has identical counts, so replicate until reaching n_points.
    ord_parts, cnt_parts = [order], [counts]
    while int(np.concatenate(cnt_parts).sum()) < n_points and len(cnt_parts) <= 8:
        ord_parts.append(order)
        cnt_parts.append(counts)
    order = np.concatenate(ord_parts)
    counts = np.concatenate(cnt_parts)
    total = min(int(counts.sum()), n_points)
    draws = _draws(total)

    # triangle assigned to each sequential draw
    tri_of_draw = np.repeat(order, counts)[:total]
    r0, r1d = draws[:, 0], draws[:, 1]
    r1 = np.sqrt(np.abs(r0))
    r2 = np.abs(r1d)
    w0 = (1.0 - r1)
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    pts = (w0[:, None] * V[F[tri_of_draw, 0]] +
           w1[:, None] * V[F[tri_of_draw, 1]] +
           w2[:, None] * V[F[tri_of_draw, 2]])
    pv = P[F[tri_of_draw]]
    spot = w0 * pv[:, 0] + w1 * pv[:, 1] + w2 * pv[:, 2]
    return pts.astype(np.float64), spot.astype(np.float32)


def potential_histogram(data, n_bins):
    h = np.zeros(n_bins, dtype=np.float32)
    lo, hi = float(data.min()), float(data.max())
    rng = hi - lo
    if rng <= 0:
        h[0] = float(data.size)
        return h
    w = rng / n_bins
    idx = ((data - lo) / w).astype(np.int64)
    idx = np.clip(idx, 0, n_bins - 1)
    np.add.at(h, idx, 1.0)
    s = h.sum()
    if s > 0:
        h /= s
    return h


def one(job):
    key, vtk_path, outdir = job
    outdir = Path(outdir)
    dst = outdir / (key + ".dat")
    if dst.exists() and dst.stat().st_size == 612 * 4:
        return key, "skip"
    try:
        from src.data_reader import convertVTK_to_numpy
        points, triangles, potential, _normpot = convertVTK_to_numpy(vtk_path)
        pts, spot = sample_points(np.asarray(points), np.asarray(triangles),
                                  np.asarray(potential, dtype=np.float32), N_POINTS)
        pcd = _o3d.geometry.PointCloud()
        pcd.points = _o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(_o3d.geometry.KDTreeSearchParamRadius(radius=0.1))
        nrm = np.asarray(pcd.normals)
        # PCL NormalEstimation orients toward viewpoint (0,0,0): replicate.
        flip = (nrm * pts).sum(axis=1) > 0
        nrm[flip] *= -1
        pcd.normals = _o3d.utility.Vector3dVector(nrm)
        fpfh = _o3d.pipelines.registration.compute_fpfh_feature(
            pcd, _o3d.geometry.KDTreeSearchParamRadius(radius=0.2))
        f = np.asarray(fpfh.data, dtype=np.float64)  # (33, N)

        means = f.mean(axis=1)
        cov = np.cov(f, bias=False)              # sample covariance, (33,33)
        iu = np.triu_indices(33)
        cov_upper = cov[iu]
        h = potential_histogram(spot, N_BINS)
        pmean = np.float32(spot.mean())
        pvar = np.float32(((spot - spot.mean()) ** 2).mean())

        feat = np.concatenate([
            means.astype(np.float32),
            cov_upper.astype(np.float32),
            h,
            np.array([pmean, pvar], dtype=np.float32),
        ])
        if feat.shape[0] != 612:
            return key, "FAIL:featlen=%d" % feat.shape[0]
        tmp = dst.with_suffix(".dat.tmp%d" % os.getpid())
        feat.astype(np.float32).tofile(tmp)
        os.replace(tmp, dst)
        return key, "ok"
    except Exception as e:  # noqa
        return key, "FAIL:%r" % e


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "results_final/fpfh"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = []
    off = pd.read_csv(VTKDIR.parent / "train_set.csv")
    off.columns = ["vtk_key", "c"]
    for k in off.vtk_key:
        k = str(k).replace(".vtk", "")
        jobs.append((k.replace(":", "_"), str(VTKDIR / (k + ".vtk")), str(outdir)))
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    for pid in te.anonymised_protein_id:
        pid = str(pid).replace(".vtk", "")
        jobs.append((pid, str(VTKDIR / (pid + ".vtk")), str(outdir)))

    fails = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for i, (key, st) in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            if st not in ("ok", "skip"):
                print(key, st, flush=True)
                fails.append((key, st))
            if i % 100 == 0:
                print("[%d/%d] %s %s fails=%d" %
                      (i + 1, len(jobs), key, st, len(fails)), flush=True)
    (outdir.parent / "fpfh_failures.txt").write_text(
        "\n".join("%s\t%s" % x for x in fails))
    print("done fails=%d" % len(fails))


if __name__ == "__main__":
    main()
