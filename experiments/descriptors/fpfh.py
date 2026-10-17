"""C2 FPFH descriptor (paper §4.6), rebuilt from the raw VTK surfaces.

Pipeline per surface:
  1. mesh mean-centred (vertex mean) and divided by its maximum vertex
     radius (|V|inf after centring), putting every surface in [-1,1].
  2. 30,000 area-weighted samples driven by a 3-D Sobol quasi-random sequence
     (replaces the "official i8 Sobol" binary; scipy.stats.qmc.Sobol,
     scramble=False; disclosed in the paper).
  3. normals estimated by Open3D radius search r_n=0.1 and oriented toward
     the origin (PCL-style viewpoint flip); FPFH at r_f=0.2 -> 33 channels.
  4. per-point Coulomb potential phi transferred from VTK vertices by nearest
     vertex in the SAME centred/scaled frame and per-mesh min-max normalised.
  5. pooled to 612 dims: 33 channel means + 561 unbiased upper-triangular
     covariance entries + 16-bin potential histogram (mass-normalised) +
     potential mean and variance.

The descriptor is version-independent: built once over the 9,244 train VTKs
and the 2,311 test VTKs; version split files only re-order/slice the matrix.

Outputs (per --split): experiments/results/desc/fpfh_{split}.npz
  {ids, X (N,612) float32}
"""
from __future__ import annotations
import argparse
import os, sys, tempfile
import multiprocessing as mp

# CPU-only: the installed Open3D wheel carries a CUDA pybind; importing it in
# forked workers (while the GPU queue is training) deadlocks the pool.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from experiments.common import paths as vp
from experiments.preprocess.build_pointclouds import _read_polydata, _polydata_to_stl

import trimesh
import open3d as o3d
from vtk.util.numpy_support import vtk_to_numpy
from scipy.stats import qmc

N_SAMPLE = 30000
R_N, R_F = 0.1, 0.2
N_BINS = 16
DIM = 33 + 33 * 34 // 2 + N_BINS + 2   # 612
OUT_DIR = vp.EXP_DIR / "results" / "desc"

_sobol = None


def sobol_unit(n, dim=3):
    # Sobol balance requires a power-of-two draw: draw the next 2**k and slice.
    m = int(2 ** np.ceil(np.log2(n)))
    u = qmc.Sobol(d=dim, scramble=False, seed=20250217).random(m)
    return u[:n]


def area_sample(mesh, u3):
    """u3: (n,3) quasi-random. Returns sampled xyz (n,3) and face index."""
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    cdf = np.cumsum(areas)
    cdf /= cdf[-1]
    fi = np.searchsorted(cdf, np.clip(u3[:, 0], 0.0, 1.0 - 1e-12), side="right")
    fi = np.clip(fi, 0, len(mesh.faces) - 1)
    u1, u2 = u3[:, 1], u3[:, 2]
    sq = np.sqrt(u1)
    b0 = 1.0 - sq
    b1 = sq * (1.0 - u2)
    b2 = sq * u2
    tri = mesh.vertices[mesh.faces[fi]]          # (n,3,3)
    pts = (b0[:, None] * tri[:, 0] +
           b1[:, None] * tri[:, 1] +
           b2[:, None] * tri[:, 2])
    return pts, fi


def process_one(job):
    vtk_path, = job
    tmp = None
    try:
        poly = _read_polydata(vtk_path)
        verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        phi0 = vtk_to_numpy(
            poly.GetPointData().GetArray("Potential")).astype(np.float64)
        fd, tmp = tempfile.mkstemp(suffix=".stl"); os.close(fd)
        _polydata_to_stl(poly, tmp)
        mesh = trimesh.load(tmp, force="mesh")

        c = verts.mean(0)
        r = float(np.linalg.norm(verts - c, axis=1).max())
        if not np.isfinite(r) or r < 1e-12:
            raise RuntimeError("degenerate radius")
        mesh.apply_translation(-c)
        mesh.apply_scale(1.0 / r)
        verts_t = (verts - c) / r

        u = sobol_unit(N_SAMPLE)
        pts, _ = area_sample(mesh, u)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(radius=R_N))
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        fpfh = np.asarray(
            o3d.pipelines.registration.compute_fpfh_feature(
                pcd, o3d.geometry.KDTreeSearchParamRadius(radius=R_F)
            ).data).T.astype(np.float32)          # (30000, 33)

        # potential at samples via nearest VTK vertex in the aligned frame
        from scipy.spatial import cKDTree
        nn = cKDTree(verts_t).query(pts, k=1)[1]
        phi = phi0[nn].astype(np.float64)
        lo, hi = float(phi.min()), float(phi.max())
        phi = (phi - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(phi)

        F = fpfh.astype(np.float64)
        means = F.mean(0)
        C = np.cov(F, rowvar=False, ddof=1)
        iu = np.triu_indices(33)
        covv = C[iu]
        hist, _ = np.histogram(phi, bins=N_BINS, range=(0.0, 1.0))
        hist = hist.astype(np.float64) / max(hist.sum(), 1)
        sig = np.concatenate([
            means, covv, hist,
            [phi.mean(), phi.var()]])
        return np.nan_to_num(sig.astype(np.float32), nan=0.0)
    except Exception as e:  # noqa: BLE001
        return ("FAIL:%s:%s" % (type(e).__name__, e))
    finally:
        if tmp and os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass


def ids_for(split):
    if split == "train":
        df = pd.read_csv(vp.csv_dir("paper") / "train_set_2.csv")
        return df["protein_id"].astype(str).values, vp.RAW_TRAIN_VTK
    names = []
    for ver in ("paper", "updated"):
        cfg = vp.version_cfg(ver)
        tdf = pd.read_csv(cfg["dataset_dir"] / cfg["test_list_file"])
        for v in tdf[cfg["test_id_col"]]:
            names.append(str(v).replace(".vtk", ""))
    return np.array(sorted(set(names), key=int)), vp.RAW_TEST_VTK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, raw = ids_for(args.split)
    if args.limit:
        ids = ids[:args.limit]
    jobs = [(str(raw / (i + ".vtk")),) for i in ids]
    print("FPFH split=%s n=%d workers=%d dim=%d" %
          (args.split, len(jobs), args.workers, DIM), flush=True)
    out = np.zeros((len(jobs), DIM), np.float32)
    fails = []
    t0 = __import__("time").time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for k, sig in enumerate(pool.imap(process_one, jobs, chunksize=2)):
            if isinstance(sig, str):
                fails.append((ids[k], sig)); out[k] = 0
            else:
                out[k] = sig
            if (k + 1) % 500 == 0 or k + 1 == len(jobs):
                print("  %d/%d elapsed=%.0fs fails=%d" %
                      (k + 1, len(jobs), __import__("time").time() - t0,
                       len(fails)), flush=True)
    path = OUT_DIR / ("fpfh_%s.npz" % args.split)
    np.savez_compressed(path, ids=ids.astype(str), X=out)
    print("saved", path, "fails=", len(fails), flush=True)
    for i, msg in fails[:10]:
        print("  FAIL", i, msg)
    if len(fails) > len(ids) * 0.01:
        sys.exit(2)


if __name__ == "__main__":
    main()
