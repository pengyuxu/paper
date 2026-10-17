"""C1 3D Zernike descriptor (paper §4.6) via the canonical Novotni/Bonn
toolchain (the Zernike stage shipped inside Map2Zernike [22]).

Pipeline per surface (identical surface frame to the FPFH build: vertex-mean
centred, divided by max vertex radius):
  * dense (65,536) area-weighted samples (Sobol QMC, same sampler as FPFH);
  * surface voxelisation on a 64^3 grid: each voxel hit by a surface sample is
    set to 1 -- this is the canonical thin shell used with the c = 0.5 surface
    cutoff (the documented c=1.7 web-server value yields a ~3.4 voxel shell;
    c=0.5 is the one-voxel rasterised surface);
  * each shell voxel receives the sign of the raw (signed) Coulomb potential
    of its nearest surface sample, producing the whole / positive / negative
    binary blocks;
  * the three binary grids are passed to the Novotni `zernike3d` binary
    (order 20), which performs the canonical COG + 2*RMS-radius normalisation,
    unit-ball masking, geometric moments and the 121 invariants;
  * signature = 121*3 = 363 invariants + the positive-shell volume fraction
    f+ = n+/(n++n-) used by the c1 volume-ratio gate (the 364th coordinate).

Outputs: experiments/results/desc/dzd_{train,test}.npz {ids, X(N,364)}.
The Bonn binary lives at methods/3dzd/ZernikeMoments/build/examples/zernike3d.
"""
from __future__ import annotations
import argparse
import os, sys, tempfile, subprocess, struct, time
import multiprocessing as mp

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from experiments.common import paths as vp
from experiments.preprocess.build_pointclouds import _read_polydata, _polydata_to_stl
from experiments.descriptors.fpfh import sobol_unit, area_sample

import trimesh
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

GRID = 64
ORDER = 20
N_SAMPLE = 65536
N_INV = 121
DIM = 3 * N_INV + 1                       # 364
BIN = str(vp.REPO_ROOT /
          "methods/3dzd/ZernikeMoments/build/examples/zernike3d")
OUT_DIR = vp.EXP_DIR / "results" / "desc"


def run_zernike(grid, tmpbase):
    raw = tmpbase + ".raw"
    grid.astype(np.float32).tofile(raw)
    p = subprocess.run([BIN, raw, str(ORDER)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("zernike3d failed: %s" % p.stderr.decode()[:200])
    with open(tmpbase + ".inv", "rb") as f:
        data = f.read()
    n = struct.unpack("i", data[:4])[0]
    inv = np.array(struct.unpack("%df" % n, data[4:4 + 4 * n]), np.float32)
    os.remove(raw); os.remove(tmpbase + ".inv")
    return inv


def process_one(job):
    vtk_path, = job
    tmpd = tempfile.mkdtemp(prefix="dzd_")
    try:
        poly = _read_polydata(vtk_path)
        verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        phi0 = vtk_to_numpy(
            poly.GetPointData().GetArray("Potential")).astype(np.float64)
        stl = os.path.join(tmpd, "m.stl")
        _polydata_to_stl(poly, stl)
        mesh = trimesh.load(stl, force="mesh")

        c = verts.mean(0)
        r = float(np.linalg.norm(verts - c, axis=1).max())
        mesh.apply_translation(-c)
        mesh.apply_scale(1.0 / r)
        verts_t = (verts - c) / r

        pts, _ = area_sample(mesh, sobol_unit(N_SAMPLE))
        nn = cKDTree(verts_t).query(pts, k=1)[1]
        phi = phi0[nn]

        vc = np.floor(31.5 + 31.0 * pts).astype(np.int64)  # voxel index
        vc = np.clip(vc, 0, GRID - 1)
        flat = (vc[:, 0] * GRID + vc[:, 1]) * GRID + vc[:, 2]
        shell, inv_idx = np.unique(flat, return_inverse=True)

        # sign per shell voxel = potential of the closest surface sample
        centers = (np.stack(np.unravel_index(shell, (GRID, GRID, GRID)),
                            1).astype(np.float64) + 0.5 - 31.5) / 31.0
        near = cKDTree(pts).query(centers, k=1)[1]
        pos = phi[near] > 0.0
        npos, nneg = int(pos.sum()), int((~pos).sum())
        fp = npos / max(npos + nneg, 1)

        g = np.zeros(GRID ** 3, np.float32)
        g[shell] = 1.0
        gp = np.zeros(GRID ** 3, np.float32); gp[shell[pos]] = 1.0
        gm = np.zeros(GRID ** 3, np.float32); gm[shell[~pos]] = 1.0

        blocks = []
        for gi, tag in ((g, "w"), (gp, "p"), (gm, "n")):
            g3 = gi.reshape(GRID, GRID, GRID)
            # C++ expects flat index (z*dim+y)*dim+x : transpose to (x,y,z)
            g3 = np.transpose(g3, (2, 1, 0)).copy()
            blocks.append(run_zernike(g3.reshape(-1),
                                      os.path.join(tmpd, tag)))
        sig = np.concatenate(blocks + [[fp]]).astype(np.float32)
        return np.nan_to_num(sig, nan=0.0)
    except Exception as e:  # noqa: BLE001
        return ("FAIL:%s:%s" % (type(e).__name__, e))
    finally:
        try:
            for f in os.listdir(tmpd):
                os.remove(os.path.join(tmpd, f))
            os.rmdir(tmpd)
        except OSError:
            pass


def ids_for(split):
    if split == "train":
        df = pd.read_csv(vp.csv_dir("paper") / "train_set_2.csv")
        return df["protein_id"].astype(str).values, vp.RAW_TRAIN_VTK
    names = set()
    for ver in ("paper", "updated"):
        cfg = vp.version_cfg(ver)
        tdf = pd.read_csv(cfg["dataset_dir"] / cfg["test_list_file"])
        for v in tdf[cfg["test_id_col"]]:
            names.add(str(v).replace(".vtk", ""))
    return np.array(sorted(names, key=int)), vp.RAW_TEST_VTK


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
    print("DZD split=%s n=%d workers=%d dim=%d" %
          (args.split, len(jobs), args.workers, DIM), flush=True)
    out = np.zeros((len(jobs), DIM), np.float32)
    fails = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for k, sig in enumerate(pool.imap(process_one, jobs, chunksize=2)):
            if isinstance(sig, str):
                fails.append((ids[k], sig))
            else:
                out[k] = sig
            if (k + 1) % 200 == 0 or k + 1 == len(jobs):
                print("  %d/%d elapsed=%.0fs fails=%d" %
                      (k + 1, len(jobs), time.time() - t0, len(fails)),
                      flush=True)
    path = OUT_DIR / ("dzd_%s.npz" % args.split)
    np.savez_compressed(path, ids=ids.astype(str), X=out)
    print("saved", path, "fails=", len(fails), flush=True)
    for i, msg in fails[:10]:
        print("  FAIL", i, msg)
    if len(fails) > len(ids) * 0.01:
        sys.exit(2)


if __name__ == "__main__":
    main()
