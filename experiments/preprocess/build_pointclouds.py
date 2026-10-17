"""VTK surface mesh -> 8192-point, 8-channel point cloud (paper §3.2).

Per mesh:
  1. vtkGenericDataObjectReader reads the legacy VTK polydata (points, faces,
     Potential and NormalPotential vertex arrays).
  2. Geometry exported to STL and loaded by trimesh.
  3. Translated to centre of mass, scaled by the equal-volume-sphere radius
     r = (3|V|/4pi)^(1/3)  (profile 'paper'; profile 'barisin' reproduces the
     released bounding-sphere scaling instead, see spec open question Q1).
  4. 8192 points by area-weighted face sampling, seed = 42 (deterministic);
     per-point normal = host face normal.
  5. VTK vertex coordinates are transformed by the SAME translation+scale and
     Potential/NormalPotential are transferred to sampled points by cKDTree k=1
     nearest neighbour (correct aligned frame; the released script omitted the
     centring of VTK vertices), then min-max normalised per mesh.

Output: <name>_pc.npy, float32, shape (8192, 8): xyz | nx ny nz | phi psi.

Usage:
  python -m experiments.preprocess.build_pointclouds --split train --workers 40
  python -m experiments.preprocess.build_pointclouds --split test  --workers 40
  python -m experiments.preprocess.build_pointclouds --split train --limit 8
"""
import argparse
import os
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from experiments.common import paths

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import trimesh
from scipy.spatial import cKDTree


def _read_polydata(vtk_path):
    reader = vtk.vtkGenericDataObjectReader()
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.SetFileName(vtk_path)
    reader.Update()
    return reader.GetOutput()


def _polydata_to_stl(polydata, stl_path):
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(stl_path)
    ok = writer.Write()
    if not ok:
        raise RuntimeError("vtk STL writer failed")


def process_one(args):
    vtk_path, out_path, profile = args
    t0 = time.time()
    if os.path.isfile(out_path):
        return (vtk_path, "skip", time.time() - t0, None)
    tmp_stl = None
    try:
        poly = _read_polydata(vtk_path)
        verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        pot = vtk_to_numpy(poly.GetPointData().GetArray("Potential")).astype(np.float64)
        npot = vtk_to_numpy(
            poly.GetPointData().GetArray("NormalPotential")).astype(np.float64)

        fd, tmp_stl = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        _polydata_to_stl(poly, tmp_stl)
        mesh = trimesh.load(tmp_stl, force="mesh")

        center = np.asarray(mesh.center_mass, dtype=np.float64)
        mesh.apply_translation(-center)
        if profile == "paper":
            vol = abs(float(mesh.volume)) if mesh.volume is not None else 0.0
            if not np.isfinite(vol) or vol < 1e-12:
                vol = abs(float(mesh.convex_hull.volume))
            scale = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        elif profile == "barisin":
            scale = (0.75 * float(mesh.bounding_sphere.volume)) ** (1.0 / 3.0)
        else:
            raise ValueError(profile)
        mesh.apply_scale(1.0 / scale)

        samples_xyz, face_idx = trimesh.sample.sample_surface(
            mesh, paths.N_POINTS, seed=paths.SAMPLE_SEED)
        samples_xyz = np.asarray(samples_xyz, dtype=np.float64)
        normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)

        # transform VTK vertices, then nearest-neighbour potential transfer.
        # paper profile: same translation+scale as the samples (aligned frame).
        # barisin profile: EXACT replica of their released script
        # (seq = verts/scaling_factor, no centring) -- their weights were
        # produced/tested with this frame mismatch, so keep it for sanity.
        vt = verts / scale if profile == "barisin" else (verts - center) / scale
        tree = cKDTree(vt)
        _, nn = tree.query(samples_xyz, k=1)
        phi = pot[nn]
        psi = npot[nn]
        for a in (phi, psi):
            lo, hi = float(a.min()), float(a.max())
            if hi - lo > 1e-12:
                a -= lo
                a /= (hi - lo)
            else:
                a.fill(0.0)

        pc = np.concatenate(
            [samples_xyz, normals, phi[:, None], psi[:, None]], axis=1)
        np.save(out_path, pc.astype(np.float32))
        return (vtk_path, "ok", time.time() - t0,
                dict(n_verts=int(len(verts)), watertight=bool(mesh.is_watertight),
                     volume=float(mesh.volume), scale=float(scale)))
    except Exception as e:  # noqa: BLE001 - surfaced as a failed-row record
        return (vtk_path, "fail:%s" % type(e).__name__, time.time() - t0, str(e))
    finally:
        if tmp_stl and os.path.exists(tmp_stl):
            try:
                os.remove(tmp_stl)
            except OSError:
                pass


def cloud_dir_for(profile):
    return paths.WORK_ROOT / "clouds" / (
        "txt8" if profile == "paper" else "txt8_barisin")


def job_list(split, profile):
    out_dir = cloud_dir_for(profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    if split == "train":
        for ver in ("paper",):  # identical train id list in both versions
            df = pd.read_csv(paths.version_cfg(ver)["dataset_dir"] / "train_set.csv")
            for name in df["protein_id"]:
                jobs.append((str(paths.RAW_TRAIN_VTK / (name + ".vtk")),
                             str(out_dir / (name + "_pc.npy")), profile))
    elif split == "test":
        # union of version test lists (paper is superset: 2311 incl. 3 deleted)
        names = set()
        for ver in ("paper", "updated"):
            cfg = paths.version_cfg(ver)
            df = pd.read_csv(cfg["dataset_dir"] / cfg["test_list_file"])
            col = df.columns[0]
            for v in df[col]:
                names.add(str(v).replace(".vtk", ""))
        for name in sorted(names, key=int):
            jobs.append((str(paths.RAW_TEST_VTK / (name + ".vtk")),
                         str(out_dir / (name + "_pc.npy")), profile))
    else:
        raise ValueError(split)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--profile", choices=["paper", "barisin"], default="paper")
    args = ap.parse_args()

    jobs = job_list(args.split, args.profile)
    if args.limit:
        jobs = jobs[:args.limit]
    existing = sum(os.path.isfile(j[1]) for j in jobs)
    print("split=%s profile=%s total=%d existing=%d workers=%d"
          % (args.split, args.profile, len(jobs), existing, args.workers),
          flush=True)
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                done = i + 1
                el = time.time() - t0
                print("  %d/%d  elapsed=%.0fs  (%.2fs/file)"
                      % (done, len(jobs), el, el / max(done, 1)), flush=True)

    n_ok = sum(r[1] in ("ok", "skip") for r in rows)
    n_fail = [r for r in rows if r[1].startswith("fail")]
    print("DONE split=%s ok=%d fail=%d wall=%.0fs"
          % (args.split, n_ok, len(n_fail), time.time() - t0), flush=True)
    for r in n_fail[:20]:
        print("  FAIL", r[0], r[1], r[3])
    meta = pd.DataFrame(
        [dict(vtk=os.path.basename(r[0]), status=r[1], sec=r[2],
              **(r[3] or {})) for r in rows])
    meta_path = paths.EXP_DIR / "results" / ("preprocess_%s_meta.csv" % args.split)
    meta.to_csv(meta_path, index=False)
    print("meta ->", meta_path)
    if n_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
