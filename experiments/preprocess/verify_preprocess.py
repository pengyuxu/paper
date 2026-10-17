"""Independent verification of regenerated point clouds (AC-1).

Checks:
  A. counts per version (train 9244; test list 2311/2308 after version filter)
  B. shape/dtype and channel ranges of every npy
  C. xyz finiteness / bounded (equal-volume scaling -> surface within ~few r)
  D. determinism: reprocess a sample of meshes into a temp dir, hash-equal
  E. independent reference recomputation on a small sample (brute-force NN,
     explicit VTK array indexing) vs the cached npy
  F. updated test list excludes 840/1277/1311

Usage:
  python -m experiments.preprocess.verify_preprocess --quick     # no full scan
  python -m experiments.preprocess.verify_preprocess
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from experiments.common import paths
from experiments.preprocess.build_pointclouds import process_one


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_clouds(ids_with_split, cloud_dir, max_radius):
    bad = []
    rows = []
    for name, split in ids_with_split:
        p = os.path.join(cloud_dir, name + "_pc.npy")
        if not os.path.isfile(p):
            bad.append((name, "missing"))
            continue
        x = np.load(p, mmap_mode="r")
        if x.shape != (paths.N_POINTS, paths.N_CHANNELS) or x.dtype != np.float32:
            bad.append((name, "shape/dtype %s %s" % (x.shape, x.dtype)))
            continue
        if not np.isfinite(x).all():
            bad.append((name, "nonfinite"))
            continue
        p6, p7 = float(x[:, 6].min()), float(x[:, 6].max())
        n6, n7 = float(x[:, 7].min()), float(x[:, 7].max())
        rmax = float(np.linalg.norm(x[:, :3], axis=1).max())
        if p6 < -1e-6 or p7 > 1 + 1e-6 or n6 < -1e-6 or n7 > 1 + 1e-6:
            bad.append((name, "potential range %f/%f %f/%f" % (p6, p7, n6, n7)))
        if rmax > max_radius:
            bad.append((name, "rmax %.3f" % rmax))
        rows.append((name, split, rmax))
    return bad, rows


def determinism_check(sample_jobs):
    tmp = tempfile.mkdtemp(prefix="pc_det_")
    try:
        mism = []
        for src, canonical, prof in sample_jobs:
            out = os.path.join(tmp, os.path.basename(canonical))
            r1 = process_one((src, out, prof))
            h1 = md5(out)
            os.remove(out)
            r2 = process_one((src, out, prof))
            h2 = md5(out)
            if r1[1].startswith("fail") or r2[1].startswith("fail") or h1 != h2:
                mism.append((os.path.basename(src), r1[1], r2[1],
                             h1[:8], h2[:8]))
        return mism
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def reference_recompute(vtk_path):
    """Explicit, dependency-heavy but independent reference: rebuild the cloud
    with plain loops over vtk array values and brute-force nearest neighbour.
    Geometry sampling still uses trimesh (single implementation available),
    but transforms/NN/normalisation are independently coded here."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    import trimesh
    import tempfile as _tf

    reader = vtk.vtkGenericDataObjectReader()
    reader.ReadAllScalarsOn()
    reader.SetFileName(vtk_path)
    reader.Update()
    poly = reader.GetOutput()
    fd, stl = _tf.mkstemp(suffix=".stl")
    os.close(fd)
    w = vtk.vtkSTLWriter()
    w.SetInputData(poly)
    w.SetFileName(stl)
    w.Write()
    mesh = trimesh.load(stl, force="mesh")
    os.remove(stl)
    c = np.array(mesh.center_mass, dtype=np.float64)
    V = abs(mesh.volume)
    if not np.isfinite(V) or V < 1e-12:
        V = abs(mesh.convex_hull.volume)
    s = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    mesh.apply_translation(-c)
    mesh.apply_scale(1.0 / s)
    pts, fidx = trimesh.sample.sample_surface(mesh, paths.N_POINTS, seed=42)
    pts = np.array(pts, dtype=np.float64)
    nrm = np.array(mesh.face_normals[fidx], dtype=np.float64)
    V0 = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    P = vtk_to_numpy(poly.GetPointData().GetArray("Potential")).astype(np.float64)
    NP = vtk_to_numpy(
        poly.GetPointData().GetArray("NormalPotential")).astype(np.float64)
    V0 = (V0 - c) / s
    idx = np.empty(len(pts), dtype=np.int64)
    for i, q in enumerate(pts):
        idx[i] = int(np.argmin(np.sum((V0 - q) ** 2, axis=1)))
    phi, psi = P[idx].copy(), NP[idx].copy()
    for a in (phi, psi):
        lo, hi = a.min(), a.max()
        if hi - lo > 1e-12:
            a -= lo
            a /= hi - lo
        else:
            a.fill(0.0)
    return np.concatenate([pts, nrm, phi[:, None], psi[:, None]], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_sample", type=int, default=8)
    args = ap.parse_args()

    results = []
    train_df = pd.read_csv(paths.version_cfg("paper")["dataset_dir"] / "train_set.csv")
    train_ids = [(n, "train") for n in train_df["protein_id"]]

    all_bad = []
    if not args.quick:
        print("scanning %d train clouds ..." % len(train_ids), flush=True)
        bad, _ = scan_clouds(train_ids, paths.CLOUD_DIR, max_radius=8.0)
        all_bad += bad
        print("  bad:", len(bad))

    for ver in ("paper", "updated"):
        cfg = paths.version_cfg(ver)
        tl = pd.read_csv(cfg["dataset_dir"] / cfg["test_list_file"])
        ids = [str(v).replace(".vtk", "") for v in tl[tl.columns[0]]]
        print("scanning %d test clouds (%s) ..." % (len(ids), ver), flush=True)
        bad, _ = scan_clouds([(i, "test") for i in ids], paths.CLOUD_DIR, 8.0)
        all_bad += bad
        if ver == "updated":
            for x in paths.UPDATED_DELETED_TEST_IDS:
                p = paths.CLOUD_DIR / (x + "_pc.npy")
                # npy may exist (shared superset) but version list must exclude it
                assert x not in ids, "deleted id %s present in updated list!" % x
                results.append(("deleted-exclusion-%s" % x, x not in ids))
        results.append(("count-%s" % ver, len(ids) == cfg["test_n"]))

    # determinism on mixed sample
    rng = np.random.RandomState(0)
    sample_train = rng.choice([n for n, _ in train_ids],
                              size=min(args.n_sample // 2, len(train_ids)),
                              replace=False)
    sample_jobs = []
    _pcfg = paths.version_cfg("paper")
    tl = pd.read_csv(_pcfg["dataset_dir"] / _pcfg["test_list_file"])
    test_names = [str(v).replace(".vtk", "") for v in tl[tl.columns[0]]]
    sample_test = rng.choice(test_names, size=args.n_sample - len(sample_train),
                             replace=False)
    for n in sample_train:
        sample_jobs.append((str(paths.RAW_TRAIN_VTK / (n + ".vtk")),
                            str(paths.CLOUD_DIR / (n + "_pc.npy")), "paper"))
    for n in sample_test:
        sample_jobs.append((str(paths.RAW_TEST_VTK / (n + ".vtk")),
                            str(paths.CLOUD_DIR / (n + "_pc.npy")), "paper"))
    mism = determinism_check(sample_jobs)
    results.append(("determinism (%d files)" % len(sample_jobs), len(mism) == 0))
    if mism:
        for m in mism:
            print("  DET MISMATCH", m)

    # independent reference comparison
    ref_ok, ref_maxdiff = True, 0.0
    for n in list(sample_test[:4]) + list(sample_train[:1]):
        src = (paths.RAW_TEST_VTK if n in set(test_names) else paths.RAW_TRAIN_VTK) / (n + ".vtk")
        if not os.path.isfile(src):
            continue
        ref = reference_recompute(str(src))
        cache = np.load(paths.CLOUD_DIR / (n + "_pc.npy"))
        if cache.shape != ref.shape:
            ref_ok = False
            print("  REF shape mismatch", n)
            continue
        d = float(np.abs(ref.astype(np.float64) - cache.astype(np.float64)).max())
        ref_maxdiff = max(ref_maxdiff, d)
        if d > 1e-5:
            ref_ok = False
            print("  REF maxdiff %.3e %s" % (d, n))
    results.append(("reference max|d|<=1e-5 (%.2e)" % ref_maxdiff, ref_ok))

    results.append(("scan-no-bad", len(all_bad) == 0))
    print("\n=== VERIFY SUMMARY ===")
    for name, ok in results:
        print("[%s] %s" % ("PASS" if ok else "FAIL", name))
    if all_bad:
        print("bad clouds (first 20):")
        for b in all_bad[:20]:
            print("  ", b)
    sys.exit(0 if all(ok for _, ok in results) and not all_bad else 1)


if __name__ == "__main__":
    main()
