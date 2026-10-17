"""C3 Guerra topological descriptors (paper §4.6), built with the author's
published primitives (marcoguerra192/ProteinClassifier2025) and gudhi 3.10.1.

RUN WITH PYTHON 3.10 (the Guerra repo requires >=3.10) and the workspace-local
gudhi wheel folder on PYTHONPATH:

  PYTHONPATH=methods/guerra/pylibs \
  /home/t640admin/anaconda3/envs/car/bin/python \
      -m experiments.descriptors.guerra_build --split train --workers 24

Per surface (179 raw coordinates; 8 always-zero H2 columns are dropped later
by the cleaning script):
  * 104 spherical-sector block: 8 octants x 13 fields (3 radial-distance
    quantiles, 3 cumulative signed-potential values, H0/H1/H2 bar counts,
    longest H0 birth/death and its two generating-vertex potentials);
  * 75 alpha block: 5x5 persistence images of the 150 most prominent H0/H1/H2
    alpha-complex points.

The implementation shells into Guerra's own compute_spherical_sectors_descs /
process_spherical_sectors_descriptors / AlphaDiag / PersImagesVectorize on a
per-protein temp layout, guaranteeing identical numbers to the released code.

Outputs: experiments/results/desc/guerra_{train,test}.npz {ids, X(N,179)}.
"""
from __future__ import annotations
import argparse, os, sys, tempfile, time
import multiprocessing as mp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GUERRA = os.path.join(REPO, "methods", "guerra", "repo")
sys.path.insert(0, GUERRA)
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd

from src.data_reader import DataSource, convertVTK_to_numpy
from src.descriptors.dscs_driver import (
    compute_descriptors, compute_spherical_sectors_descs,
    process_spherical_sectors_descriptors)
from experiments.common import paths as vp

OUT_DIR = vp.EXP_DIR / "results" / "desc"


def _per_file(job):
    vtk_path, = job
    pid = os.path.splitext(os.path.basename(vtk_path))[0]
    d = tempfile.mkdtemp(prefix="guerra_")
    vdir = os.path.join(d, "vtk"); ndir = os.path.join(d, "np")
    rdir = os.path.join(d, "raw")
    os.makedirs(vdir); os.makedirs(ndir); os.makedirs(rdir)
    try:
        points, triangles, potentials, norm_pot = convertVTK_to_numpy(vtk_path)
        np.savez(os.path.join(ndir, pid + ".vtk.npz"),
                 points=points, triangles=triangles,
                 potentials=potentials, norm_potentials=norm_pot)
        os.symlink(vtk_path, os.path.join(vdir, pid + ".vtk"))

        alpha, _ = compute_descriptors(
            vdir, model="AlphaProminent", data=vdir,
            Num_Prominent=150, PersImPoints=5,
            which_quantiles=[0.25, 0.5, 0.75])
        alpha = alpha[0]

        sec_path = os.path.join(d, "sec.npy")
        compute_spherical_sectors_descs(vdir, ndir + "/", rdir + "/")
        process_spherical_sectors_descriptors(
            vdir, rdir + "/", ndir + "/", sec_path, 3)
        sec = np.load(sec_path)[0]
        return np.concatenate([sec, alpha]).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        return ("FAIL:%s:%s" % (type(e).__name__, e))
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


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
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, raw = ids_for(args.split)
    if args.limit:
        ids = ids[:args.limit]
    jobs = [(str(raw / (i + ".vtk")),) for i in ids]
    print("GUERRA split=%s n=%d workers=%d" %
          (args.split, len(jobs), args.workers), flush=True)
    out = np.zeros((len(jobs), 179), np.float32)
    fails = []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for k, sig in enumerate(pool.imap(_per_file, jobs, chunksize=1)):
            if isinstance(sig, str):
                fails.append((ids[k], sig))
            else:
                out[k] = sig
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                print("  %d/%d elapsed=%.0fs fails=%d" %
                      (k + 1, len(jobs), time.time() - t0, len(fails)),
                      flush=True)
    path = OUT_DIR / ("guerra_%s.npz" % args.split)
    np.savez_compressed(path, ids=ids.astype(str), X=out)
    print("saved", path, "fails=", len(fails), flush=True)
    for i, msg in fails[:10]:
        print("  FAIL", i, msg)
    if len(fails) > len(ids) * 0.01:
        sys.exit(2)


if __name__ == "__main__":
    main()
