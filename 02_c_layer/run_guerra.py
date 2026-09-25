"""C3: Guerra et al. topological descriptors (Guerra_v1), faithful to
github.com/marcoguerra192/ProteinClassifier2025 (SHREC 2025 paper, sec 3.5).

Per VTK we emit:
  alpha  (75,)   3 x 5x5 persistence images of the alpha-complex diagrams
                 (dim 0,1,2; top-150 prominent points), identical to their
                 AlphaProminent pipeline.
  sector (104,)  8 octant sectors x 13 raw features (3 radial-distance
                 quantiles, 3 cumulative radial potentials, counts of
                 H0/H1/H2 bars longer than median-radius/10, birth/death of
                 the longest finite H0 bar, potentials at its generators).
                 Their cleaning later drops the always-zero H2 column in
                 each block (->96), see eval_guerra.py.

Pure CPU (gudhi/CGAL). Multiprocessing, resumable, failures recorded.
"""
import os, sys, time, argparse
from pathlib import Path
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
BASE = Path(__file__).resolve().parent.parent
GUERRA = BASE / "external/guerra"
VTKDIR = BASE.parent / "data"
sys.path.insert(0, str(GUERRA))


def one(job):
    key, vtk_path, outdir = job
    outdir = Path(outdir)
    done = outdir / (key + ".npz")
    if done.exists():
        return key, "skip", 0.0
    t0 = time.time()
    try:
        import numpy as np
        from src.data_reader import convertVTK_to_numpy
        from src.descriptors.alpha_prominent import AlphaDiag, PersImagesVectorize
        from src.descriptors.distance_dist import centroid, distances_from_point, quantiles_of_distance
        from src.descriptors.spherical_sectors import common_sector, which_sector
        from gudhi import SimplexTree

        points, tris, pot, _npot = convertVTK_to_numpy(vtk_path)
        pot = pot.reshape(-1)
        if points.shape[0] < 4 or tris.shape[0] < 1:
            raise RuntimeError("empty mesh")

        # ---- alpha complex persistence images (their exact calls) ----
        d0, d1, d2 = AlphaDiag(points.tolist(), 150)
        alpha = PersImagesVectorize(d0, d1, d2, 5).astype(np.float32)

        # ---- 8 octant sectors, lower-star filtration per sector ----
        cen = centroid(points)
        dists = distances_from_point(points, cen)
        SCs = [SimplexTree() for _ in range(8)]
        SecPoints = [[] for _ in range(8)]
        for tri in tris:
            a, b, c3 = int(tri[0]), int(tri[1]), int(tri[2])
            try:
                sec = common_sector([points[a], points[b], points[c3]], cen) - 1
            except ValueError:
                continue
            t = [a, b, c3]
            SCs[sec].insert(t)
            SCs[sec].assign_filtration(t, filtration=max(dists[a], dists[b], dists[c3]))
            for e in ([a, b], [a, c3], [b, c3]):
                SCs[sec].insert(e)
                SCs[sec].assign_filtration(e, filtration=max(dists[e[0]], dists[e[1]]))
        for p in range(points.shape[0]):
            sec = which_sector(points[p], cen) - 1
            SecPoints[sec].append(p)
            SCs[sec].insert([p])
            SCs[sec].assign_filtration([p], filtration=dists[p])

        qs = [0.25, 0.5, 0.75]
        row = np.zeros(0)
        for i, st in enumerate(SCs):
            st.compute_persistence()
            st.set_dimension(3)
            dgm0 = st.persistence_intervals_in_dimension(0)
            dgm1 = st.persistence_intervals_in_dimension(1)
            dgm2 = st.persistence_intervals_in_dimension(2)
            gens = st.lower_star_persistence_generators()
            tp = np.array(SecPoints[i], dtype=int)
            if tp.size:
                q = quantiles_of_distance(points[tp], cen, qs)
                ordering = np.argsort(dists[tp])
                rc = np.cumsum(pot[tp[ordering]])
                idx = (np.array(qs) * tp.size).astype(int)
                cc = rc[idx]
            else:
                q, cc = [0.0, 0.0, 0.0], np.zeros(3)

            def cnt(d, finite_death):
                if d is None or len(d) == 0:
                    return 0
                pers = d[:, 1] - d[:, 0]
                m = pers >= q[1] / 10.0
                if finite_death:
                    m &= pers < np.inf
                return int(m.sum())

            c0, c1, c2 = cnt(dgm0, True), cnt(dgm1, True), cnt(dgm2, False)
            longest = np.zeros(2)
            gp = np.zeros(2)
            try:
                if dgm0 is not None and len(dgm0):
                    pers0 = dgm0[:, 1] - dgm0[:, 0]
                    fin = pers0 < np.inf
                    if fin.any():
                        li = int(np.argmax(np.where(fin, pers0, -np.inf)))
                        longest = dgm0[li]
                        gp = pot[np.asarray(gens[0][0][li], dtype=int)]
            except Exception:
                pass
            row = np.concatenate([row, np.asarray(q, float), np.asarray(cc, float),
                                  [c0, c1, c2], longest, gp])

        sector = row.astype(np.float32)
        if sector.shape[0] != 104:
            raise RuntimeError("bad sector dim %d" % sector.shape[0])
        np.savez_compressed(done, alpha=alpha, sector=sector)
        return key, "ok", time.time() - t0
    except Exception as e:
        return key, "FAIL:" + repr(e)[:160], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "results_final/guerra"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    jobs = []
    # train: our split order (7428 fit + 1816 val), keys with underscore.
    # VTK filenames carry ':' -> map via the official train_set.csv.
    off = pd.read_csv(BASE.parent / "train_set.csv")
    off.columns = ["vtk_key", "class_id"]
    key2vtk = {str(r.vtk_key).replace(".vtk", "").replace(":", "_"):
               str(r.vtk_key).replace(".vtk", "") for r in off.itertuples(index=False)}
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    for k in tr.protein_id.astype(str):
        vtk_name = key2vtk[k]
        jobs.append((k, str(VTKDIR / (vtk_name + ".vtk")), str(outdir)))
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    for pid in te.anonymised_protein_id.astype(str):
        pid = pid.replace(".vtk", "")
        jobs.append((pid, str(VTKDIR / (pid + ".vtk")), str(outdir)))

    missing = [j for j in jobs if not Path(j[1]).exists()]
    print("jobs=%d missing-vtk=%d" % (len(jobs), len(missing)), flush=True)
    if missing[:3]:
        print(missing[:3], flush=True)
    if args.limit:
        jobs = jobs[:args.limit]

    failures = []
    with Pool(args.workers) as pool:
        for i, (key, st, dt) in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            if st.startswith("FAIL"):
                failures.append((key, st))
            if i % 200 == 0 or st not in ("ok", "skip"):
                print("[%d/%d] %s %s %.1fs" % (i + 1, len(jobs), key, st[:80], dt),
                      flush=True)
    (outdir / "failures.txt").write_text("\n".join("%s\t%s" % (k, s) for k, s in failures))
    print("done failures=%d" % len(failures), flush=True)


if __name__ == "__main__":
    main()
