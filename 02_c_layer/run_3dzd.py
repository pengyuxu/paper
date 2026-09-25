"""C1: batch 3DZD descriptor computation, faithful to kiharalab/SHREC2025.

Per VTK: 3 PLY (shape / positive-potential faces / negative-potential faces)
-> OBJ -> bin/obj2grid -g 64 -> bin/map2zernike -c 0.5 -> 121 invariants each.
Only difference vs official vtk2zd.py: strip vertex normals from shape OBJ
(trimesh>=4.7 exports 'f a//na' which the provided obj2grid binary cannot parse;
trimesh 4.6.8 pinned by the repo did not emit vn).
Outputs per key: <out>/<key>.npz with inv(363,) and volume (shape mesh volume).
"""
import os, re, sys, csv, time, argparse, subprocess, tempfile, shutil
from pathlib import Path
from multiprocessing import Pool
os.environ.setdefault("OMP_NUM_THREADS", "1")

BASE = Path(__file__).resolve().parent.parent
BINDIR = BASE / "external/SHREC2025/bin"
VTKDIR = BASE.parent / "data"          # .../shrec2025/data with 11565 vtk files


def one(job):
    key, vtk_path, outdir = job
    outdir = Path(outdir)
    done = outdir / (key + ".npz")
    if done.exists():
        return key, "skip", 0.0
    t0 = time.time()
    tmp = tempfile.mkdtemp(dir=str(outdir), prefix=".t_%s_" % key[:8])
    try:
        st = _one_impl(key, vtk_path, outdir, tmp)
        return key, st, time.time() - t0
    except Exception as e:
        return key, "FAIL:" + repr(e)[:160], time.time() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _one_impl(key, vtk_path, outdir, tmpdir):
    import vtk
    import trimesh
    import numpy as np
    done = Path(outdir) / (key + ".npz")

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    polydata = reader.GetOutput()

    # ---- three PLY surfaces (logic identical to official vtk2zd.py) ----
    def write_ply(pd, path):
        w = vtk.vtkPLYWriter()
        w.SetFileName(path); w.SetInputData(pd); w.SetFileTypeToASCII(); w.Write()

    base = os.path.join(tmpdir, key)
    write_ply(polydata, base + "_shape.ply")

    points = polydata.GetPoints(); polys = polydata.GetPolys()
    pot = polydata.GetPointData().GetArray("Potential")
    if points is None or polys is None or pot is None or points.GetNumberOfPoints() == 0:
        raise RuntimeError("malformed vtk (points/polys/potential missing)")
    n = points.GetNumberOfPoints()
    vals = np.zeros(n)
    for i in range(n):
        vals[i] = pot.GetTuple1(i)
    pos_polys = vtk.vtkCellArray(); neg_polys = vtk.vtkCellArray()
    polys.InitTraversal(); idl = vtk.vtkIdList()
    while polys.GetNextCell(idl):
        k = idl.GetNumberOfIds()
        if k < 3:
            continue
        ids = [idl.GetId(i) for i in range(k)]
        vp = vals[ids]
        cell = vtk.vtkIdList()
        for i in ids:
            cell.InsertNextId(i)
        if np.all(vp > 0):
            pos_polys.InsertNextCell(cell)
        elif np.all(vp < 0):
            neg_polys.InsertNextCell(cell)

    def write_sub(polys_, name):
        pd2 = vtk.vtkPolyData(); pd2.SetPoints(points); pd2.SetPolys(polys_)
        cl = vtk.vtkCleanPolyData(); cl.PointMergingOff(); cl.SetInputData(pd2); cl.Update()
        o = cl.GetOutput()
        if o.GetNumberOfPoints() > 0 and o.GetNumberOfPolys() > 0:
            write_ply(o, base + "_" + name + ".ply"); return True
        return False

    has_pos = write_sub(pos_polys, "pos")
    has_neg = write_sub(neg_polys, "neg")

    # ---- OBJ export; shape needs vn stripped for obj2grid parser ----
    vn_re = re.compile(rb"^vn ")
    face_re = re.compile(rb"^f (\d+)//\d+ (\d+)//\d+ (\d+)//\d+")

    def export_sub(name):
        ply = f"{base}_{name}.ply"
        if not Path(ply).exists():
            return None
        trimesh.load(ply, process=False).export(f"{base}_{name}.obj", file_type="obj")
        return f"{base}_{name}.obj"

    shape_obj = base + "_shape.obj"
    shape_mesh = trimesh.load(base + "_shape.ply", process=False)
    vol = float(shape_mesh.volume)
    shape_mesh.export(shape_obj, file_type="obj")
    raw = Path(shape_obj).read_bytes().splitlines()
    out = []
    for ln in raw:
        if vn_re.match(ln):
            continue
        m = face_re.match(ln)
        if m:
            ln = b"f " + b" ".join(m.groups())
        out.append(ln)
    Path(shape_obj).write_bytes(b"\n".join(out) + b"\n")

    pos_obj = export_sub("pos")
    neg_obj = export_sub("neg")

    def to_inv(obj):
        r = subprocess.run([str(BINDIR / "obj2grid"), "-g", "64", obj],
                           capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        if r.returncode != 0:
            raise RuntimeError("obj2grid failed %s: %s" % (obj, r.stderr.decode()[:200]))
        grid = obj + ".grid"
        r = subprocess.run([str(BINDIR / "map2zernike"), "-c", "0.5", grid],
                           capture_output=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        if r.returncode != 0:
            raise RuntimeError("map2zernike failed %s" % grid)
        with open(grid + ".inv") as f:
            f.readline()
            return np.array([float(x) for x in f.readlines()], dtype=np.float32)

    inv_shape = to_inv(shape_obj)
    inv_pos = to_inv(pos_obj) if pos_obj else np.zeros(121, np.float32)
    inv_neg = to_inv(neg_obj) if neg_obj else np.zeros(121, np.float32)

    np.savez_compressed(done, inv=np.concatenate([inv_shape, inv_pos, inv_neg]),
                        volume=np.float64(vol))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "results_final/3dzd"))
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    jobs = []
    # official train_set.csv gives vtk keys with ':' (e.g. 8ugd_8:R:3U_model1);
    # our ids replace ':' with '_'. Use OUR underscore ids as output keys.
    off = pd.read_csv(VTKDIR.parent / "train_set.csv")
    off.columns = ["vtk_key", "class_id"]
    for _, r in off.iterrows():
        vtk_key = str(r.vtk_key).replace(".vtk", "")
        our_key = vtk_key.replace(":", "_")
        jobs.append((our_key, str(VTKDIR / (vtk_key + ".vtk")), str(outdir)))
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    for pid in te.anonymised_protein_id:
        pid = str(pid).replace(".vtk", "")   # csv stores '0.vtk'; output key is '0'
        jobs.append((pid, str(VTKDIR / (pid + ".vtk")), str(outdir)))

    missing = [j for j in jobs if not Path(j[1]).exists()]
    print("jobs=%d missing-vtk=%d" % (len(jobs), len(missing)))
    if missing[:3]:
        print(missing[:3])
    if args.limit:
        jobs = jobs[:args.limit]

    failures = []
    with Pool(args.workers) as pool:
        for i, (key, st, dt) in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            if st.startswith("FAIL"):
                failures.append((key, st))
            if i % 500 == 0 or st not in ("ok", "skip"):
                print("[%d/%d] %s %s %.1fs" % (i + 1, len(jobs), key, st[:80], dt), flush=True)
    (Path(args.out) / "failures.txt").write_text(
        "\n".join("%s\t%s" % (k, s) for k, s in failures))
    print("done failures=%d" % len(failures))


if __name__ == "__main__":
    main()
