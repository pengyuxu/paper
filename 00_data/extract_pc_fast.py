#!/usr/bin/env python
"""Fast extraction of point clouds from VTK protein surfaces.

Faithful reproduction of extract_pc_from_vtk/convert_extract_features.py,
with the per-point nearest-neighbour search replaced by scipy cKDTree
(identical results, orders of magnitude faster).

Output: {outdir}/{basename}_pc.npy  -- float array (8192, 8):
        [x, y, z, nx, ny, nz, potential, normal_potential]
        (xyz/normals on the scaled mesh, potentials min-max normalised)
"""
import os
import sys
import argparse
import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree


def sample_pc_from_mesh(mesh, n_points, sample_normals=True):
    mesh.apply_translation(-mesh.center_mass)
    scaling_factor = pow(3 / 4 * mesh.bounding_sphere.volume, 1 / 3)
    mesh.apply_scale(1 / scaling_factor)

    samples, normals = trimesh.sample.sample_surface(mesh, n_points, seed=42)
    samples = np.array(samples)
    if sample_normals:
        normals = np.array(mesh.face_normals[normals])
        samples = np.concatenate([samples, normals], axis=1)
    return samples, scaling_factor


def convert_file(filepath, n_points):
    reader = vtk.vtkGenericDataObjectReader()
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.SetFileName(filepath)
    reader.Update()
    out = reader.GetOutput()

    # write to a temp STL like the original script (trimesh loads it back);
    # unique per process so multiple workers can run in parallel
    tmp_stl = f"/tmp/_shrec_tmp2_{os.getpid()}.stl"
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(reader.GetOutputPort())
    writer.SetFileName(tmp_stl)
    if writer.Write() != 1:
        return None

    mesh = trimesh.load(tmp_stl)
    pc, scaling_factor = sample_pc_from_mesh(mesh, n_points)

    seq = vtk_to_numpy(out.GetPoints().GetData()) / scaling_factor
    pot = vtk_to_numpy(out.GetPointData().GetArray("Potential")).astype(np.float64)
    npot = vtk_to_numpy(out.GetPointData().GetArray("NormalPotential")).astype(np.float64)
    min_val, max_val = pot.min(), pot.max()
    min_val2, max_val2 = npot.min(), npot.max()

    tree = cKDTree(seq)
    idx = tree.query(pc[:, :3], k=1)[1]
    potential = (pot[idx] - min_val) / (max_val - min_val)
    potential_normal = (npot[idx] - min_val2) / (max_val2 - min_val2)

    new_pc = np.concatenate([pc, potential[:, None], potential_normal[:, None]], axis=1)
    return new_pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--mapfile", default=None,
                    help="csv with header 'outname,vtkname' mapping output npy basename to vtk basename (no ext)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.mapfile:
        import csv as _csv
        pairs = []
        with open(args.mapfile) as f:
            r = _csv.reader(f)
            header = next(r)
            for row in r:
                if len(row) >= 2:
                    pairs.append((row[0].strip(), row[1].strip()))
        names = pairs  # (output basename, vtk basename)
    else:
        names = [(os.path.splitext(f)[0], os.path.splitext(f)[0])
                 for f in os.listdir(args.indir) if f.endswith(".vtk")]

    ok, skip, fail = 0, 0, 0
    for i, (outname, vtkname) in enumerate(names):
        outpath = os.path.join(args.outdir, outname + "_pc.npy")
        if os.path.isfile(outpath):
            skip += 1
            continue
        try:
            pc = convert_file(os.path.join(args.indir, vtkname + ".vtk"), args.n_points)
            if pc is None:
                fail += 1
                continue
            np.save(outpath, pc.astype(np.float32))
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAIL {vtkname}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(names)}] ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"DONE ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
