# Data download instructions

## 1. VTK raw data (SHREC official)

Download the original VTK protein surface meshes from the SHREC 2025 official site:

- Train set: https://shrec2025.drugdesign.fr/files/train_set.tar.xz
- Test set: https://shrec2025.drugdesign.fr/files/test_set.tar.xz

After extraction you should have:
- `train_set/` containing `<protein_id>.vtk` files
- `test_set/` containing `<protein_id>.vtk` files

## 2. Preprocessed point clouds (new_data/)

The 8192-point / 8-channel point clouds used for training/eval are **already
committed** to this package under `new_data/` (2321 test files, ~1.2 GB).
If you need the training point clouds (9244 files) or want to re-download:

- https://drive.contact.de/s/TuXORXxgrKPVPmM

Place additional files under `replication/new_data/`, matching the
`BASE / "new_data"` path used by `01_a_layer/dump_mcfps.py`.

Each file `{basename}_pc.npy` is a float array of shape `(8192, 8)`:
`[x, y, z, nx, ny, nz, potential, normal_potential]`.

## 3. Build point clouds yourself (optional)

If you want to regenerate the point clouds from the VTK meshes instead of
downloading them, use:

```bash
python 00_data/extract_pc_fast.py --vtk_dir <path_to_train_set> --out_dir new_data/txt8
python 00_data/extract_pc_fast.py --vtk_dir <path_to_test_set>  --out_dir new_data/txt8
```

`extract_pc_fast.py` extracts 8192 points with 8 channels per mesh (scipy
`cKDTree` replaces the slow per-point NN search; results are identical to the
reference `convert_extract_features.py`).

## 4. Splits CSVs

The split CSVs are already committed under `00_data/splits/`:
- `train_set_2.csv` — 9244 rows; columns `protein_id, class_id`
  (first 7428 = train, last 1816 = val)
- `test_set_2.csv` — 2321 rows; column `anonymised_protein_id`
- `test_set_ground_truth.csv` — 2321 rows; columns `anonymised_protein_id, class_id`
