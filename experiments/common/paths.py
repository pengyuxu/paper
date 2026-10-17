"""Single source of truth for dataset versions, paths, and CSV layouts.

Versions
--------
paper   : SHREC25 track version,  97 classes, 9244 train / 2311 test
updated : 2025-10-28 correction,  99 classes, 9244 train / 2308 test
          (class 16 split into 16/97/98; test proteins 840/1277/1311 removed)

All experiment scripts MUST import VERSIONS / paths from here rather than
hard-coding 97/2321 or CSV column names.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments"
RAW_DIR = REPO_ROOT / "data_raw"
WORK_ROOT = REPO_ROOT / "work"
RESULTS_DIR = EXP_DIR / "results"
NEWCODE_DIR = REPO_ROOT / "newcode"

# Shared point clouds (identical geometry for both versions; labels differ)
CLOUD_DIR = WORK_ROOT / "clouds" / "txt8"          # 11,553 *_pc.npy merged
RAW_TRAIN_VTK = RAW_DIR / "train_set_vtk"
RAW_TEST_VTK = RAW_DIR / "test_set_vtk"

N_POINTS = 8192
N_CHANNELS = 8
TRAIN_ROWS = 7428      # row-order split: first 7428 train, rest val
VAL_ROWS = 1816
SAMPLE_SEED = 42

# Deleted test proteins in the updated version (anonymised numeric ids)
UPDATED_DELETED_TEST_IDS = ("840", "1277", "1311")

VERSIONS = {
    "paper": {
        "num_classes": 97,
        "test_n": 2311,
        "dataset_dir": REPO_ROOT / "dataset_paper_version",
        # test list csv (single column)
        "test_list_file": "test_set.csv",
        "test_id_col": "anonymised_protein_id",
        "id_is_vtk_suffix": True,          # entries like "2286.vtk"
        # ground truth csv columns
        "gt_file": "test_set_ground_truth.csv",
        "gt_orig_col": "protein_id",
        "gt_anon_col": "anonymised_protein_id",
        "gt_label_col": "class_id",
    },
    "updated": {
        "num_classes": 99,
        "test_n": 2308,
        "dataset_dir": REPO_ROOT / "dataset_updated_281025",
        "test_list_file": "test_set.csv",
        "test_id_col": "anonymized_protein_id",
        "id_is_vtk_suffix": True,
        "gt_file": "test_set_ground_truth.csv",
        "gt_orig_col": "pdb_id",
        "gt_anon_col": "protein_id",
        "gt_label_col": "class_id",
    },
}


def version_cfg(version):
    if version not in VERSIONS:
        raise ValueError("unknown version %r; expected one of %s"
                         % (version, sorted(VERSIONS)))
    return VERSIONS[version]


def work_dir(version):
    return WORK_ROOT / version


def csv_dir(version):
    return work_dir(version) / "csv"


def new_data_dir(version):
    return work_dir(version) / "new_data"


def log_dir(version):
    return work_dir(version) / "log" / "classification_shrec2025"


def cache_dir(version):
    return work_dir(version) / "results_final" / "cache"


def result_table_dir(version):
    return RESULTS_DIR / version / "tables"


def ensure_dirs(version):
    for d in (csv_dir(version), new_data_dir(version), log_dir(version),
              cache_dir(version), result_table_dir(version)):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    for v, c in VERSIONS.items():
        print(v, "num_classes=%d test_n=%d test_id_col=%s"
                 % (c["num_classes"], c["test_n"], c["test_id_col"]))
        print("  gt cols:", c["gt_orig_col"], c["gt_anon_col"], c["gt_label_col"])
        print("  work:", work_dir(v))
