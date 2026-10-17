"""Materialise per-version work directories from the canonical dataset CSVs.

Layout produced (per version v in {paper, updated}):
  work/v/csv/train_set_2.csv          <- dataset_*/train_set.csv
  work/v/csv/test_set_ground_truth.csv<- dataset_*/test_set_ground_truth.csv
  work/v/new_data/test_set_2.csv      <- dataset_*/test_set.csv
  work/v/new_data/txt8 -> ../../clouds/txt8   (shared, geometry is identical)

Original dataset files are never modified.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import paths


def setup(version):
    cfg = paths.version_cfg(version)
    paths.ensure_dirs(version)
    src = cfg["dataset_dir"]

    pairs = [
        (src / "train_set.csv", paths.csv_dir(version) / "train_set_2.csv"),
        (src / cfg["gt_file"], paths.csv_dir(version) / "test_set_ground_truth.csv"),
        (src / cfg["test_list_file"], paths.new_data_dir(version) / "test_set_2.csv"),
    ]
    for s, d in pairs:
        shutil.copy2(s, d)
        print("copied %s -> %s" % (s.name, d.relative_to(paths.REPO_ROOT)))

    # shared cloud dir + symlink
    paths.CLOUD_DIR.mkdir(parents=True, exist_ok=True)
    link = paths.new_data_dir(version) / "txt8"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        elif link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            print("kept existing %s" % link)
    if not link.exists():
        link.symlink_to(paths.CLOUD_DIR.resolve())
        print("symlink %s -> %s" %
              (link.relative_to(paths.REPO_ROOT),
               paths.CLOUD_DIR.resolve().relative_to(paths.REPO_ROOT)))


if __name__ == "__main__":
    versions = sys.argv[1:] or list(paths.VERSIONS)
    for v in versions:
        print("=== setup", v)
        setup(v)
