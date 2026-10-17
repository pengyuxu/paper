"""Repro: full data load FIRST, then CUDA init (train_v5 ordering)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NEWCODE = os.path.join(REPO, "newcode")
sys.path.append(os.path.join(NEWCODE, "models")); sys.path.append(NEWCODE); sys.path.append(REPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import pandas as pd
import torch

print("t0 avail", torch.cuda.is_available(), flush=True)
from data_utils.ProteinToPointCloudProcessor import DataProcessor as DP
from experiments.common import paths as vp

v = sys.argv[1] if len(sys.argv) > 1 else "paper"
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 0
csv = pd.read_csv(vp.csv_dir(v) / "train_set_2.csv")
files = csv.protein_id.values if not nmax else csv.protein_id.values[:nmax]
labs = csv.class_id.values if not nmax else csv.class_id.values[:nmax]
t = time.time()
ds = DP(str(vp.new_data_dir(v)) + "/", files, labs, pc_folder="/txt8/")
print("loaded", len(ds), "clouds in %.1fs" % (time.time() - t), flush=True)

import resource
print("open fds:", len(os.listdir("/proc/self/fd")),
      "mem GB: %.2f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), flush=True)
print("post-load avail", torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
try:
    x = torch.zeros(8, device="cuda:0")
    print("LATE_CUDA_OK", x.device, flush=True)
except Exception as e:
    print("LATE_CUDA_FAIL", str(e)[:200], flush=True)
