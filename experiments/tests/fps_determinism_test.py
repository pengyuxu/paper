"""Bitwise determinism test for the FPS monkey-patch (TR Task 5).

patched 'first' mode: two eval forward passes must be BITWISE identical.
patched 'seeded' mode: same pass index reproduces; different pass differs.
"""
import os, sys, importlib
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NEWCODE = os.path.join(REPO, "newcode")
sys.path += [os.path.join(NEWCODE, "models"), NEWCODE, REPO]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import pandas as pd
import torch
from data_utils.ProteinToPointCloudProcessor import DataProcessor as DP
from experiments.common import paths as vp
from experiments.common.deterministic_fps import FPSPatcher

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
mod = importlib.import_module("riconv2_cls_v2_features2_largest")
m = mod.get_model(97, 2, normal_channel=True).cuda().eval()
csv = pd.read_csv(vp.csv_dir("paper") / "train_set_2.csv")
ds = DP(str(vp.new_data_dir("paper")) + "/", csv.protein_id.values[:8],
        csv.class_id.values[:8], pc_folder="/txt8/")
pts = torch.Tensor(ds[:][0]).cuda()

import models.riconv2_utils_features as uf

def fwd():
    with torch.no_grad():
        o, _ = m(pts)
        if o.dim() == 3:
            o = o.mean(1)
    torch.cuda.synchronize()
    return o.float().cpu().numpy().copy()

import experiments.common.deterministic_fps as df
mods = [n for n in sys.modules if "riconv2_utils" in n]
print("patched modules:", mods, flush=True)
p = FPSPatcher(mode="first", seed=42).install()
_ = fwd()  # warmup (allocator/cudnn first-call effects)
a1, a2 = fwd(), fwd()
p.restore()
print("first-mode bitwise identical:", np.array_equal(a1, a2),
      "max|d|=", float(np.abs(a1 - a2).max()), flush=True)

p = FPSPatcher(mode="seeded", seed=42).install()
p.pass_index = 0; s0a = fwd()
p.pass_index = 0; s0b = fwd()
p.pass_index = 1; s1 = fwd()
p.restore()
print("seeded reproducible:", np.array_equal(s0a, s0b),
      "| distinct pass differs:", not np.array_equal(s0a, s1),
      "| diff mean abs:", round(float(np.abs(s0a - s1).mean()), 6), flush=True)
print("FPS_TEST_DONE", flush=True)
