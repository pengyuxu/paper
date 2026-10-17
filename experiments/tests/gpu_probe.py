"""GPU probe: tiny dataset, immediate forward/backward, prints memory/timing."""
import os, sys, time, importlib
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NEWCODE = os.path.join(REPO, "newcode")
sys.path.append(os.path.join(NEWCODE, "models"))
sys.path.append(NEWCODE)
sys.path.append(REPO)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import numpy as np
import pandas as pd
import torch
from data_utils.ProteinToPointCloudProcessor import DataProcessor as DP
from experiments.common import paths as vp

print("avail", torch.cuda.is_available(), "n", torch.cuda.device_count(), flush=True)
mod = importlib.import_module("riconv2_cls_v2_features2_largest")
m = mod.get_model(97, 2, normal_channel=True).cuda().train()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
v = "paper"
csv = pd.read_csv(vp.csv_dir(v) / "train_set_2.csv")
ds = DP(str(vp.new_data_dir(v)) + "/", csv.protein_id.values[:32],
        csv.class_id.values[:32], pc_folder="/txt8/")
ld = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=2,
                                  persistent_workers=True)
for i, (pts, y) in enumerate(ld):
    pts, y = pts.cuda(), y.cuda()
    t = time.time()
    o, _ = m(pts)
    if o.dim() == 3:
        o = o.mean(1)
    loss = torch.nn.functional.cross_entropy(o, y.long())
    loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    print("step", i, "dt=%.3fs" % (time.time() - t),
          "memMB=%.0f" % (torch.cuda.memory_allocated() / 1e6), flush=True)
    if i >= 3:
        break
print("PROBE_DONE", flush=True)
