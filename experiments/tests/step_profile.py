"""Profile a faithful train_v5 step: data wait / h2d / fwd / bwd split."""
import os, sys, time, importlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NEWCODE = os.path.join(REPO, "newcode")
sys.path += [os.path.join(NEWCODE, "models"), NEWCODE, REPO]
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from data_utils.ProteinToPointCloudProcessor import DataProcessor as DP
from experiments.common import paths as vp

v = sys.argv[1] if len(sys.argv) > 1 else "paper"
nc = vp.VERSIONS[v]["num_classes"]
csv = pd.read_csv(vp.csv_dir(v) / "train_set_2.csv")
ds = DP(str(vp.new_data_dir(v)) + "/", csv.protein_id.values[:512],
        csv.class_id.values[:512], pc_folder="/txt8/")
ld = DataLoader(ds, batch_size=16, shuffle=True, num_workers=10)
mod = importlib.import_module("riconv2_cls_v2_features2_largest")
m = mod.get_model(nc, 2, normal_channel=True).cuda().train()
opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
it = iter(ld)
for i in range(12):
    t0 = time.time()
    pts, y = next(it)
    t1 = time.time()
    pts = pts.cuda(non_blocking=False); y = y.cuda()
    torch.cuda.synchronize(); t2 = time.time()
    o, _ = m(pts)
    torch.cuda.synchronize(); t3 = time.time()
    loss = torch.nn.functional.cross_entropy(o, y.long())
    loss.backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); t4 = time.time()
    if i >= 4:
        print("data=%.3f h2d=%.3f fwd=%.3f bwd=%.3f total=%.3f"
              % (t1-t0, t2-t1, t3-t2, t4-t3, t4-t0), flush=True)
print("PROFILE_DONE", flush=True)
