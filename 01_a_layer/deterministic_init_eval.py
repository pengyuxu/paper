"""Route C: deterministic-initialization control (paper Section V-G).

Requires numpy >= 2 (the checkpoints were pickled with numpy 2.x) and
torch; no other dependencies. The stochastic FPS initial point is
replaced at runtime (the shipped model files are untouched) with two
deterministic rules:
  D0 : initial centroid = point index 0
  Ds : initial centroid from a fixed-seed RNG (reproducible pseudo-random)
and compared against the random single draw (mean over 25 draws) and
MC-FPS K=25, on validation and test for the v3 and v5-s2 checkpoints.

Point clouds are read directly as .npy (new_data/{id}_pc.npy).
Writes results_final/deterministic_init_summary.json.
"""
import os, sys, json, importlib, argparse
from pathlib import Path
import numpy as np
import torch

assert int(np.__version__.split(".")[0]) >= 2, "numpy >= 2 required"

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "01_a_layer" / "models"))
import riconv2_utils_features as U


class DS(torch.utils.data.Dataset):
    def __init__(self, root, ids, y, folder="/"):
        names = [i[:-4] if i.endswith(".vtk") else i for i in ids]
        self.x = np.stack([np.load(root + folder + n + "_pc.npy")
                           for n in names]).astype(np.float32)
        self.y = np.asarray(y, np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


PATCH_STATE = {"mode": "random"}


def fps_patched(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    if PATCH_STATE["mode"] == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif PATCH_STATE["mode"] == "seeded":
        g = torch.Generator(device="cpu").manual_seed(0)
        farthest = torch.randint(0, N, (B,), generator=g,
                                 dtype=torch.long).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    bidx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[bidx, farthest, :].view(B, 1, 3)
        dist = ((xyz - c) ** 2).sum(-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def load_model(log_dir, ckpt):
    mod = importlib.import_module("riconv2_cls_v2_features2_largest")
    model = mod.get_model(97, 2, normal_channel=True).cuda()
    state = torch.load(BASE / "log/classification_shrec2025" / log_dir /
                       "checkpoints" / ckpt, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model.eval()


@torch.no_grad()
def run_pass(model, loader):
    out = []
    for batch in loader:
        pts, _ = batch
        o, _ = model(pts.cuda())
        out.append(o.float().cpu())
    return torch.cat(out).numpy()


def metrics(lp, y):
    p = lp.argmax(1)
    classes = np.unique(np.concatenate([y, p]))
    rec, f1 = [], []
    for c in classes:
        tp = int(((p == c) & (y == c)).sum())
        gc, pc = int((y == c).sum()), int((p == c).sum())
        rec.append(tp / gc if gc else 0.0)
        f1.append(2 * tp / (gc + pc) if (gc + pc) else 0.0)
    return {"acc": float((p == y).mean()),
            "bal": float(np.mean(rec)), "mac": float(np.mean(f1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    U.farthest_point_sample = fps_patched

    ids = np.genfromtxt(BASE / "csv/train_set_2.csv", delimiter=",",
                        skip_header=1, dtype=str, usecols=0)
    yall = np.genfromtxt(BASE / "csv/train_set_2.csv", delimiter=",",
                         skip_header=1, dtype=np.int64, usecols=1)
    tids = np.genfromtxt(BASE / "csv/test_set_2.csv", delimiter=",",
                         skip_header=1, dtype=str, usecols=0)
    root = str(BASE / "new_data") + "/"
    yv, yt = yall[7428:], np.genfromtxt(
        BASE / "csv/test_set_ground_truth.csv", delimiter=",", skip_header=1,
        dtype=np.int64, usecols=1)
    va_ld = torch.utils.data.DataLoader(
        DS(root, ids[7428:], yv), batch_size=16, shuffle=False, num_workers=4)
    te_ld = torch.utils.data.DataLoader(
        DS(root, tids, yt), batch_size=16, shuffle=False, num_workers=4)

    checkpoints = [("v3", "riconv_large_shrec_improved_v3", "best_model.pth"),
                   ("s2", "riconv_large_v5_s2", "swa_model.pth")]
    summary = {}
    for name, ld, ck in checkpoints:
        model = load_model(ld, ck)
        entry = {}
        for mode, loader, y, tag in [
                ("first", va_ld, yv, "val"), ("seeded", va_ld, yv, "val"),
                ("first", te_ld, yt, "test"), ("seeded", te_ld, yt, "test")]:
            PATCH_STATE["mode"] = mode
            lp = run_pass(model, loader)
            entry["%s_%s" % (mode, tag)] = metrics(lp, y)
            if name == "v3" and mode == "first" and tag == "val":
                entry["repeat_identical"] = bool(
                    np.array_equal(lp, run_pass(model, loader)))
        summary[name] = entry
        print("[%s]" % name, json.dumps(entry, indent=1), flush=True)

    out = BASE / "results_final/deterministic_init_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
