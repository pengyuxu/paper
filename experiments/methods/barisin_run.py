"""Barisin et al. RINetwork method, unified wrapper for both dataset versions.

Faithful re-implementation driver around the ORIGINAL code under
methods/barisin/ (models + provider are imported from there, never copied):

  recipe (identical to train_classification_protein{,_features2}.py):
    run1 = riconv2_cls_v2_largest   (geometry, input sliced to 6ch in training)
    run2 = riconv2_cls_v2_features2_largest (8ch geometry + potentials)
    250 ep, bs16, Adam lr1e-3 wd1e-4, StepLR(step20,gamma.7) called BEFORE the
    first epoch (replicated), plain CrossEntropyLoss, class weights computed
    but unused (as in the released scripts), num_workers10, drop_last train.

Deviations (documented):
  - paths/CSV parametrised via experiments.common.paths; point clouds are read
    from our prebuilt folders (paper profile = work/clouds/txt8 by default;
    sanity can point to the barisin-profile clouds).
  - official scripts set NO random seed; --seed pins torch/numpy/random for
    reproducibility.
  - checkpoints/metrics go to work/<ver>/log/classification_shrec2025/<log_dir>.

Modes:
  train  : full recipe training with automatic best_model.pth resume
  infer  : dump ids + probs (+labels) npz for val or test split
"""
from __future__ import annotations
import argparse
import csv as _csv
import importlib
import os, sys, random

# --- CUDA shim: honour --gpu before importing torch -------------------------
_av = sys.argv
_gpu = "0"
for _i, _a in enumerate(_av):
    if _a == "--gpu" and _i + 1 < len(_av):
        _gpu = _av[_i + 1]
    elif _a.startswith("--gpu="):
        _gpu = _a.split("=", 1)[1]
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BARISIN = os.path.join(REPO, "methods", "barisin")
sys.path.insert(0, BARISIN)                      # top-level `provider`
sys.path.insert(0, os.path.join(BARISIN, "models"))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from experiments.common import paths as vp
from experiments.common import metrics_lib

RUN_MODEL = {
    1: "riconv2_cls_v2_largest",
    2: "riconv2_cls_v2_features2_largest",
}


class CloudDataset(Dataset):
    """Loads 8192x8 clouds fully into RAM, matching the original DataProcessor."""

    def __init__(self, cloud_dir, names, labels=None, strip_vtk=False):
        arrs = []
        for nm in names:
            key = str(nm)
            if strip_vtk and key.endswith(".vtk"):
                key = key[:-4]
            p = os.path.join(cloud_dir, key + "_pc.npy")
            arrs.append(np.load(p).astype(np.float32))
        self.points = np.array(arrs)
        self.labels = (None if labels is None
                       else np.asarray(labels, dtype=np.int64))

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, i):
        if self.labels is None:
            return self.points[i]
        return self.points[i], self.labels[i]


def official_balanced_acc(model_out, target, num_class):
    """Replica of Barisin test() per-class accumulation macro average."""
    class_acc = np.zeros((num_class, 3))
    for cat in np.unique(target.cpu()):
        c = model_out[target == cat].eq(target[target == cat].long()).cpu().sum()
        class_acc[cat, 0] += c.item() / float(model_out[target == cat].numel())
        class_acc[cat, 1] += 1
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    return float(np.mean(class_acc[:, 2]))


def evaluate(model, loader, device, run, num_class, max_batches=0):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
            pts, y = batch
            pts = pts.to(device)
            pred, _ = model(pts[:, :, :6] if run == 1 else pts)
            if pred.dim() == 3:
                pred = pred.mean(1)
            all_p.append(pred.exp().cpu().numpy())
            all_y.append(y.numpy())
            if max_batches and bi + 1 >= max_batches:
                break
    probs = np.concatenate(all_p, 0)
    y = np.concatenate(all_y, 0)
    pred = probs.argmax(1)
    acc = float((pred == y).mean())
    bal = official_balanced_acc(torch.as_tensor(pred), torch.as_tensor(y),
                                num_class)
    return acc, bal, probs, y


def build_model(run, num_class, device):
    mod = importlib.import_module(RUN_MODEL[run])
    model = mod.get_model(num_class, 2, normal_channel=True)
    for m in model.modules():
        if "ReLU" in m.__class__.__name__:
            m.inplace = True
    return model.to(device)


def cloud_dir(profile):
    return str(vp.WORK_ROOT / "clouds" /
               ("txt8" if profile == "paper" else "txt8_barisin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, choices=[1, 2], required=True)
    ap.add_argument("--mode", choices=["train", "infer"], required=True)
    ap.add_argument("--version", choices=["paper", "updated"], default="updated")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--epoch", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--cloud_profile", choices=["paper", "barisin"],
                    default="paper")
    ap.add_argument("--cloud_dir", default=None,
                    help="override cloud folder (e.g. official 2321 clouds)")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--test_list_csv", default=None,
                    help="override test id-list CSV (strict 2321 sanity)")
    ap.add_argument("--gt_csv", default=None,
                    help="override ground-truth CSV")
    ap.add_argument("--id_col", default=None)
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_train_batches", type=int, default=0)
    ap.add_argument("--max_val_batches", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    cfg = vp.version_cfg(args.version)
    num_class = cfg["num_classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = args.log_dir or ("barisin_run%d" % args.run)
    exp_dir = vp.log_dir(args.version) / log_dir
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    cdir = args.cloud_dir or cloud_dir(args.cloud_profile)
    print("Barisin run%d version=%s model=%s device=%s classes=%d clouds=%s"
          % (args.run, args.version, RUN_MODEL[args.run], device, num_class,
             cdir), flush=True)

    model = build_model(args.run, num_class, device)
    provider = importlib.import_module("provider")  # methods/barisin/provider.py

    # ---------------- data ----------------
    train_csv = pd.read_csv(vp.csv_dir(args.version) / "train_set_2.csv")
    names_all, lab_all = train_csv.protein_id.values, train_csv.class_id.values
    tr_n, va_n = names_all[:vp.TRAIN_ROWS], lab_all[:vp.TRAIN_ROWS]
    va_id, va_y = names_all[vp.TRAIN_ROWS:], lab_all[vp.TRAIN_ROWS:]

    def infer_data():
        if args.split == "val":
            return va_id, va_y, False
        list_csv = args.test_list_csv or str(
            cfg["dataset_dir"] / cfg["test_list_file"])
        id_col = args.id_col or cfg["test_id_col"]
        lab_col = args.label_col or cfg["gt_label_col"]
        tdf = pd.read_csv(list_csv)
        ids = tdf[id_col].values
        gt = pd.read_csv(args.gt_csv or
                         (vp.csv_dir(args.version) / cfg["gt_file"]))
        anon_col = (cfg["gt_anon_col"] if args.gt_csv is None
                    else (args.id_col or id_col))
        g = gt.set_index(anon_col)[lab_col].to_dict()
        gmap = {str(k)[:-4] if str(k).endswith(".vtk") else str(k): int(v)
                for k, v in g.items()}
        key = [str(x)[:-4] if str(x).endswith(".vtk") else str(x)
               for x in ids]
        labels = np.array([gmap[k] for k in key], dtype=np.int64)
        return ids, labels, True

    if args.mode == "train":
        tr_ds = CloudDataset(cdir, tr_n, lab_all[:vp.TRAIN_ROWS])
        va_ds = CloudDataset(cdir, va_id, va_y)
        tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, drop_last=True)
        va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers)
        criterion = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)

        ck = exp_dir / "checkpoints" / "best_model.pth"
        start_epoch, best_acc = 0, 0.0
        if ck.exists():
            sd = torch.load(ck, map_location=device, weights_only=False)
            model.load_state_dict(sd["model_state_dict"])
            opt.load_state_dict(sd["optimizer_state_dict"])
            start_epoch, best_acc = int(sd["epoch"]), float(sd["instance_acc"])
            print("resume from %s epoch=%d best=%.4f" % (ck, start_epoch, best_acc),
                  flush=True)
        rows = [["epoch", "lr", "train_acc", "val_acc", "val_bal"]]
        for epoch in range(start_epoch, args.epoch):
            model.train(); sched.step()      # official: step BEFORE first epoch
            cor, tot, loss_sum = 0, 0, 0.0
            tbar = tqdm(tr_ld, desc="ep%d" % (epoch + 1), smoothing=0.9,
                        leave=False)
            for bi, (pts, y) in enumerate(tbar):
                opt.zero_grad()
                pts = pts.numpy()
                pts = provider.random_point_dropout(pts)
                pts[:, :, 0:3] = provider.random_scale_point_cloud(pts[:, :, 0:3])
                pts[:, :, 0:3] = provider.shift_point_cloud(pts[:, :, 0:3])
                pts = torch.as_tensor(np.ascontiguousarray(pts)).to(device)
                y = y.to(device)
                pred, _ = model(pts[:, :, :6] if args.run == 1 else pts)
                if pred.dim() == 3:
                    y2 = y.unsqueeze(-1).repeat(1, pred.shape[1]).view(-1)
                    loss = criterion(pred.contiguous().view(-1, num_class), y2)
                    pred = pred.mean(1)
                else:
                    loss = criterion(pred, y.long())
                loss.backward(); opt.step()
                cor += pred.argmax(1).eq(y.long()).sum().item()
                tot += pts.shape[0]; loss_sum += float(loss)
                if args.max_train_batches and bi + 1 >= args.max_train_batches:
                    break
            train_acc = cor / max(tot, 1)
            vacc, vbal, _, _ = evaluate(model, va_ld, device, args.run,
                                        num_class, args.max_val_batches)
            rows.append([epoch + 1, opt.param_groups[0]["lr"],
                         "%.6f" % train_acc, "%.6f" % vacc, "%.6f" % vbal])
            print("ep%d lr=%.6f train=%.4f val acc=%.4f bal=%.4f (best %.4f)"
                  % (epoch + 1, opt.param_groups[0]["lr"], train_acc, vacc,
                     vbal, best_acc), flush=True)
            if vacc >= best_acc:
                best_acc = vacc
                torch.save({"epoch": epoch + 1, "instance_acc": vacc,
                            "class_acc": vbal,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict()}, ck)
            with open(exp_dir / "logs" / "val_metrics.csv", "w", newline="") as f:
                w = _csv.writer(f); w.writerows(rows)
        print("done best=%.4f -> %s" % (best_acc, ck), flush=True)

    else:  # infer
        ids, labels, strip = infer_data()
        ds = CloudDataset(cdir, ids, labels, strip_vtk=strip)
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)
        ck = args.ckpt or str(exp_dir / "checkpoints" / "best_model.pth")
        sd = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(sd["model_state_dict"])
        probs = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(ld, desc="infer"):
                pts = batch[0].to(device)
                pred, _ = model(pts[:, :, :6] if args.run == 1 else pts)
                if pred.dim() == 3:
                    pred = pred.mean(1)
                probs.append(pred.exp().cpu().numpy())
        probs = np.concatenate(probs, 0)
        m = metrics_lib.evaluate(probs, labels, num_class)
        print("infer run%d %s n=%d ckpt=%s" %
              (args.run, args.split, len(ids), ck), flush=True)
        for k in metrics_lib.METRIC_ORDER:
            print("  %-14s %.4f" % (k, m[k]), flush=True)
        out = args.out or str(exp_dir / ("preds_%s.npz" % args.split))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, ids=np.array([str(x) for x in ids]),
                            probs=probs.astype(np.float32), labels=labels)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()
