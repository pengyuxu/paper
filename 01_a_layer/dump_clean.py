"""Dump v3 best-model clean log-probs and 512-d pre-fc3 features for train/val/test.

Outputs (results_final/cache/):
  train_clean.npz: feats(N,512) f16, logp(N,97) f16, y(N)
  val_clean.npz:   feats, logp, y
  test_clean.npz:  feats, logp, ids
Model returns log_softmax; feats captured at fc3 input via pre-hook.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import importlib
from pathlib import Path
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / "models"))
from data_utils.ProteinToPointCloudProcessor import DataProcessor as DSPaired
from data_utils.ProteinToPointCloudProcessorTest import DataProcessor as DSTest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--splits", default="train,val,test",
                    help="comma-separated subset, e.g. 'train,val'")
    ap.add_argument("--log_dir", default="riconv_large_shrec_improved_v3")
    ap.add_argument("--out", default=str(BASE / "results_final" / "cache"))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    mod = importlib.import_module("riconv2_cls_v2_features2_largest")
    model = mod.get_model(97, 2, normal_channel=True).cuda()
    ck = torch.load(BASE / "log/classification_shrec2025" / args.log_dir /
                    "checkpoints/best_model.pth", weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    feats_holder = {}
    model.fc3.register_forward_pre_hook(
        lambda m, inp: feats_holder.__setitem__("x", inp[0].detach()))

    @torch.no_grad()
    def run(loader, paired):
        Fs, Lp, Y = [], [], []
        for batch in tqdm(loader):
            if paired:
                pts, y = batch; Y.append(y.numpy())
            else:
                pts = batch
            pts = pts.cuda()
            out, _ = model(pts)
            if out.dim() == 3: out = out.mean(dim=1)
            Lp.append(out.float().cpu().numpy().astype(np.float16))
            Fs.append(feats_holder["x"].float().cpu().numpy().astype(np.float16))
        return (np.concatenate(Fs), np.concatenate(Lp),
                np.concatenate(Y) if paired else None)

    wanted = set(s.strip() for s in args.splits.split(","))

    if "train" in wanted or "val" in wanted:
        # exact v3 sequential split (first 7428 / last 1816)
        csv = pd.read_csv(BASE / "csv/train_set_2.csv")
        files, labels = csv["protein_id"].values, csv["class_id"].values
        if "train" in wanted:
            tr_ds = DSPaired(str(BASE / "new_data") + "/", files[:7428], labels[:7428],
                             pc_folder="/txt8/")
            tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=16, shuffle=False, num_workers=10)
            f, lp, y = run(tr_ld, True)
            np.savez_compressed(out / "train_clean.npz", feats=f, logp=lp, y=y)
            print("train", f.shape, lp.shape, y.shape,
                  "acc=%.4f" % ((lp.argmax(1) == y).mean()), flush=True)
        if "val" in wanted:
            va_ds = DSPaired(str(BASE / "new_data") + "/", files[7428:], labels[7428:],
                             pc_folder="/txt8/")
            va_ld = torch.utils.data.DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=10)
            f, lp, y = run(va_ld, True)
            np.savez_compressed(out / "val_clean.npz", feats=f, logp=lp, y=y)
            print("val", f.shape, "acc=%.4f" % ((lp.argmax(1) == y).mean()), flush=True)

    if "test" in wanted:
        t = pd.read_csv(BASE / "new_data/test_set_2.csv")
        te_ds = DSTest(str(BASE / "new_data") + "/", t["anonymised_protein_id"],
                       split="test", pc_folder="/txt8/")
        te_ld = torch.utils.data.DataLoader(te_ds, batch_size=16, shuffle=False, num_workers=10)
        f, lp, _ = run(te_ld, False)
        np.savez_compressed(out / "test_clean.npz", feats=f, logp=lp,
                            ids=t["anonymised_protein_id"].values)
        print("test", f.shape, lp.shape, flush=True)


if __name__ == "__main__":
    main()
