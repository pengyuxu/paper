"""Dump v5 variant val/test log-probs (clean single pass).

Outputs results_final/cache/v5_{tag}_val.npz {logp(N,97) f16, y}
        results_final/cache/v5_{tag}_test.npz {logp(N,97) f16, ids}
Same data pipeline / model module as dump_clean.py (identical architecture).
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
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--ckpt", default="best_macrof1.pth")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out = BASE / "results_final" / "cache"

    mod = importlib.import_module("riconv2_cls_v2_features2_largest")
    model = mod.get_model(97, 2, normal_channel=True).cuda()
    ck = torch.load(BASE / "log/classification_shrec2025" / args.log_dir /
                    "checkpoints" / args.ckpt, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    print("loaded %s epoch=%s" % (args.ckpt, ck.get("epoch", "?")), flush=True)

    @torch.no_grad()
    def run(loader, paired):
        Lp, Y = [], []
        for batch in tqdm(loader):
            if paired:
                pts, y = batch; Y.append(y.numpy())
            else:
                pts = batch
            pts = pts.cuda()
            o, _ = model(pts)
            if o.dim() == 3: o = o.mean(dim=1)
            Lp.append(o.float().cpu().numpy().astype(np.float16))
        return np.concatenate(Lp), (np.concatenate(Y) if paired else None)

    csv = pd.read_csv(BASE / "csv/train_set_2.csv")
    va_ds = DSPaired(str(BASE / "new_data") + "/",
                     csv["protein_id"].values[7428:], csv["class_id"].values[7428:],
                     pc_folder="/txt8/")
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=10)
    lp, y = run(va_ld, True)
    np.savez_compressed(out / ("v5_%s_val.npz" % args.tag), logp=lp, y=y)
    print("%s val" % args.tag, lp.shape, "acc=%.4f" % ((lp.argmax(1) == y).mean()), flush=True)

    t = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_ds = DSTest(str(BASE / "new_data") + "/", t["anonymised_protein_id"],
                   split="test", pc_folder="/txt8/")
    te_ld = torch.utils.data.DataLoader(te_ds, batch_size=16, shuffle=False, num_workers=10)
    lp, _ = run(te_ld, False)
    np.savez_compressed(out / ("v5_%s_test.npz" % args.tag), logp=lp,
                        ids=t["anonymised_protein_id"].values)
    print("%s test" % args.tag, lp.shape, flush=True)


if __name__ == "__main__":
    main()
