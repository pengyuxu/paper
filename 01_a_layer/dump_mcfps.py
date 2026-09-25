"""Monte-Carlo over FPS random seeds: repeated CLEAN forward passes (no
rotation), all log-probs saved for nested K selection.

Outputs cache/mcfps_{val,test}_logp.npy: (K,N,97) f16.
The model is stochastic at eval because farthest-point sampling seeds from a
random point at every set-abstraction level; averaging logp removes that
variance (same idea as MC-dropout, different noise source).

--log_dir / --ckpt allow reuse for the v5 checkpoints.
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
    ap.add_argument("--log_dir", default="riconv_large_shrec_improved_v3")
    ap.add_argument("--ckpt", default="best_model.pth")
    ap.add_argument("--K", type=int, default=25)
    ap.add_argument("--prefix", default="mcfps")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    out = BASE / "results_final" / "cache"

    mod = importlib.import_module("riconv2_cls_v2_features2_largest")
    model = mod.get_model(97, 2, normal_channel=True).cuda()
    ck = torch.load(BASE / "log/classification_shrec2025" / args.log_dir /
                    "checkpoints" / args.ckpt, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    print("loaded %s epoch=%s ; K=%d clean passes"
          % (args.ckpt, ck.get("epoch", "?"), args.K), flush=True)

    csv = pd.read_csv(BASE / "csv/train_set_2.csv")
    va_ds = DSPaired(str(BASE / "new_data") + "/",
                     csv["protein_id"].values[7428:], csv["class_id"].values[7428:],
                     pc_folder="/txt8/")
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=10)
    t = pd.read_csv(BASE / "new_data/test_set_2.csv")
    te_ds = DSTest(str(BASE / "new_data") + "/", t["anonymised_protein_id"],
                   split="test", pc_folder="/txt8/")
    te_ld = torch.utils.data.DataLoader(te_ds, batch_size=16, shuffle=False, num_workers=10)

    @torch.no_grad()
    def dump(loader, n):
        allp = np.zeros((args.K, n, 97), np.float16)
        for k in range(args.K):
            Lp = []
            for batch in tqdm(loader, desc="pass%d" % (k + 1), leave=False):
                pts = batch[0] if isinstance(batch, (list, tuple)) else batch
                o, _ = model(pts.cuda())
                Lp.append(o.float().cpu().numpy().astype(np.float16))
            allp[k] = np.concatenate(Lp)
        return allp

    pv = dump(va_ld, 1816)
    np.save(out / ("%s_val_logp.npy" % args.prefix), pv)
    print("val saved", pv.shape, flush=True)
    pt = dump(te_ld, 2321)
    np.save(out / ("%s_test_logp.npy" % args.prefix), pt)
    print("test saved", pt.shape, "ALL_MCFPS_DONE" % (), flush=True)


if __name__ == "__main__":
    main()
