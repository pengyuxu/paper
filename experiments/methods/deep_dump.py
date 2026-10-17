"""Deterministic single-draw / MC-FPS cache dumper for deep runs.

Every deep run (12 queue runs, all based on train_v5.py with the
riconv2_cls_v2_features2_largest architecture) is dumped with the network's
stochastic farthest-point sampling replaced by experiments' deterministic
FPSPatcher:

  --fps first  : D0 = point 0, bitwise-reproducible single forward pass.
  --fps gen    : Ds = initial FPS point from a fixed-seed CPU generator,
                 one bitwise-reproducible single pass.
  --fps seeded : K reproducible, decorrelated passes; pass k uses a
                 splitmix64 hash of (seed, k, fps_call, batch_row).
                 MC-K aggregates the first K passes (mean logits).

Determinism requirements (must precede torch import):
  CUBLAS_WORKSPACE_CONFIG=:4096:8  (set here via setdefault)
  cudnn deterministic, torch.use_deterministic_algorithms, deterministic
  eigh sign -- all enabled inside FPSPatcher.install().

Outputs (fusion-compatible): experiments/results/<version>/deep/
  <run>__<ckpt>__<fps><seed>_K<K>_{val,test}.npz
    ids    (N,) str ('.vtk' stripped)
    scores (K,N,C) float16 raw logits (per FPS pass; 3-D head averaged)
    labels (N,) int64

MC aggregation downstream: scores.mean(0) == (1/K) sum_k f(x;s_k).

Usage:
  python -m experiments.methods.deep_dump --version paper --run v5_s2 \
      --ckpt swa_model.pth --fps seeded --seed 42 --K 25 --gpu 0
"""
import os
import sys
import argparse

# REQUIRED before torch import for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _av = sys.argv[1:]
    _gpu = "0"
    for _i, _a in enumerate(_av):
        if _a == "--gpu" and _i + 1 < len(_av):
            _gpu = _av[_i + 1]
        elif _a.startswith("--gpu="):
            _gpu = _a.split("=", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

from pathlib import Path  # noqa: E402
import importlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NEW = REPO / "newcode"
sys.path.insert(0, str(NEW))
sys.path.insert(0, str(NEW / "models"))
sys.path.insert(0, str(REPO))

from data_utils.ProteinToPointCloudProcessor import DataProcessor as DSPaired  # noqa: E402
from data_utils.ProteinToPointCloudProcessorTest import DataProcessor as DSTest  # noqa: E402
from experiments.common import paths as vp  # noqa: E402
from experiments.common.deterministic_fps import FPSPatcher  # noqa: E402
from experiments.common.metrics_lib import evaluate  # noqa: E402


def strip_vtk(a):
    return np.array([str(x)[:-4] if str(x).endswith(".vtk") else str(x)
                     for x in a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    ap.add_argument("--run", required=True,
                    help="log_dir name under work/<ver>/log/classification_shrec2025/")
    ap.add_argument("--ckpt", default="best_model.pth")
    ap.add_argument("--fps", choices=["seeded", "first", "gen"],
                    default="seeded")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=25)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    if args.fps in ("first", "gen"):
        args.K = 1
    vcfg = vp.version_cfg(args.version)
    nc = vcfg["num_classes"]
    outdir = vp.RESULTS_DIR / args.version / "deep"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "%s__%s__%s%s_K%d" % (args.run, args.ckpt.replace(".pth", ""),
                                args.fps, args.seed, args.K)

    mod = importlib.import_module("riconv2_cls_v2_features2_largest")
    model = mod.get_model(nc, 2, normal_channel=True).cuda()
    ckpath = (vp.log_dir(args.version) / args.run / "checkpoints" / args.ckpt)
    ck = torch.load(ckpath, weights_only=False, map_location="cuda")
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print("loaded %s epoch=%s n_averaged=%s fps=%s seed=%d K=%d" %
          (ckpath, ck.get("epoch", "?"), ck.get("n_averaged", "-"),
           args.fps, args.seed, args.K), flush=True)

    # Install AFTER importing the model/utils modules so auto-discovery finds
    # both riconv2_utils* module objects.
    patcher = FPSPatcher(mode=args.fps, seed=args.seed).install()

    # ---------------- data ----------------
    tcsv = pd.read_csv(vp.csv_dir(args.version) / "train_set_2.csv")
    va_ids = tcsv["protein_id"].values[vp.TRAIN_ROWS:]
    va_y = tcsv["class_id"].values[vp.TRAIN_ROWS:].astype(np.int64)
    va_ds = DSPaired(str(vp.new_data_dir(args.version)) + "/", va_ids, va_y,
                     pc_folder="/txt8/")
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=16, shuffle=False,
                                        num_workers=args.workers)

    tdf = pd.read_csv(vcfg["dataset_dir"] / vcfg["test_list_file"])
    idcol = vcfg["test_id_col"]
    te_ids_raw = tdf[idcol].values
    te_ids = strip_vtk(te_ids_raw)
    gtdf = pd.read_csv(vp.csv_dir(args.version) / vcfg["gt_file"])
    g = gtdf.set_index(vcfg["gt_anon_col"])[vcfg["gt_label_col"]].to_dict()
    gmap = {str(k)[:-4] if str(k).endswith(".vtk") else str(k): int(v)
            for k, v in g.items()}
    te_y = np.array([gmap[i] for i in te_ids], np.int64)
    te_ds = DSTest(str(vp.new_data_dir(args.version)) + "/", te_ids_raw,
                   split="test", pc_folder="/txt8/")
    te_ld = torch.utils.data.DataLoader(te_ds, batch_size=16, shuffle=False,
                                        num_workers=args.workers)

    @torch.no_grad()
    def dump(loader, n):
        out = np.zeros((args.K, n, nc), np.float16)
        for k in range(args.K):
            patcher.pass_index = k
            rows = []
            for batch in tqdm(loader, desc="%s pass%d" % (args.run, k + 1),
                              leave=False):
                pts = batch[0] if isinstance(batch, (list, tuple)) else batch
                o, _ = model(pts.cuda())
                if o.dim() == 3:
                    o = o.mean(dim=1)
                rows.append(o.float().cpu().numpy().astype(np.float16))
            out[k] = np.concatenate(rows)
        return out

    Pv = dump(va_ld, len(va_ids))
    Pt = dump(te_ld, len(te_ids))
    np.savez_compressed(outdir / ("%s_val.npz" % tag),
                        ids=strip_vtk(va_ids), scores=Pv, labels=va_y)
    np.savez_compressed(outdir / ("%s_test.npz" % tag),
                        ids=te_ids, scores=Pt, labels=te_y)

    for name, P, y in (("val", Pv, va_y), ("test", Pt, te_y)):
        for kk in sorted({1, 3, min(25, args.K)}):
            if kk > args.K:
                continue
            m = evaluate(P[:kk].mean(0), y, nc)
            print("%s MC%-2d acc=%.4f bal=%.4f f1m=%.4f" %
                  (name, kk, m["acc"], m["bal_acc"], m["f1_macro"]),
                  flush=True)
        if args.K > 1:
            ps = [evaluate(P[k], y, nc)["f1_macro"]
                  for k in range(args.K)]
            print("%s single-pass f1m mean=%.4f sd=%.4f range=[%.4f,%.4f]"
                  % (name, float(np.mean(ps)), float(np.std(ps, ddof=1)),
                         float(np.min(ps)), float(np.max(ps))), flush=True)
    print("DEEP_DUMP_DONE", tag, flush=True)


if __name__ == "__main__":
    main()
