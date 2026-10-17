"""Layer-B rotational TTA evaluation (paper Sec. 5.6).

25 views = identity + 24 proper cube rotations; mean-LOGIT aggregation
(not mean-probability). FPS kept seeded-deterministic (pass 0) so every
view sees the same draw -- isolates the rotation effect from FPS noise.

Usage:
  python -m experiments.methods.tta_eval --version paper \
      --run v3_noSynth_s2 --ckpt best_model.pth --gpu 0

Output: experiments/results/<ver>/deep/<run>__<ckpt>__tta25_{val,test}.npz
  ids (N,), scores (25,N,C) float16 raw logits per view, labels (N,)
"""
import os
import sys
import argparse

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
from experiments.common.rot_tta import cube_rotations, apply_rotation  # noqa: E402

from experiments.methods.deep_dump import strip_vtk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    ap.add_argument("--run", default="v3_noSynth_s2")
    ap.add_argument("--ckpt", default="best_model.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    vcfg = vp.version_cfg(args.version)
    nc = vcfg["num_classes"]
    outdir = vp.RESULTS_DIR / args.version / "deep"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "%s__%s__tta25" % (args.run, args.ckpt.replace(".pth", ""))

    mod = importlib.import_module("riconv2_cls_v2_features2_largest")
    model = mod.get_model(nc, 2, normal_channel=True).cuda()
    ckpath = (vp.log_dir(args.version) / args.run / "checkpoints" / args.ckpt)
    ck = torch.load(ckpath, weights_only=False, map_location="cuda")
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print("loaded %s epoch=%s" % (ckpath, ck.get("epoch", "?")), flush=True)

    patcher = FPSPatcher(mode="seeded", seed=args.seed).install()
    patcher.pass_index = 0  # same draw for all 25 views

    rots = [np.eye(3, dtype=np.float32)] + cube_rotations()
    assert len(rots) == 25

    tcsv = pd.read_csv(vp.csv_dir(args.version) / "train_set_2.csv")
    va_ids = tcsv["protein_id"].values[vp.TRAIN_ROWS:]
    va_y = tcsv["class_id"].values[vp.TRAIN_ROWS:].astype(np.int64)
    va_ds = DSPaired(str(vp.new_data_dir(args.version)) + "/", va_ids, va_y,
                     pc_folder="/txt8/")
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=16, shuffle=False,
                                        num_workers=args.workers)

    tdf = pd.read_csv(vcfg["dataset_dir"] / vcfg["test_list_file"])
    te_ids_raw = tdf[vcfg["test_id_col"]].values
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
        out = np.zeros((25, n, nc), np.float16)
        for vi, R in enumerate(rots):
            Rt = torch.as_tensor(R, device="cuda")
            rows = []
            for batch in tqdm(loader, desc="view%02d" % vi, leave=False):
                pts = batch[0] if isinstance(batch, (list, tuple)) else batch
                pts = apply_rotation(pts.cuda(), Rt)
                o, _ = model(pts)
                if o.dim() == 3:
                    o = o.mean(dim=1)
                rows.append(o.float().cpu().numpy().astype(np.float16))
            out[vi] = np.concatenate(rows)
        return out

    Pv = dump(va_ld, len(va_ids))
    Pt = dump(te_ld, len(te_ids))
    np.savez_compressed(outdir / ("%s_val.npz" % tag),
                        ids=strip_vtk(va_ids), scores=Pv, labels=va_y)
    np.savez_compressed(outdir / ("%s_test.npz" % tag),
                        ids=te_ids, scores=Pt, labels=te_y)

    for name, P, y in (("val", Pv, va_y), ("test", Pt, te_y)):
        m = evaluate(P.mean(0).astype(np.float64), y, nc)
        print("%s TTA25 mean-logit acc=%.4f bal=%.4f f1m=%.4f" %
              (name, m["acc"], m["bal_acc"], m["f1_macro"]), flush=True)
        m1 = evaluate(P[0].astype(np.float64), y, nc)
        print("%s identity-only  acc=%.4f bal=%.4f f1m=%.4f" %
              (name, m1["acc"], m1["bal_acc"], m1["f1_macro"]), flush=True)
    print("TTA_EVAL_DONE", tag, flush=True)


if __name__ == "__main__":
    main()
