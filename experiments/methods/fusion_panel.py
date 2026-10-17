"""Exhaustive fusion-panel scans for the SHREC2025 dual-version rerun.

Reproduces paper Sec 4.7-4.11 protocol:
  p_fused = sum_m w_m softmax(s_m / tau_m),  w_m in {0,1,2}
with *family-level* temperatures
  tau_3DZD in {0.5,1,2,4,8} for members c1,c1b,c1c,c1d
  tau_FPFH in {0.5,1,2}     for members c2,c2b,c2d
  tau = 1                   for c3,c3b,c4 and the deep members.

Panel composition (paper Sec 5.8 / Tables 7-8, order = Table 6):
  P7  = c1,c1b,c1c,c1d,c2,c2b,c2d                        (classical canonical)
  P10 = P7 + c3,c3b,c4                                   (all classical)
  P11 = P10 + v3 MC-FPS K25                              (needs --deep_v3)
  Final fixed config (Sec 4.11): v5-SWA MC25, c1, c2 with weights 2,2,2,
        tau_3DZD=0.5, tau_FPFH=1.0 (needs --deep_v5).

For every panel the scan enumerates all nonzero weight vectors x 15
temperature pairs with a single GEMM per temperature pair, scores validation
macro-F1 (torch, chunked), then applies the one-SE rule with B=300 validation
bootstrap resamples (se computed at the validation argmax config). Selected
and argmax configs are evaluated on val and test; the full ranking is stored
(top-N CSV).

Member caches: experiments/results/<version>/descriptors/<name>_{val,test}.npz
with keys ids/scores/labels.
Deep caches (Task 13): npz with ids/scores/labels; scores may be (K,N,C)
pass logits (aggregated in PROBABILITY space via logsumexp, per MC-FPS)
or (N,C).
"""


import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch

from experiments.common import paths as vp, stats_lib
from experiments.common.metrics_lib import evaluate

# member key -> (cache stem, temperature family)
CLASSICAL = [
    ("c1", "dzd_c1", "dzd"),
    ("c1b", "dzd_c1b", "dzd"),
    ("c1c", "dzd_c1c", "dzd"),
    ("c1d", "dzd_c1d", "dzd"),
    ("c2", "fpfh_c2", "fpfh"),
    ("c2b", "fpfh_c2b", "fpfh"),
    ("c2d", "fpfh_c2d", "fpfh"),
    ("c3", "guerra_c3_trainfit", "one"),
    ("c3b", "guerra_c3b_trainfit", "one"),
    ("c4", "c4", "one"),
]
PANELS = {
    "p7": ["c1", "c1b", "c1c", "c1d", "c2", "c2b", "c2d"],
    "p10": [m[0] for m in CLASSICAL],
}
TAU_DZD = [0.5, 1.0, 2.0, 4.0, 8.0]
TAU_FPFH = [0.5, 1.0, 2.0]
TOP_SAVE = 2000
RNG = np.random.RandomState(20250217)


def load_member(version, stem, split):
    p = vp.RESULTS_DIR / version / "descriptors" / ("%s_%s.npz" % (stem, split))
    d = np.load(p, allow_pickle=True)
    return d["ids"].astype(str), d["scores"].astype(np.float32), d["labels"]


def load_deep(path):
    d = np.load(path, allow_pickle=True)
    S = d["scores"]
    if S.ndim == 3:  # (K,N,C) pass logits -> MC-FPS: mean in PROBABILITY
        # space (logsumexp), never raw-logit arithmetic mean.
        S = stats_lib.mc_aggregate(S.astype(np.float64))
    return d["ids"].astype(str), S.astype(np.float32), d["labels"]


def load_all(version, needed, deep_v3, deep_v5):
    out = {}
    stems = dict((k, s) for k, s, _f in CLASSICAL)
    for key in needed:
        if key in ("v3", "v5"):
            continue
        out[key] = {sp: load_member(version, stems[key], sp)
                    for sp in ("val", "test")}
    deeps = {"v3": deep_v3, "v5": deep_v5}
    for tag in ("v3", "v5"):
        path = deeps[tag]
        if tag in needed and path:
            out[tag] = {"test": load_deep(path),
                        "val": load_deep(path.replace("_test.npz", "_val.npz"))}
    return out


def align(recs, keys, split):
    """Align member score arrays to the first member's id order."""
    ids0, S0, y0 = recs[keys[0]][split]
    Ss = [S0]
    y0 = np.asarray(y0)
    for k in keys[1:]:
        ids, S, y = recs[k][split]
        idx = {v: i for i, v in enumerate(ids)}
        o = np.array([idx[v] for v in ids0])
        Ss.append(S[o])
        assert np.array_equal(np.asarray(y)[o], y0)
    return ids0, np.stack(Ss, 0), y0


def logsoftmax_np(S, tau):
    Z = S.astype(np.float64) / tau
    Z -= Z.max(-1, keepdims=True)
    Z -= np.log(np.exp(Z).sum(-1, keepdims=True))
    return Z


def macro_f1_chunks(L, y, present, chunk=1024, device="cpu"):
    """L: (W,N,C) combined log-mixtures -> macro-F1 per weight row."""
    W, N, C = L.shape
    yt = torch.as_tensor(np.asarray(y), device=device)
    pres = torch.as_tensor(present, device=device, dtype=torch.float64)
    n_pres = pres.sum()
    out = torch.empty(W, dtype=torch.float64, device=device)
    for s in range(0, W, chunk):
        pred = torch.as_tensor(np.ascontiguousarray(L[s:s + chunk]),
                               device=device).argmax(-1)  # (K,N)
        K = pred.shape[0]
        flat_idx = (torch.arange(K, device=device).view(K, 1) * (C * C)
                    + yt.view(1, N) * C + pred).reshape(-1)
        conf = torch.zeros(K * C * C, dtype=torch.float64, device=device)
        conf.index_add_(0, flat_idx,
                        torch.ones_like(flat_idx, dtype=torch.float64))
        conf = conf.view(K, C, C)
        tp = conf.diagonal(dim1=1, dim2=2)
        fp = conf.sum(1) - tp
        fn = conf.sum(2) - tp
        f1 = 2 * tp / torch.clamp(2 * tp + fp + fn, min=1e-12)
        out[s:s + K] = (f1 * pres.view(1, C)).sum(1) / n_pres
    return out.cpu().numpy()


def weight_vectors(M):
    W = np.array(list(itertools.product((0, 1, 2), repeat=M)), dtype=np.float64)
    return W[W.sum(1) > 0]


def _f1_from_pred(pred, y, C, present):
    tp = np.zeros(C); fp = np.zeros(C); fn = np.zeros(C)
    for c in range(C):
        pc = pred == c
        yc = y == c
        tp[c] = np.sum(pc & yc)
        fp[c] = np.sum(pc & ~yc)
        fn[c] = np.sum(~pc & yc)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    return f1[present].mean()


def scan(recs, keys, families, nc, B=300):
    _ids, Sval, yval = align(recs, keys, "val")
    _idt, Ste, yte = align(recs, keys, "test")
    present = np.isin(np.arange(nc), yval)
    Wm = weight_vectors(len(keys))
    M = len(keys)
    records = []
    for td in TAU_DZD:
        for tf in TAU_FPFH:
            taus = [td if f == "dzd" else tf if f == "fpfh" else 1.0
                    for f in families]
            Lm = np.stack([logsoftmax_np(Sval[m], taus[m]) for m in range(M)], 0)
            M_, N, C = Lm.shape
            flat = Lm.reshape(M_, -1)
            for s in range(0, len(Wm), 2048):
                Wb = Wm[s:s + 2048]
                base = (Wb @ flat).reshape(len(Wb), N, C)
                f1v = macro_f1_chunks(base, yval, present)
                for j, fv in enumerate(f1v):
                    gi = s + j
                    records.append({"rank_id": gi, "tau_dzd": td,
                                    "tau_fpfh": tf, "val_f1m": float(fv),
                                    "weights": json.dumps(
                                        [int(x) for x in Wm[gi]])})
    df = pd.DataFrame(records).sort_values(
        "val_f1m", ascending=False).reset_index(drop=True)

    def comb_for(row):
        w = np.array(json.loads(row["weights"]), dtype=np.float64)
        taus = [row.tau_dzd if f == "dzd"
                else row.tau_fpfh if f == "fpfh" else 1.0
                for f in families]
        Lm = np.stack([logsoftmax_np(Sval[m], taus[m]) for m in range(M)], 0)
        Lmt = np.stack([logsoftmax_np(Ste[m], taus[m]) for m in range(M)], 0)
        return ((w @ Lm.reshape(M, -1)).reshape(len(yval), -1),
                (w @ Lmt.reshape(M, -1)).reshape(len(yte), -1))

    # bootstrap SE at the argmax config
    row_star = df.iloc[0]
    Lv_star, _ = comb_for(row_star)
    pred_star = Lv_star.argmax(1)
    N = len(yval)
    se_vals = [_f1_from_pred(pred_star[RNG.randint(0, N, N)], yval, nc, present)
               for _ in range(B)]
    se = float(np.std(se_vals, ddof=1))
    return df, comb_for, (yval, yte, present), se


def simplest_within(df, cutoff):
    cand = df[df.val_f1m >= cutoff].copy()
    cand["nz"] = cand.weights.apply(
        lambda s: sum(x != 0 for x in json.loads(s)))
    cand["wsum"] = cand.weights.apply(lambda s: sum(json.loads(s)))
    cand = cand.sort_values(["nz", "wsum", "val_f1m"],
                            ascending=[True, True, False])
    return cand.iloc[0]


METKEYS = ["acc", "bal_acc", "f1_macro", "prec_macro", "rec_macro",
           "f1_weighted"]


def eval_cfg(row, comb_for, yval, yte, nc, tag, summary):
    Lv, Lt = comb_for(row)
    mv = evaluate(Lv, yval, nc)
    mt = evaluate(Lt, yte, nc)
    summary[tag] = {"weights": json.loads(row["weights"]),
                    "tau_dzd": float(row.tau_dzd),
                    "tau_fpfh": float(row.tau_fpfh),
                    "val": {k: float(mv[k]) for k in METKEYS},
                    "test": {k: float(mt[k]) for k in METKEYS}}
    print("%-14s w=%s t=(%.1f,%.1f) val f1m=%.4f bal=%.4f acc=%.4f | "
          "test f1m=%.4f bal=%.4f acc=%.4f" % (
              tag, summary[tag]["weights"], row.tau_dzd, row.tau_fpfh,
              mv["f1_macro"], mv["bal_acc"], mv["acc"],
              mt["f1_macro"], mt["bal_acc"], mt["acc"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    ap.add_argument("--panels", nargs="+", default=["p7", "p10"],
                    choices=["p7", "p10", "p11", "final"])
    ap.add_argument("--deep_v3", default=None)
    ap.add_argument("--deep_v5", default=None)
    ap.add_argument("--B", type=int, default=300)
    args = ap.parse_args()

    fam_map = {k: f for k, _s, f in CLASSICAL}
    fam_map.update({"v3": "one", "v5": "one"})
    needed = set()
    if "p7" in args.panels:
        needed |= set(PANELS["p7"])
    if "p10" in args.panels or "p11" in args.panels:
        needed |= set(PANELS["p10"])
    if "p11" in args.panels:
        needed.add("v3")
    if "final" in args.panels:
        needed |= {"v5", "c1", "c2"}
    recs = load_all(args.version, needed, args.deep_v3, args.deep_v5)
    nc = vp.version_cfg(args.version)["num_classes"]
    outdir = vp.RESULTS_DIR / args.version / "fusion"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"version": args.version}

    for panel in ("p7", "p10", "p11"):
        if panel not in args.panels:
            continue
        if panel == "p7":
            keys = list(PANELS["p7"])
        elif panel == "p10":
            keys = list(PANELS["p10"])
        else:
            if "v3" not in recs:
                print("skip p11: --deep_v3 missing")
                continue
            keys = list(PANELS["p10"]) + ["v3"]
        fams = [fam_map[k] for k in keys]
        print("== scan", panel, keys, flush=True)
        df, comb_for, (yval, yte, present), se = scan(
            recs, keys, fams, nc, B=args.B)
        df.head(TOP_SAVE).to_csv(
            outdir / ("%s_scan_top%d.csv" % (panel, TOP_SAVE)), index=False)
        cutoff = df.iloc[0].val_f1m - se
        row_onse = simplest_within(df, cutoff)
        print("[%s] argmax val_f1m=%.4f  se(c*)=%.4f  cutoff=%.4f" %
              (panel, df.iloc[0].val_f1m, se, cutoff), flush=True)
        eval_cfg(df.iloc[0], comb_for, yval, yte, nc,
                 panel + "_argmax", summary)
        eval_cfg(row_onse, comb_for, yval, yte, nc,
                 panel + "_onse", summary)
        summary[panel + "_se"] = se

    if "final" in args.panels and "v5" in recs:
        keys = ["v5", "c1", "c2"]
        taus = [1.0, 0.5, 1.0]
        w = np.array([2.0, 2.0, 2.0])
        _i, Sv, yv = align(recs, keys, "val")
        _i, St, yt = align(recs, keys, "test")
        Lm = np.stack([logsoftmax_np(Sv[m], taus[m]) for m in range(3)], 0)
        Lmt = np.stack([logsoftmax_np(St[m], taus[m]) for m in range(3)], 0)
        Lv = (w @ Lm.reshape(3, -1)).reshape(len(yv), -1)
        Lt = (w @ Lmt.reshape(3, -1)).reshape(len(yt), -1)
        mv = evaluate(Lv, yv, nc)
        mt = evaluate(Lt, yt, nc)
        summary["final"] = {"weights": [2, 2, 2], "tau_dzd": 0.5,
                            "tau_fpfh": 1.0,
                            "val": {k: float(mv[k]) for k in METKEYS},
                            "test": {k: float(mt[k]) for k in METKEYS}}
        print("FINAL val f1m=%.4f bal=%.4f acc=%.4f | test f1m=%.4f "
              "bal=%.4f acc=%.4f" % (mv["f1_macro"], mv["bal_acc"],
                                     mv["acc"], mt["f1_macro"],
                                     mt["bal_acc"], mt["acc"]), flush=True)
    elif "final" in args.panels:
        print("skip final: --deep_v5 missing")

    with open(outdir / "fusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved", outdir / "fusion_summary.json")


if __name__ == "__main__":
    main()
