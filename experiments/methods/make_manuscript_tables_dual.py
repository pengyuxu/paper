"""Render dual-version (paper / updated) manuscript tables and prose stats.

Task 18. Every table value and every prose statistic is read from a generated
cache or recomputed deterministically from one; nothing is hand-copied. The
only exception, mandated by decision A5, is the Table 8 track-report integer
block, which is transcribed verbatim from the organisers' bar charts and is
kept in a clearly labelled separate block (those integers cannot be regenerated

from any released per-sample cache).

Outputs:
  experiments/results/manuscript_tables_dual.md   (paste-ready tables)
  experiments/results/manuscript_stats.json       (prose statistics)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import paths as vp, stats_lib
from experiments.common.metrics_lib import evaluate, softmax
from experiments.common import diagnostics as dg
from experiments.methods import fusion_panel as fp

R = vp.RESULTS_DIR
VERS = ("paper", "updated")
NC = {v: int(vp.version_cfg(v)["num_classes"]) for v in VERS}
RNG = np.random.default_rng(20250217)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def x4(x):
    """0.9347 -> '.9346' style used in the manuscript (leading zero dropped)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return ("%.4f" % x)[1:] if abs(x) < 1 else "%.4f" % x


def pu(p, u):
    if isinstance(p, str) or isinstance(u, str):
        return "%s / %s" % (p, u)
    return "%s / %s" % (x4(p), x4(u))


def deep_json(v):
    return json.load(open(R / v / "deep" / "deep_analysis.json"))


def efind(d, run, K=25, ckpt=None):
    out = [x for x in d if x["run"] == run and x["K"] == K
           and (ckpt is None or x["ckpt"] == ckpt)]
    return out[0]


def cv(e, sp, K):
    return e["splits"][sp]["curve"][str(K)]


def load_npz(v, rel):
    z = np.load(R / v / rel, allow_pickle=True)
    return z["ids"].astype(str), z["scores"].astype(np.float64), z["labels"]


def deep_logp(v, run, ckpt, K, sp):
    rel = "deep/%s__%s__seeded42_K25_%s.npz" % (run, ckpt, sp)
    _ids, S, y = load_npz(v, rel)
    return stats_lib.mc_aggregate(S, K), y


# --------------------------------------------------------------------------
# per-pass calibration + logit-gap diagnostics (Table 3, Sec 4.3, Fig 2)
# --------------------------------------------------------------------------
def pass_calibration(v, run, ckpt):
    """ECE/AURC mean(std) over the 25 individual seeded draws."""
    out = {}
    for sp in ("val", "test"):
        _i, S, y = load_npz(
            v, "deep/%s__%s__seeded42_K25_%s.npz" % (run, ckpt, sp))
        eces, aurcs = [], []
        for k in range(S.shape[0]):
            probs = softmax(S[k])
            eces.append(dg.ece(probs, y)["ece"])
            aurcs.append(dg.aurc(probs, y)["aurc"])
        out[sp] = {"ece_mean": float(np.mean(eces)),
                   "ece_sd": float(np.std(eces, ddof=1)),
                   "aurc_mean": float(np.mean(aurcs)),
                   "aurc_sd": float(np.std(aurcs, ddof=1))}
    return out


def logit_gaps(v, run="v3_noSynth_s2", ckpt="best_model", sp="val"):
    _i, S, y = load_npz(
        v, "deep/%s__%s__seeded42_K25_%s.npz" % (run, ckpt, sp))
    K, N, C = S.shape
    # full range across 25 draws: per sample per class range, take class max
    rng = (S.max(0) - S.min(0)).max(1)            # (N,)
    # all draw-pair class-wise max gap, pooled over samples and 300 pairs
    pair_g = []
    for a in range(K):
        for b in range(a + 1, K):
            pair_g.append(np.abs(S[a] - S[b]).max(1))
    pair_g = np.concatenate(pair_g)
    return {"range_med": float(np.median(rng)),
            "range_p90": float(np.percentile(rng, 90)),
            "pair_med": float(np.median(pair_g)),
            "pair_p90": float(np.percentile(pair_g, 90)),
            "pair_max": float(pair_g.max())}


# --------------------------------------------------------------------------
# rho error-set overlap (Sec 5.8.1)
# --------------------------------------------------------------------------
def rho_overlaps(v):
    need = ["v5", "c1", "c1d", "c2", "v3", "c3"]
    deep_v3 = str(R / v /
                  "deep/v3_noSynth_s2__best_model__seeded42_K25_val.npz")
    deep_v5 = str(R / v /
                  "deep/v5_s2__swa_model__seeded42_K25_val.npz")
    recs = fp.load_all(v, need, deep_v3, deep_v5)
    _i, S, y = fp.align(recs, need, "val")
    # S: (M,N,C); deep entries are already mc-aggregated logp
    pred = S.argmax(2)
    wrong = {k: pred[j] != y for j, k in enumerate(need)}

    def rho(a, b):
        inter = np.sum(wrong[a] & wrong[b])
        union = np.sum(wrong[a] | wrong[b])
        return float(inter / union) if union else float("nan")
    return {"c1_vs_s2": rho("c1", "v5"),
            "c1d_vs_s2": rho("c1d", "v5"),
            "c2_vs_s2": rho("c2", "v5"),
            "v3_vs_s2": rho("v3", "v5"),
            "c3_vs_s2": rho("c3", "v5"),
            "c3_mac": float(evaluate(S[need.index("c3")], y, NC[v])
                            ["f1_macro"])}


# --------------------------------------------------------------------------
# fused per-sample log-probabilities (Final and P11), for the bootstraps
# --------------------------------------------------------------------------
def fused_logp(v, sp, which):
    if which == "final":
        need = ["v5", "c1", "c2"]
        deep_v3 = ""
        deep_v5 = str(R / v / (
            "deep/v5_s2__swa_model__seeded42_K25_%s.npz" % sp))
        recs = fp.load_all(v, need, deep_v3, deep_v5)
        _i, S, y = fp.align(recs, need, sp)
        taus = [1.0, 0.5, 1.0]
        w = np.array([2, 2, 2])
    else:  # p11 argmax: 10 classical (Table 6 order) + v3 MC-FPS K25
        t78 = pd.read_csv(R / v / "tables" / "table78_fusion.csv")
        row = t78[(t78.panel == "p11") &
                  (t78.selection == ("argmax" if which == "p11"
                                     else which))].iloc[0]
        need = ["c1", "c1b", "c1c", "c1d", "c2", "c2b", "c2d",
                "c3", "c3b", "c4", "v3"]
        deep_v3 = str(R / v / (
            "deep/v3_noSynth_s2__best_model__seeded42_K25_%s.npz" % sp))
        recs = fp.load_all(v, need, deep_v3, "")
        _i, S, y = fp.align(recs, need, sp)
        fam = ["dzd"] * 4 + ["fpfh"] * 3 + ["one"] * 3 + ["deep"]
        taus = [row.tau_dzd if f == "dzd" else row.tau_fpfh
                if f == "fpfh" else 1.0 for f in fam]
        w = np.array(ast.literal_eval(row.weights)) \
            if isinstance(row.weights, str) else np.array(row.weights)
    L = np.stack([fp.logsoftmax_np(S[j], taus[j])
                  for j in range(len(need))], 0)
    return np.tensordot(w.astype(np.float64), L, axes=(0, 0)), y


def boot_mac(logp_a, logp_b, y, nc):
    r = stats_lib.paired_bootstrap(logp_a, logp_b, y, nc,
                                   keys=["f1_macro"], B=500, seed=42)
    return r["f1_macro"]


def boot_accmac(logp_a, logp_b, y, nc):
    r = stats_lib.paired_bootstrap(logp_a, logp_b, y, nc,
                                   keys=["acc", "f1_macro"], B=500, seed=43)
    return r


def acc_halfwidth(logp, y, nc, B=1000):
    rng = np.random.default_rng(44)
    n = len(y)
    vals = np.empty(B)
    for b in range(B):
        ii = rng.integers(0, n, n)
        vals[b] = evaluate(logp[ii], y[ii], nc)["acc"]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    point = evaluate(logp, y, nc)["acc"]
    return float(max(point - lo, hi - point))


# --------------------------------------------------------------------------
# table renderers
# --------------------------------------------------------------------------
def t2_rows():
    """Fixed cached random single draw (seeded pass 0) + MC-FPS K sweep."""
    rows = []
    spec_single = [("v3, single draw", "v3_noSynth_s2", "best_model"),
                   ("v5 s1-best, single draw", "v5_s1", "best_model"),
                   ("v5 s2-SWA, single draw", "v5_s2", "swa_model")]
    for lab, run, ck in spec_single:
        def get(v, run, ck, sp, K):
            d = deep_json(v)
            e = [x for x in d if x["run"] == run and x["K"] == 25
                 and x["ckpt"] == ck][0]
            return cv(e, sp, K)
        p = [*get("paper", run, ck, "val", 1).values(),
             *get("paper", run, ck, "test", 1).values()]
        u = [*get("updated", run, ck, "val", 1).values(),
             *get("updated", run, ck, "test", 1).values()]
        rows.append((lab, [pu(p[i], u[i]) for i in range(6)]))
    # v3 K sweep 2..25
    for K in (2, 3, 5, 10, 25):
        p, u = [], []
        for v, tgt in (("paper", p), ("updated", u)):
            d = deep_json(v)
            e = [x for x in d if x["run"] == "v3_noSynth_s2"
                 and x["K"] == 25 and x["ckpt"] == "best_model"][0]
            tgt[:] = [*cv(e, "val", K).values(), *cv(e, "test", K).values()]
        rows.append(("v3 MC-FPS K = %d" % K,
                     [pu(p[i], u[i]) for i in range(6)]))
    # s1 best K25, s2 SWA K25
    for lab, run, ck in (("v5 s1-best MC-FPS K = 25", "v5_s1", "best_model"),
                         ("v5 s2-SWA MC-FPS K = 25", "v5_s2", "swa_model")):
        p, u = [], []
        for v, tgt in (("paper", p), ("updated", u)):
            d = deep_json(v)
            e = [x for x in d if x["run"] == run and x["K"] == 25
                 and x["ckpt"] == ck][0]
            tgt[:] = [*cv(e, "val", 25).values(), *cv(e, "test", 25).values()]
        rows.append((lab, [pu(p[i], u[i]) for i in range(6)]))
    return rows


def t3_rows(stats):
    rows = []
    for ckpt_lab, run, ck in (("v3", "v3_noSynth_s2", "best_model"),
                              ("s2", "v5_s2", "swa_model")):
        for sp in ("val", "test"):
            cells = []
            for v in VERS:
                pc = stats[v]["calib"]["%s_%s" % (run, ck)][sp]
                d = deep_json(v)
                e = [x for x in d if x["run"] == run and x["K"] == 25
                     and x["ckpt"] == ck][0]
                g = e["splits"][sp]["diag"]
                cells.append(("%s (%s)" % (x4(pc["ece_mean"]),
                                           x4(pc["ece_sd"])),
                              x4(g["ece15"]), x4(pc["aurc_mean"]),
                              x4(g["aurc"]), x4(g["bald_mean"])))
            # interleave paper/updated per metric
            rows.append((ckpt_lab, sp,
                          [pu(cells[0][i], cells[1][i]) for i in range(5)]))
    return rows


def t4_rows():
    rows = []
    prot = [("random (single draw, mean)", "random"),
            ("D0 (point 0)", "d0"), ("Ds (fixed generator)", "ds"),
            ("K25 (MC-FPS)", "k25")]
    t4 = {v: pd.read_csv(R / v / "tables" / "table4_modes.csv")
          .query("run=='v3_noSynth_s2'") for v in VERS}
    for lab, key in prot:
        p, u = [], []
        for v, tgt in (("paper", p), ("updated", u)):
            r = t4[v]
            spv = r[r.split == "val"].iloc[0]
            spt = r[r.split == "test"].iloc[0]
            if key == "random":
                tgt[:] = [spv.random_mean_acc, spv.random_mean_mac,
                          spt.random_mean_acc, spt.random_mean_mac]
            elif key == "d0":
                tgt[:] = [spv.d0_acc, spv.d0_mac, spt.d0_acc, spt.d0_mac]
            elif key == "ds":
                tgt[:] = [spv.ds_acc, spv.ds_mac, spt.ds_acc, spt.ds_mac]
            else:
                tgt[:] = [spv.k25_acc, spv.k25_mac, spt.k25_acc, spt.k25_mac]
        rows.append((lab, [pu(p[i], u[i]) for i in range(4)]))
    return rows


def t6_rows():
    mem = [("c1", "c1"), ("c1b", "c1b"), ("c1c", "c1c"), ("c1d", "c1d"),
           ("c2", "c2"), ("c2b", "c2b"), ("c2d", "c2d"),
           ("c3", "c3_mix"), ("c3b", "c3b_mix"), ("c4", "c4")]
    labels = {
        "c1": "c1 363, 1-NN+gate", "c1b": "c1b 363, inv-dist",
        "c1c": "c1c 363, block-z", "c1d": "c1d 363, RBF-SVM",
        "c2": "c2 612, RBF-SVM", "c2b": "c2b 612, ExtraTrees",
        "c2d": "c2d 612, cosine kNN",
        "c3": "c3 171, SimpleNN 8-view", "c3b": "c3b 171, RBF-SVM",
        "c4": "c4 975, RBF-SVM"}
    dfs = {v: pd.read_csv(R / v / "tables" / "table6_classical.csv")
           .set_index("member") for v in VERS}
    rows = []
    for _m, key in mem:
        p, u = [], []
        for v, tgt in (("paper", p), ("updated", u)):
            df = dfs[v]
            if key == "c3_mix":
                vr = df.loc["c3"]
                tr = df.loc["c3_fullrefit"]
            elif key == "c3b_mix":
                vr = df.loc["c3b"]
                tr = df.loc["c3b_fullrefit"]
            else:
                vr = tr = df.loc[key]
            tgt[:] = [vr.val_acc, vr.val_bal_acc, vr.val_f1_macro,
                      tr.test_acc, tr.test_bal_acc, tr.test_f1_macro]
        rows.append((labels[_m], [pu(p[i], u[i]) for i in range(6)]))
    return rows


def t78_rows(panel_a, sel_a, panel_b=None, sel_b=None):
    t = {v: pd.read_csv(R / v / "tables" / "table78_fusion.csv") for v in VERS}

    def grab(v, panel, sel):
        r = t[v].query("panel==@panel and selection==@sel").iloc[0]
        return [r.val_acc, r.val_bal, r.val_mac,
                r.test_acc, r.test_bal, r.test_mac]
    out = []
    for lab, panel, sel in ((None, panel_a, sel_a),
                            (None, panel_b, sel_b)):
        if panel is None:
            continue
        p = grab("paper", panel, sel)
        u = grab("updated", panel, sel)
        out.append([pu(p[i], u[i]) for i in range(6)])
    return out


def t9_rows():
    run_order = [("Full recipe (v5 s2)", "v5_s2_swa"),
                 ("-RareSynth", "v3_noSynth_s2"),
                 ("Focal gamma=2", "abl_focalg2_s2"),
                 ("-PointCutMix", "abl_noCutMix_s2"),
                 ("Class-Bal beta=.9999", "abl_cbbeta_s2")]
    t = {v: pd.read_csv(R / v / "tables" / "table9_ablation.csv")
         for v in VERS}
    blocks = {"standalone": [], "frozen_fusion": []}
    for lab, run in run_order:
        for block in blocks:
            p, u = [], []
            for v, tgt in (("paper", p), ("updated", u)):
                if run == "v5_s2_swa":
                    # full recipe standalone: v5_s2 SWA MC-FPS K25
                    d = deep_json(v)
                    e = [x for x in d if x["run"] == "v5_s2" and x["K"] == 25
                         and x["ckpt"] == "swa_model"][0]
                    ev, et = e["splits"]["val"], e["splits"]["test"]
                    tgt[:] = [ev["curve"]["25"]["acc"],
                              ev["curve"]["25"]["bal"],
                              ev["curve"]["25"]["mac"],
                              et["curve"]["25"]["acc"],
                              et["curve"]["25"]["bal"],
                              et["curve"]["25"]["mac"]]
                else:
                    rv = t[v].query("block==@block and run==@run and split=='val'").iloc[0]
                    rt = t[v].query("block==@block and run==@run and split=='test'").iloc[0]
                    tgt[:] = [rv.acc, rv.bal_acc, rv.f1_macro,
                              rt.acc, rt.bal_acc, rt.f1_macro]
            blocks[block].append((lab, [pu(p[i], u[i]) for i in range(6)]))
    # full-recipe fusion row = final panel (table78/fusion_summary)
    f = {v: json.load(open(R / v / "fusion" / "fusion_summary.json"))
         for v in VERS}
    p = [f["paper"]["final"]["val"][k] for k in
         ("acc", "bal_acc", "f1_macro")] + [
        f["paper"]["final"]["test"][k] for k in
        ("acc", "bal_acc", "f1_macro")]
    u = [f["updated"]["final"]["val"][k] for k in
         ("acc", "bal_acc", "f1_macro")] + [
        f["updated"]["final"]["test"][k] for k in
        ("acc", "bal_acc", "f1_macro")]
    blocks["frozen_fusion"][0] = ("Full recipe (v5 s2)",
                                 [pu(p[i], u[i]) for i in range(6)])
    return blocks


def tta_rows():
    t = {v: pd.read_csv(R / v / "tables" / "table_tta.csv") for v in VERS}
    out = []
    for method, lab in (("identity", "single-seed identity"),
                        ("tta25_meanlogit", "rotational TTA K=25")):
        p, u = [], []
        for v, tgt in (("paper", p), ("updated", u)):
            rv = t[v].query("split=='val' and method==@method").iloc[0]
            rt = t[v].query("split=='test' and method==@method").iloc[0]
            tgt[:] = [rv.acc, rv.f1_macro, rt.acc, rt.f1_macro]
        out.append((lab, [pu(p[i], u[i]) for i in range(4)]))
    return out


def t17_rows():
    t = {v: pd.read_csv(R / v / "tables" / "table17_groupsize.csv")
         for v in VERS}
    out = []
    for g in ("rare", "small", "medium", "large"):
        p, u = [], []
        for v, tgt in (("paper", p), ("updated", u)):
            for cond in ("v3_single", "v3_K25", "final"):
                r = t[v].query("group==@g and condition==@cond").iloc[0]
                tgt.append(float(r.acc))
            nt = t[v].query("group==@g and condition=='final'").iloc[0].n_test
            tgt.append(int(nt))
        cells = [pu(p[i], u[i]) for i in range(3)] + ["%d / %d" % (p[3], u[3])]
        out.append((g, cells))
    return out


# Table 8 track-report integers (A5: bar-chart values, verbatim, paper only).
A5 = [
    ("Machine learning", None),
    ("Tatsuma (FPFH-SVM)", (88, 85, 88, 90, 88)),
    ("Tehrani v2*", (17, 3, 15, 22, 17)),
    ("Training-free", None),
    ("Peng (8-view images, GoogLeNet kNN)", (91, 89, 91, 91, 91)),
    ("Deep learning", None),
    ("Barisin v1", (92, 80, 91, 92, 92)),
    ("Barisin v2", (93, 85, 92, 93, 93)),
    ("3D-SURFER (3DZD-SVM)", (84, 76, 83, 84, 84)),
    ("Guerra v1", (80, 71, 79, 79, 79)),
    ("Guerra v2", (80, 75, 79, 79, 79)),
    ("Guerra v3", (80, 81, 80, 80, 80)),
    ("Guerra v4 (8-view ensemble)", (87, 67, 79, 87, 80)),
    ("3D-PROSPER v1", (80, 68, 78, 79, 80)),
    ("3D-PROSPER v2", (80, 80, 67, 80, 80)),
    ("RISurProtNet*", (31, 8, 26, 49, 32)),
    ("Tehrani v1", (72, 43, 71, 75, 72)),
    ("ProtoCluster", (71, 42, 67, 72, 71)),
]
LEGACY_NAME = {
    "tatsuma": "Tatsuma (FPFH-SVM)", "tehrani_v2": "Tehrani v2*",
    "peng": "Peng (GoogLeNet kNN)", "barisin_geom": "Barisin v1 (geom)",
    "barisin_geom_prop": "Barisin v2 (geom+prop)",
    "kagaya_3D-Surfer": "3D-SURFER (Kagaya)", "guerra_v1": "Guerra v1",
    "guerra_v2": "Guerra v2", "guerra_v3": "Guerra v3",
    "guerra_v4": "Guerra v4", "he_v1": "3D-PROSPER v1",
    "he_v2": "3D-PROSPER v2", "li_RISurProNet": "RISurProtNet*",
    "tehrani_v1": "Tehrani v1", "yang_ProtoCluster": "ProtoCluster",
    "barisin_run1_updated": "Barisin updated run1 (6 ch, 250 ep)",
    "barisin_run2_updated": "Barisin updated run2 (8 ch, 250 ep)"}
LEGACY_ORDER = ["tatsuma", "tehrani_v2", "peng", "barisin_geom",
                "barisin_geom_prop", "kagaya_3D-Surfer", "guerra_v1",
                "guerra_v2", "guerra_v3", "guerra_v4", "he_v1", "he_v2",
                "li_RISurProNet", "tehrani_v1", "yang_ProtoCluster",
                "barisin_run1_updated", "barisin_run2_updated"]


def t10_legacy_rows():
    d = {}
    for v in VERS:
        df = pd.read_csv(R / v / "tables" / "table10_crossmethod.csv")
        d[v] = {r.method: r for r in df.itertuples()}
    rows = []
    for m in LEGACY_ORDER:
        cells = []
        for col in ("acc", "bal", "mac"):
            pv = getattr(d["paper"].get(m), col, None) if m in d["paper"] else None
            uv = getattr(d["updated"].get(m), col, None) if m in d["updated"] else None
            pv = None if pv is None or (isinstance(pv, float) and np.isnan(pv)) else pv
            uv = None if uv is None or (isinstance(uv, float) and np.isnan(uv)) else uv
            cells.append(pu(pv, uv) if (pv is not None or uv is not None) else "–")
        rows.append((LEGACY_NAME[m], cells))
    return rows


def t10_ours_rows():
    """cached fixed single draw (seeded pass0), MC-FPS K25, P11, Final."""
    out = []
    # single and K25 from v3 seeded curve
    p_s, u_s, p_k, u_k = [], [], [], []
    for v, ts, tk in (("paper", p_s, p_k), ("updated", u_s, u_k)):
        d = deep_json(v)
        e = [x for x in d if x["run"] == "v3_noSynth_s2" and x["K"] == 25
             and x["ckpt"] == "best_model"][0]
        for sp, tgt in (("val", None),):
            pass
        ts[:] = [*cv(e, "test", 1).values()]
        tk[:] = [*cv(e, "test", 25).values()]
    out.append(("RIConv++ v3, cached single draw",
                [pu(p_s[i], u_s[i]) for i in range(3)]))
    out.append(("RIConv++ v3, MC-FPS K=25",
                [pu(p_k[i], u_k[i]) for i in range(3)]))
    # P11 (val-argmax) and Final from table78 / fusion_summary
    t78 = {v: pd.read_csv(R / v / "tables" / "table78_fusion.csv")
           for v in VERS}
    for lab, getter in (
        ("P11 (11-member, val argmax)",
         lambda v: t78[v].query("panel=='p11' and selection=='argmax'").iloc[0]),
        ("Final panel (frozen)",
         lambda v: pd.read_csv(R / v / "tables" / "table78_fusion.csv")
         .query("panel=='final'").iloc[0])):
        p, u = [], []
        for v, tgt in (("paper", p), ("updated", u)):
            r = getter(v)
            tgt[:] = [r.test_acc, r.test_bal, r.test_mac]
        out.append((lab, [pu(p[i], u[i]) for i in range(3)]))
    return out


# --------------------------------------------------------------------------
# prose statistics
# --------------------------------------------------------------------------
def build_stats():
    stats = {}
    for v in VERS:
        d = deep_json(v)
        v3 = [x for x in d if x["run"] == "v3_noSynth_s2" and x["K"] == 25
              and x["ckpt"] == "best_model"][0]
        s2 = [x for x in d if x["run"] == "v5_s2" and x["K"] == 25
              and x["ckpt"] == "swa_model"][0]
        stats[v] = {
            "flip_val": v3["splits"]["val"]["diag"]["flip_rate"],
            "flip_test": v3["splits"]["test"]["diag"]["flip_rate"],
            "single_mac_val": {
                "mean": v3["splits"]["val"]["single_mean_mac"],
                "sd": v3["splits"]["val"]["single_sd_mac"],
                "min": v3["splits"]["val"]["single_min_mac"],
                "max": v3["splits"]["val"]["single_max_mac"]},
            "v3_val_k25": cv(v3, "val", 25),
            "v3_test_k1": cv(v3, "test", 1),
            "v3_test_k25": cv(v3, "test", 25),
            "s2_test_k25": cv(s2, "test", 25),
            "s2_val_k25": cv(s2, "val", 25),
            "gaps_val": logit_gaps(v),
            "calib": {
                "v3_noSynth_s2_best_model":
                    pass_calibration(v, "v3_noSynth_s2", "best_model"),
                "v5_s2_swa_model":
                    pass_calibration(v, "v5_s2", "swa_model")},
        }
        # temperature top-30 in the 7-member scan
        scan = pd.read_csv(R / v / "fusion" / "p7_scan_top2000.csv")
        stats[v]["tau_fpfh_top30_half"] = int(
            (scan.head(30).tau_fpfh == 0.5).sum())
        stats[v]["rho"] = rho_overlaps(v)

    # paired bootstraps
    bs = {}
    for v in VERS:
        nc = NC[v]
        # K=10 vs K=25 v3 val accuracy
        _i, Sv, yv = load_npz(v, "deep/v3_noSynth_s2__best_model__seeded42_K25_val.npz")
        l10 = stats_lib.mc_aggregate(Sv, 10)
        l25 = stats_lib.mc_aggregate(Sv, 25)
        bs[v] = {}
        bs[v]["k10_vs_k25_val_acc"] = stats_lib.paired_bootstrap(
            l25, l10, yv, nc, keys=["acc"], B=500, seed=42)["acc"]
        # final vs P11-argmax, validation macro-F1 (Sec 5.10)
        fv, yfv = fused_logp(v, "val", "final")
        pv_, ypv = fused_logp(v, "val", "p11")
        bs[v]["final_vs_p11_val_mac"] = boot_mac(fv, pv_, yfv, nc)
        # fusion gain final vs v3 K25 on test (acc + mac)
        ft, yft = fused_logp(v, "test", "final")
        v3t, yv3t = deep_logp(v, "v3_noSynth_s2", "best_model", 25, "test")
        bs[v]["final_vs_v3k25_test"] = boot_accmac(ft, v3t, yft, nc)
        bs[v]["final_test_acc_halfwidth"] = acc_halfwidth(ft, yft, nc)
    for v in VERS:
        stats[v]["bootstrap"] = bs[v]
    return stats


# --------------------------------------------------------------------------
# markdown assembly
# --------------------------------------------------------------------------
HDR6 = "| Configuration | Val Acc | Val Bal | Val Mac | Test Acc | Test Bal | Test Mac |"
SEP6 = "|---|---|---|---|---|---|---|"


def emit(stats):
    L = ["# Dual-version manuscript tables (paper / updated)",
         "",
         "Each numeric cell is `paper-version / updated-version`; paper = 97 "
         "classes, n_test=2311; updated = 99 classes, n_test=2308. All values "
         "are generated by `make_manuscript_tables_dual.py` from the caches "
         "under experiments/results/<version>/.", ""]

    L += ["## Table 3", "", HDR6, SEP6]
    for lab, c in t2_rows():
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("")

    L += ["## Table 4",
          "",
          "| Ckpt | Split | ECE single mean(sd) | ECE K25 | AURC single | AURC K25 | MI nats |",
          "|---|---|---|---|---|---|---|"]
    for ck, sp, c in t3_rows(stats):
        L.append("| %s | %s | %s |" % (ck, sp, " | ".join(c)))
    L.append("")

    L += ["## Table 5", "",
          "| Protocol | Val Acc | Val Mac | Test Acc | Test Mac |",
          "|---|---|---|---|---|"]
    for lab, c in t4_rows():
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("")

    L += ["## Supplementary Table S1", "", HDR6, SEP6]
    for lab, c in t6_rows():
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("")

    L += ["## Supplementary Table S2", "", HDR6, SEP6]
    for lab, c in zip(("P7 (7-member C-only, one-SE)",
                       "P11 (11-member, val argmax)", "Final (frozen)"),
                      t78_rows("p7", "onse", "p11", "argmax") +
                      t78_rows("final", "fixed")):
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("")

    L += ["## Supplementary Table S3", "", HDR6, SEP6]
    for lab, c in zip(("P10 (argmax, 10 members)", "P7 (one-SE, 7 members)"),
                      t78_rows("p10", "argmax") + t78_rows("p7", "onse")):
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("")

    blocks = t9_rows()
    L += ["## Supplementary Table S7", "", HDR6, SEP6,
          "| Isolated deep model (MC-FPS K=25) | | | | | | |"]
    for lab, c in blocks["standalone"]:
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("| Substituted into the frozen fusion | | | | | | |")
    for lab, c in blocks["frozen_fusion"]:
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("")

    L += ["## Table 7 (rotational TTA ablation)", "",
          "| Variant | Val Acc | Val Mac | Test Acc | Test Mac |",
          "|---|---|---|---|---|"]
    for lab, c in tta_rows():
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("")

    L += ["## Supplementary Table S8 (accuracy by class size)", "",
          "| Group | v3 single acc | v3 K25 acc | Final acc | n_test |",
          "|---|---|---|---|---|"]
    for g, c in t17_rows():
        L.append("| %s | %s |" % (g, " | ".join(c)))
    L.append("")

    # Table 8
    L += ["## Table 8", "",
          "### Block A — track-report values (paper version, n=2311), read "
          "from the organisers' bar charts (Figs. 7–11 of [1]); integer "
          "percentage points, approximate (decision A5; not regenerable from "
          "any released per-sample cache).",
          "",
          "| System | Acc | Bal | F1 | Prec | Rec |",
          "|---|---|---|---|---|---|"]
    for lab, vals in A5:
        if vals is None:
            L.append("| _%s_ |  |  |  |  |  |" % lab)
        else:
            L.append("| %s | %s |" % (lab, " | ".join("%d" % z for z in vals)))
    L += ["",
          "### Block B — recomputed from the released submission files on the "
          "fixed class grid (paper / updated). Barisin updated run1/run2 are "
          "new 250-epoch trainings on the updated profile (n=2308).",
          "",
          "| System | Acc | Bal | Macro-F1 |",
          "|---|---|---|---|"]
    for lab, c in t10_legacy_rows():
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L += ["",
          "### Block C — this work (recomputed, paper / updated).",
          "",
          "| System | Acc | Bal | Macro-F1 |",
          "|---|---|---|---|"]
    for lab, c in t10_ours_rows():
        L.append("| %s | %s |" % (lab, " | ".join(c)))
    L.append("")
    return "\n".join(L) + "\n"


def main():
    stats = build_stats()
    (R / "manuscript_stats.json").write_text(json.dumps(stats, indent=2))
    md = emit(stats)
    (R / "manuscript_tables_dual.md").write_text(md)
    for v in VERS:
        s = stats[v]
        print("==", v, "==")
        print(" flip val/test %.4f/%.4f  single mac %.3f+/-%.3f [%.3f,%.3f]"
              % (s["flip_val"], s["flip_test"],
                 s["single_mac_val"]["mean"], s["single_mac_val"]["sd"],
                 s["single_mac_val"]["min"], s["single_mac_val"]["max"]))
        g = s["gaps_val"]
        print(" range med/p90 %.2f/%.2f  pair med/p90/max %.2f/%.2f/%.2f"
              % (g["range_med"], g["range_p90"], g["pair_med"],
                 g["pair_p90"], g["pair_max"]))
        print(" rho", {k: round(x, 3) for k, x in s["rho"].items()})
        b = s["bootstrap"]
        print(" K10vK25 d%.4f P%.3f CI[%.4f,%.4f]" % (
            b["k10_vs_k25_val_acc"]["delta"], b["k10_vs_k25_val_acc"]["p"],
            *b["k10_vs_k25_val_acc"]["ci95"]))
        print(" final-v-p11 valmac d%.4f P%.3f CI[%.4f,%.4f]" % (
            b["final_vs_p11_val_mac"]["delta"],
            b["final_vs_p11_val_mac"]["p"],
            *b["final_vs_p11_val_mac"]["ci95"]))
        bg = b["final_vs_v3k25_test"]
        print(" final-v-v3 test acc d%.4f CI[%.4f,%.4f] mac d%.4f "
              "CI[%.4f,%.4f] halfwidth %.4f" % (
                  bg["acc"]["delta"], *bg["acc"]["ci95"],
                  bg["f1_macro"]["delta"], *bg["f1_macro"]["ci95"],
                  b["final_test_acc_halfwidth"]))
        print(" tau_fpfh=.5 in p7 top30:", s["tau_fpfh_top30_half"])
    print("wrote", R / "manuscript_tables_dual.md",
          "and manuscript_stats.json")


if __name__ == "__main__":
    main()
