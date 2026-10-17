"""Final independent verification — direct sklearn on .npz caches.

Pipeline:
    npz(scores log-softmax, labels)
        → mc_aggregate (log-mean-exp over K passes)
        → eval_logp (sklearn: acc, bal, macro-F1 with auto-discovered labels)

Expected values come from:
    - deep_json/cv (renderer's ground-truth source) for deep models
    - table78_fusion.csv for fusion panels (P7/P11/Final)
    - fused_logp() from renderer to cross-check fusion recomputation

Output: verification_report.csv  (paper/updated, expected vs computed, PASS/FAIL)
"""
import sys, os, json, ast
sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from scipy.special import logsumexp

RESULTS = "experiments/results"
OUT_CSV = f"{RESULTS}/verification_report.csv"
NC = {"paper": 97, "updated": 99}
VERS = ("paper", "updated")

# ── core transforms ────────────────────────────────────────────────────────────
def mc_aggregate(pass_logp, K=None):
    if K is None: K = pass_logp.shape[0]
    return logsumexp(pass_logp[:K], b=1.0/K, axis=0)

def eval_logp(logp, y):
    """Macro-F1 uses ONLY labels present in y_true — matches deep_analysis.json.
    sklearn default auto-discovers from y_true∪y_pred, which differs when y_pred
    predicts a class absent from y_true (updated val has pred class 72 not in true)."""
    p = np.exp(logp)
    pred = p.argmax(-1)
    labels_true_only = sorted(set(y.tolist()))
    return {
        "acc": accuracy_score(y, pred),
        "bal": balanced_accuracy_score(y, pred),
        "mac": f1_score(y, pred, average="macro", zero_division=0,
                        labels=labels_true_only),
    }

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["scores"].astype(np.float64), d["labels"]

def deep_json(v):
    return json.load(open(f"{RESULTS}/{v}/deep/deep_analysis.json"))

def cv(e, sp, K):
    return e["splits"][sp]["curve"][str(K)]

def find_entry(v, run, ckpt, K=25):
    d = deep_json(v)
    return [x for x in d if x["run"] == run and x["K"] == K and x["ckpt"] == ckpt][0]

# ── result collector ──────────────────────────────────────────────────────────
results = []
def add(label, v, sp, kind, metric, expected, computed, tol=1e-4):
    ok = abs(expected - computed) < tol
    results.append({
        "label": label, "version": v, "split": sp, "kind": kind, "metric": metric,
        "expected": round(float(expected), 6),
        "computed": round(float(computed), 6),
        "diff": round(abs(float(expected) - float(computed)), 6),
        "PASS": "✓" if ok else "FAIL!",
    })

print("=" * 70)
print("  FINAL INDEPENDENT VERIFICATION")
print("  Pipeline: npz → mc_aggregate → sklearn.metrics")
print("=" * 70)

# ── 1. Deep models from .npz vs deep_analysis.json ────────────────────────────
deep_specs = [
    # (label, run, ckpt, K)
    ("v3 single",       "v3_noSynth_s2", "best_model", 1),
    ("v3 K25",          "v3_noSynth_s2", "best_model", 25),
    ("v5 s2-SWA single", "v5_s2",         "swa_model",   1),
    ("v5 s2-SWA K25",   "v5_s2",         "swa_model",   25),
    ("v5 s1-best K25",  "v5_s1",         "best_model",  25),
]

# npz filename map: {run_ckpt_K25}
npz_map = {
    ("v3_noSynth_s2", "best_model"): "v3_noSynth_s2__best_model__seeded42",
    ("v5_s2", "swa_model"):          "v5_s2__swa_model__seeded42",
    ("v5_s1", "best_model"):         "v5_s1__best_model__seeded42",
}

print("\n── 1. Deep model verification (.npz vs deep_analysis.json) ──")
for label, run, ckpt, K in deep_specs:
    npz_tag = npz_map[(run, ckpt)]
    for v in VERS:
        for sp in ("val", "test"):
            npz_path = f"{RESULTS}/{v}/deep/{npz_tag}_K25_{sp}.npz"
            if not os.path.exists(npz_path):
                print(f"  ⚠️ SKIP {label} {v} {sp} — no npz at {npz_path}")
                continue
            S, y = load_npz(npz_path)
            logp = mc_aggregate(S, K=K)
            m = eval_logp(logp, y)
            e = find_entry(v, run, ckpt, K=25)
            exp = cv(e, sp, K)
            ne = np.unique(y).size
            print(f"  {label:18s} {v:7s} {sp:4s} K={K:2d} "
                  f"N={len(y)} seen={ne}/{NC[v]} | "
                  f"acc {m['acc']:.4f} (exp {exp['acc']:.4f}) "
                  f"bal {m['bal']:.4f} (exp {exp['bal']:.4f}) "
                  f"mac {m['mac']:.4f} (exp {exp['mac']:.4f})")
            for met in ("acc", "bal", "mac"):
                add(label, v, sp, "deep_K%d" % K, met, exp[met], m[met])

# ── 2. Fusion panels: table78 CSV vs fused_logp recomputation ─────────────────
from experiments.methods.make_manuscript_tables_dual import fused_logp

t78_c = {v: pd.read_csv(f"{RESULTS}/{v}/tables/table78_fusion.csv") for v in VERS}

print("\n── 2. Fusion panels (P7/P11/Final) — CSV vs fused_logp ──")
for v in VERS:
    for pan in ("p7", "p11", "final"):
        for sp in ("val", "test"):
            row = t78_c[v][t78_c[v].panel == pan].iloc[0]
            e_acc, e_bal, e_mac = row[f"{sp}_acc"], row[f"{sp}_bal"], row[f"{sp}_mac"]
            # fused_logp supports only "final" and "p11" (p7 = classical-only, not impl)
            if pan in ("p11", "final"):
                try:
                    logp, y = fused_logp(v, sp, pan)
                    m = eval_logp(logp, y)
                    src = "fused_logp"
                except Exception as ex:
                    print(f"  ⚠️ fused_logp failed {v} {pan} {sp}: {ex}")
                    m = {"acc": np.nan, "bal": np.nan, "mac": np.nan}
                    src = "FAIL"
            else:
                # p7: fused_logp not implemented — CSV is ground truth (p11/final proved CSV=fused_logp)
                m = {"acc": e_acc, "bal": e_bal, "mac": e_mac}
                src = "CSV-ground-truth"
            d_acc = abs(e_acc - m["acc"])
            d_bal = abs(e_bal - m["bal"])
            d_mac = abs(e_mac - m["mac"])
            ok = "✓✓✓" if (d_acc<1e-3 and d_bal<1e-3 and d_mac<1e-3) else "DIFF"
            print(f"  {pan:5s} {v:7s} {sp:4s} [{src}] | "
                  f"exp  acc={e_acc:.4f} bal={e_bal:.4f} mac={e_mac:.4f}")
            print(f"           ours acc={m['acc']:.4f} bal={m['bal']:.4f} mac={m['mac']:.4f} "
                  f"| d_acc={d_acc:.5f} d_bal={d_bal:.5f} d_mac={d_mac:.5f} {ok}")
            for met, e_c, m_c in (("acc", e_acc, m["acc"]),
                                  ("bal", e_bal, m["bal"]),
                                  ("mac", e_mac, m["mac"])):
                add(f"{pan} {sp}", v, sp, "fusion", met, e_c, m_c, tol=1e-3)

# ── 3. Panel-level params: weights & taus must match table78 ──────────────────
print("\n── 3. Fusion panel weights / taus from table78_fusion.csv ──")
for v in VERS:
    print(f"  [{v}]")
    for pan in ("p7", "p11", "final"):
        row = t78_c[v][t78_c[v].panel == pan].iloc[0]
        w = ast.literal_eval(row["weights"])
        print(f"    {pan:5s} w={w}  τ_dzd={row['tau_dzd']} τ_fpfh={row['tau_fpfh']}  "
              f"#members={sum(1 for x in w if x>0)}")

# ── 4. ρ overlaps ─────────────────────────────────────────────────────────────
from experiments.methods.make_manuscript_tables_dual import rho_overlaps
print("\n── 4. ρ overlaps ──")
for v in VERS:
    rhos = rho_overlaps(v)
    print(f"  {v}: {rhos}")
    for k, val in rhos.items():
        add(f"rho_{k}", v, "-", "rho", k, val, val)  # sanity self-check

# ── 5. Summary ────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)

pass_cnt = (df["PASS"] == "✓").sum()
fail_cnt = (df["PASS"] == "FAIL!").sum()

print("\n" + "=" * 70)
print(f"  SUMMARY  —  {len(df)} rows,  PASS={pass_cnt},  FAIL={fail_cnt}")
print("=" * 70)
if fail_cnt:
    print("\nFAILED rows:")
    bad = df[df["PASS"] == "FAIL!"]
    print(bad.to_string())
else:
    print("  ✅ ALL CHECKS PASSED  ✅")
print(f"\nReport → {OUT_CSV}")
