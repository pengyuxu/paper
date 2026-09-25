"""Route A: calibration and uncertainty analysis (paper Section V-F).

Uses the stored 25-draw MC-FPS log-probability caches (no retraining):
  results_final/cache/mcfps_{val,test}_logp.npy         (v3)
  results_final/cache/mcfps_v5s2_{val,test}_logp.npy    (v5 s2 SWA)

For each cache compares a single FPS-seed draw against K=25 averaging on:
  - ECE (15 equal-width bins) and reliability-diagram bin statistics
  - predictive entropy and mutual information (BALD-type epistemic signal):
        MI = H(E[p_k]) - E[H(p_k)]
  - selective prediction: risk-coverage curves and AURC
Writes:
  results_final/calibration_summary.json
  results_final/calibration_curves.npz
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

BASE = Path(__file__).resolve().parent.parent
CACHE = BASE / "results_final" / "cache"
OUT = BASE / "results_final"
NBINS = 15
SEED = 42


def softmax(lp):
    lp = lp.astype(np.float64)
    lp -= lp.max(-1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(-1, keepdims=True)


def ece(probs, y, n_bins=NBINS):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = pred == y
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    rows = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1]) if i else \
            (conf >= bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            gap = m.mean() * abs(acc[m].mean() - conf[m].mean())
            e += gap
            rows.append((0.5 * (bins[i] + bins[i + 1]),
                         acc[m].mean(), conf[m].mean(), int(m.sum())))
    return float(e), rows


def entropy(p, axis=-1):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis)


def aurc(probs, y):
    conf = probs.max(1)
    order = np.argsort(-conf)
    risk = (np.cumsum((probs.argmax(1) != y)[order])
            / np.arange(1, len(y) + 1))
    cov = np.arange(1, len(y) + 1) / len(y)
    return float(np.trapz(risk, cov)), cov, risk


def analyze(lp, y):
    probs = softmax(lp)                      # (K, N, C)
    mean_p = probs.mean(0)                   # probability mixture, K=25

    rng = np.random.default_rng(SEED)
    acc_e, mac_e, aurc_e = [], [], []
    rel_rows = None
    for k in range(probs.shape[0]):
        e, rows = ece(probs[k], y)
        acc_e.append(e)
        mac_e.append(f1_score(y, probs[k].argmax(1), average="macro",
                              zero_division=0))
        a, _, _ = aurc(probs[k], y)
        aurc_e.append(a)
        if k == 0:
            rel_rows = rows

    e25, rel25 = ece(mean_p, y)
    a25, cov, risk = aurc(mean_p, y)

    h_mean = entropy(mean_p).mean()                 # predictive entropy K25
    h_single = entropy(probs).mean()                # aleatoric avg
    mi = float((entropy(mean_p) - entropy(probs).mean(0)).mean())

    return {
        "single": {
            "ece_mean": float(np.mean(acc_e)), "ece_std": float(np.std(acc_e)),
            "macF1_mean": float(np.mean(mac_e)),
            "acc_mean": float(np.mean([accuracy_score(y, probs[k].argmax(1))
                                      for k in range(25)])),
            "aurc_mean": float(np.mean(aurc_e)),
            "entropy": float(h_single),
        },
        "k25": {
            "ece": e25, "macF1": f1_score(y, mean_p.argmax(1), average="macro",
                                         zero_division=0),
            "acc": accuracy_score(y, mean_p.argmax(1)),
            "aurc": a25, "entropy": float(h_mean), "mi": mi,
        },
        "rel_single": rel_rows, "rel_k25": rel25,
        "rc_cov": cov.tolist(),
        "rc_k25_risk": risk.tolist(),
    }


def main():
    yv = np.load(CACHE / "v5_s2_swa_val.npz")["y"]
    yt = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values

    jobs = {
        "v3_val":  ("mcfps_val_logp.npy", yv),
        "v3_test": ("mcfps_test_logp.npy", yt),
        "s2_val":  ("mcfps_v5s2_val_logp.npy", yv),
        "s2_test": ("mcfps_v5s2_test_logp.npy", yt),
    }
    summary, curves = {}, {}
    for name, (f, y) in jobs.items():
        r = analyze(np.load(CACHE / f), y)
        # per-draw mean risk-coverage curve (confidence quantile averaged)
        probs = softmax(np.load(CACHE / f))
        risks = []
        for k in range(25):
            _, c_, rr = aurc(probs[k], y)
            risks.append(rr)
        r["rc_single_risk"] = np.mean(risks, 0).tolist()
        curves[name] = {"cov": r.pop("rc_cov"),
                        "single": r.pop("rc_single_risk"),
                        "k25": r.pop("rc_k25_risk")}
        summary[name] = r
        s, q = r["single"], r["k25"]
        print("[%s]" % name)
        print("  single : ECE %.4f±%.4f  acc %.4f  macF1 %.4f  AURC %.4f  H %.3f"
              % (s["ece_mean"], s["ece_std"], s["acc_mean"],
                 s["macF1_mean"], s["aurc_mean"], s["entropy"]))
        print("  K=25   : ECE %.4f     acc %.4f  macF1 %.4f  AURC %.4f  H %.3f"
              % (q["ece"], q["acc"], q["macF1"], q["aurc"], q["entropy"]))
        print("           MI (BALD) %.4f" % q["mi"])

    (OUT / "calibration_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(OUT / "calibration_curves.npz",
             **{k + "_" + sk: np.array(v[sk])
                for k, v in curves.items()
                for sk in ("cov", "single", "k25")})
    print("saved", OUT / "calibration_summary.json")


if __name__ == "__main__":
    main()
