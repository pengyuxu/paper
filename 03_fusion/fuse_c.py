"""C-layer fusion: v3 RIConv++ (logits, from A-layer clean dumps) +
c1 3DZD 1-NN distance + c2 Tatsuma FPFH-SVM OVO margins + c3 Guerra topo MLP.

All knobs chosen ONLY on v3 val (last 1816 of train_set_2.csv); then frozen
and applied to test once. Two fusion families (Guerra_v4 style):
  (A) hard weighted majority, ties -> drop lowest-confidence voter -> prior
  (B) calibrated probability blend  sum_m w_m * softmax(score_m / tau_m)
Optional empty-class guard (classes with only 2 train samples; the track's
9 "empty" test classes 26,42,44,50,58,63,72,77,95) is itself selected on val.
"""
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
R = BASE / "results_final"
N_FIT = 7428


def met(y, p):
    return np.array([accuracy_score(y, p), balanced_accuracy_score(y, p),
                     f1_score(y, p, average="macro", zero_division=0),
                     f1_score(y, p, average="micro", zero_division=0)])


def c1_class_scores(tr_y, dist, taus):
    """Per-query min distance to each gallery class -> softmax probs per tau."""
    nq, ng = dist.shape
    assert ng == len(tr_y)
    dmin = np.full((nq, 97), np.inf, np.float64)
    finite = np.isfinite(dist)
    for c in range(97):
        cols = np.flatnonzero(tr_y == c)
        if cols.size:
            sub = dist[:, cols]
            sub = np.where(np.isfinite(sub), sub, np.inf)
            dmin[:, c] = sub.min(1)
    # volume-gated (inf) classes: neutral fill = that query's worst finite
    # per-class distance, so unavailable classes rank below every available
    # one but share a single constant (never beat genuinely far classes).
    rowmax = np.where(np.isfinite(dmin).any(1),
                      np.nanmax(np.where(np.isfinite(dmin), dmin, np.nan), axis=1),
                      np.nanmax(np.where(finite, dist, np.nan)))
    dmin = np.where(np.isfinite(dmin), dmin, rowmax[:, None])
    out = {}
    for t in taus:
        z = -dmin / t
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        out[t] = e / e.sum(1, keepdims=True)
    return out


def softmax_with_taus(sc, taus):
    out = {}
    for t in taus:
        z = sc / t
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        out[t] = e / e.sum(1, keepdims=True)
    return out


def hard_majority(hard, conf, weights, prior):
    """Guerra_v4 rule generalized: weighted majority; if the top weight total
    is tied, exclude the proposal of the least confident voter and fall back
    to a-priori frequency among remaining tied classes."""
    n = hard.shape[0]
    pred = np.zeros(n, int)
    for i in range(n):
        v = np.zeros(97)
        for m in range(hard.shape[1]):
            v[hard[i, m]] += weights[m]
        top = v.max()
        cand = np.flatnonzero(v == top)
        if cand.size == 1:
            pred[i] = cand[0]
            continue
        # tie: drop the class proposed by the lowest-confidence voter,
        # repeatedly while >1 tied class, then prior tie-break
        cand = set(cand.tolist())
        order = np.argsort(conf[i])  # least confident first
        for m in order:
            if len(cand) == 1:
                break
            cand.discard(int(hard[i, m]))
        cand = list(cand) if cand else range(97)
        pred[i] = cand[np.argmax(prior[cand])]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft-weights", default="0,1,2")
    ap.add_argument("--t1", default="0.5,1,2,4,8")        # c1 distance tau
    ap.add_argument("--t2", default="0.5,1,2")            # c2 margin tau
    ap.add_argument("--select", default="onese",
                    choices=["argmax", "bootstrap", "onese"])
    ap.add_argument("--topk", type=int, default=300)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse results_final/fuse_scan_top.npz scan cache")
    ap.add_argument("--no-extra", action="store_true",
                    help="use only the canonical 7 C-members "
                         "(ignore c1d/c2d/c4 auto-discovery)")
    ap.add_argument("--out", default="test_c_fusion.csv",
                    help="output CSV filename under results_final/")
    ap.add_argument("--out-arrays", default="c_fusion_arrays.npz",
                    help="output npz filename under results_final/")
    args = ap.parse_args()

    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    ytr, yval = tr.class_id.values[:N_FIT], tr.class_id.values[N_FIT:]
    yte = pd.read_csv(BASE / "csv/test_set_ground_truth.csv").class_id.values
    prior = np.bincount(ytr, minlength=97) / N_FIT

    # ---- voter probabilities on val and test ----
    c1 = np.load(R / "c1_arrays.npz")
    t1s = [float(x) for x in args.t1.split(",")]
    c1v = c1_class_scores(ytr, c1["val_dist"], t1s)
    c1t = c1_class_scores(tr.class_id.values, c1["test_dist"], t1s)
    c2 = np.load(R / "c2_arrays.npz")
    t2s = [float(x) for x in args.t2.split(",")]
    c2v = softmax_with_taus(c2["val_margin"], t2s)
    c2t = softmax_with_taus(c2["test_margin"], t2s)
    c3 = np.load(R / "c3_arrays.npz")
    p3v, p3t = c3["val_soft"], c3["test_soft"]
    c1b = np.load(R / "c1b_arrays.npz")
    pbv, pbt = c1b["val_soft"], c1b["test_soft"]
    c1c = np.load(R / "c1c_arrays.npz")
    pcv, pct = c1c["val_soft"], c1c["test_soft"]
    have_c2b = (R / "c2b_arrays.npz").exists()
    if have_c2b:
        c2b = np.load(R / "c2b_arrays.npz")
        psv, pst = c2b["val_soft"], c2b["test_soft"]
    have_c3b = (R / "c3b_arrays.npz").exists()
    if have_c3b:
        c3b = np.load(R / "c3b_arrays.npz")
        pqv, pqt = c3b["val_soft"], c3b["test_soft"]
    # further optional soft members, auto-included in fixed canonical order
    extra = {}
    for nm in () if args.no_extra else ("c1d", "c2c", "c2d", "c4"):
        if (R / (nm + "_arrays.npz")).exists():
            z = np.load(R / (nm + "_arrays.npz"))
            extra[nm] = (z["val_soft"], z["test_soft"],
                         z["val_pred"], z["test_pred"])

    have_v3 = (R / "cache/val_clean.npz").exists() and (R / "cache/test_clean.npz").exists()
    if have_v3:
        vc = np.load(R / "cache/val_clean.npz")
        tc = np.load(R / "cache/test_clean.npz")
        p0v = np.exp(vc["logp"]); p0v /= p0v.sum(1, keepdims=True)
        p0t = np.exp(tc["logp"]); p0t /= p0t.sum(1, keepdims=True)
        assert p0v.shape[0] == len(yval) and p0t.shape[0] == len(yte)
    else:
        print("NOTE: v3 clean dumps missing -> C-only fusion (w_v3 forced 0)")
        p0v = p0t = None

    names = ["v3 RIConv++", "c1 3DZD", "c2 FPFH", "c3 Guerra", "c1b zdz-knn", "c1c zdz-block"]

    def table(P, y, tag):
        print("  %-12s acc=%.4f bal=%.4f macF1=%.4f" % (tag, *met(y, P.argmax(1))[:3]))

    print("== single-model reference ==")
    print("val:")
    if have_v3:
        table(p0v, yval, names[0])
    table(c1v[t1s[1]], yval, names[1])
    table(c2v[t2s[1]], yval, names[2]); table(p3v, yval, names[3])
    table(pbv, yval, names[4]); table(pcv, yval, names[5])
    print("test:")
    if have_v3:
        table(p0t, yte, names[0])
    table(c1t[t1s[1]], yte, names[1])
    table(c2t[t2s[2]], yte, names[2]); table(p3t, yte, names[3])
    table(pbt, yte, names[4]); table(pct, yte, names[5])
    if have_c3b:
        table(pqv, yval, "c3b topo-skl"); table(pqt, yte, "c3b topo-skl")
    if have_c2b:
        table(psv, yval, "c2b FPFH-ET"); table(pst, yte, "c2b FPFH-ET")

    # val selection uses fit-split counts; frozen test application uses full train
    guard_fit = np.flatnonzero(np.bincount(ytr, minlength=97) <= 2)
    guard_full = np.flatnonzero(np.bincount(tr.class_id.values, minlength=97) <= 2)

    # ---- (B) probability blend grid (vectorized over weight combos) ----
    # Canonical member order (v3 prepended when present):
    # c1, c2, c3, c3b?, c1b, c1c, c2b?
    wgrid = [int(x) for x in args.soft_weights.split(",")]
    nval = len(yval)
    tc = np.bincount(yval, minlength=97).astype(np.float64)
    present_c = tc > 0                       # classes present in val

    def macro_acc(preds):                    # preds: (K, nval) int
        K = preds.shape[0]
        bpc = np.bincount((np.arange(K)[:, None] * 97 + preds).ravel(),
                          minlength=K * 97).reshape(K, 97).astype(np.float64)
        correct = preds == yval[None, :]
        btp = np.bincount((np.arange(K)[:, None] * 97
                           + np.where(correct, yval[None, :], 0)).ravel(),
                          weights=correct.ravel(),
                          minlength=K * 97).reshape(K, 97)
        denom = 2 * btp + (bpc - btp) + (tc[None, :] - btp)
        f1 = np.divide(2 * btp, denom, out=np.zeros_like(denom), where=denom > 0)
        union = present_c[None, :] | (bpc > 0)
        mac = (f1 * union).sum(1) / union.sum(1)
        acc = correct.mean(1)
        return mac, acc

    def member_softs(tau1, tau2):
        """Canonical soft-prob lists (val, test), v3 first if present."""
        fv = [c1v[tau1], c2v[tau2], p3v]
        ft = [c1t[tau1], c2t[tau2], p3t]
        if have_c3b:
            fv.append(pqv); ft.append(pqt)
        fv += [pbv, pcv]; ft += [pbt, pct]
        if have_c2b:
            fv.append(psv); ft.append(pst)
        for nm in ("c1d", "c2c", "c2d", "c4"):
            if nm in extra:
                fv.append(extra[nm][0]); ft.append(extra[nm][1])
        if have_v3:
            fv = [p0v] + fv; ft = [p0t] + ft
        return fv, ft

    def member_hards(tau1, tau2):
        """Hard predictions + confidences, same order as member_softs."""
        hv = [c1["val_pred"], c2["val_pred"], c3["val_pred"]]
        ht = [c1["test_pred"], c2["test_pred"], c3["test_pred"]]
        cv = [c1v[tau1].max(1), c2v[tau2].max(1), p3v.max(1)]
        ct = [c1t[tau1].max(1), c2t[tau2].max(1), p3t.max(1)]
        if have_c3b:
            hv.append(c3b["val_pred"]); ht.append(c3b["test_pred"])
            cv.append(pqv.max(1)); ct.append(pqt.max(1))
        hv += [c1b["val_pred"], c1c["val_pred"]]
        ht += [c1b["test_pred"], c1c["test_pred"]]
        cv += [pbv.max(1), pcv.max(1)]
        ct += [pbt.max(1), pct.max(1)]
        if have_c2b:
            hv.append(c2b["val_pred"]); ht.append(c2b["test_pred"])
            cv.append(psv.max(1)); ct.append(pst.max(1))
        for nm in ("c1d", "c2c", "c2d", "c4"):
            if nm in extra:
                sv_, st_, hvv_, htt_ = extra[nm]
                hv.append(hvv_); ht.append(htt_)
                cv.append(sv_.max(1)); ct.append(st_.max(1))
        if have_v3:
            hv = [p0v.argmax(1)] + hv; ht = [p0t.argmax(1)] + ht
            cv = [p0v.max(1)] + cv; ct = [p0t.max(1)] + ct
        return hv, ht, cv, ct

    legend = (["v3 RIConv++"] if have_v3 else []) + \
             ["c1 3DZD", "c2 FPFH", "c3 Guerra"] + \
             (["c3b topo-skl"] if have_c3b else []) + ["c1b zdz-knn", "c1c zdz-block"] + \
             (["c2b FPFH-ET"] if have_c2b else []) + \
             [{"c1d": "c1d zdz-svm", "c2c": "c2c fpfh-svmP",
               "c2d": "c2d fpfh-knn", "c4": "c4 zdz+fpfh-svm"}[nm]
              for nm in ("c1d", "c2c", "c2d", "c4") if nm in extra]
    print("blend member order:", legend, flush=True)

    # Scan keeps the TOPK val configs (with predictions); final choice is
    # either plain argmax or a bootstrap-robust selection (resample val with
    # replacement and pick the config with best mean macro-F1; ties ->
    # simpler weights). Everything uses val labels only.
    TOPK, NBOOT = args.topk, args.boot
    rng = np.random.default_rng(777)
    cache_npz, cache_json = R / "fuse_scan_top.npz", R / "fuse_scan_meta.json"
    top_mac, top_acc, top_pred, top_meta = None, None, None, []

    def absorb(mac, acc, pr, metas):
        nonlocal top_mac, top_acc, top_pred, top_meta
        if top_mac is None:
            m0, a0, p0, me0 = mac, acc, pr, metas
        else:
            m0 = np.concatenate([top_mac, mac])
            a0 = np.concatenate([top_acc, acc])
            p0 = np.concatenate([top_pred, pr], 0)
            me0 = top_meta + metas
        keep = np.argsort(-m0)[:TOPK]
        top_mac = m0[keep]; top_acc = a0[keep]
        top_pred = p0[keep]; top_meta = [me0[i] for i in keep]

    if args.use_cache and cache_npz.exists() and cache_json.exists():
        import json
        zc = np.load(cache_npz)
        top_pred, top_mac, top_acc = zc["pred"], zc["mac"], zc["acc"]
        with open(cache_json) as f:
            mc = json.load(f)
        top_meta = [(tuple(m["w"]), m["tau1"], m["tau2"], m["guard"])
                    for m in mc]
        print("loaded scan cache: %d configs" % len(top_mac), flush=True)
    else:
        for tau1 in t1s:
            for tau2 in t2s:
                Mv, _ = member_softs(tau1, tau2)
                M = np.stack(Mv).astype(np.float64)    # (m, nval, 97)
                nm = M.shape[0]
                W = np.array(list(itertools.product(wgrid, repeat=nm)),
                             np.float64)
                W = W[W.sum(1) > 0]
                Mflat = M.reshape(nm, -1)
                CH = 512
                for i0 in range(0, len(W), CH):
                    Wc = W[i0:i0 + CH]
                    sc = (Wc @ Mflat).reshape(len(Wc), nval, 97)
                    for guard in (False, True):
                        if guard:
                            sc2 = sc.copy()
                            sc2[:, :, guard_fit] = -1
                        else:
                            sc2 = sc
                        pr = sc2.argmax(2)
                        mac, acc = macro_acc(pr)
                        metas = [(tuple(int(x) for x in Wc[j]),
                                  tau1, tau2, guard)
                                 for j in range(len(Wc))]
                        absorb(mac, acc, pr, metas)

        # persist top-K scan results for offline selection-rule analysis
        import json
        np.savez_compressed(R / "fuse_scan_top.npz", pred=top_pred,
                            mac=top_mac, acc=top_acc)
        with open(R / "fuse_scan_meta.json", "w") as f:
            json.dump([{"w": list(m[0]), "tau1": m[1], "tau2": m[2],
                        "guard": bool(m[3])} for m in top_meta], f)

    if args.select == "argmax":
        sel = 0
    else:
        K = len(top_mac)
        bmac = np.zeros(K); bacc = np.zeros(K); bsq = np.zeros(K)
        for b in range(NBOOT):
            bi = rng.integers(0, nval, nval)
            yb = yval[bi]; pb = top_pred[:, bi]
            tcb = np.bincount(yb, minlength=97).astype(np.float64)
            pres = tcb > 0
            bpc = np.bincount((np.arange(K)[:, None] * 97 + pb).ravel(),
                              minlength=K * 97).reshape(K, 97).astype(np.float64)
            cor = pb == yb[None, :]
            btp = np.bincount((np.arange(K)[:, None] * 97
                               + np.where(cor, yb[None, :], 0)).ravel(),
                              weights=cor.ravel(),
                              minlength=K * 97).reshape(K, 97)
            den = 2 * btp + (bpc - btp) + (tcb[None] - btp)
            f1 = np.divide(2 * btp, den, out=np.zeros_like(den), where=den > 0)
            uni = pres[None] | (bpc > 0)
            x = (f1 * uni).sum(1) / uni.sum(1)
            bmac += x
            bsq += x * x
            bacc += cor.mean(1)
        bmac /= NBOOT; bacc /= NBOOT
        bse = np.sqrt(np.maximum(bsq / NBOOT - bmac ** 2, 0))
        ibest = int(np.argmax(bmac))
        if args.select == "bootstrap":
            order = sorted(range(K),
                           key=lambda i: (-bmac[i], -bacc[i],
                                          sum(top_meta[i][0]), -top_mac[i]))
            sel = order[0]
        else:
            # pre-registered one-SE rule: within one bootstrap SE of the
            # best mean, pick the simplest config (sum weights, then number
            # of nonzero members), then higher mean
            thr = bmac[ibest] - bse[ibest]
            pool = [i for i in range(K) if bmac[i] >= thr]
            pool.sort(key=lambda i: (sum(top_meta[i][0]),
                                     sum(x > 0 for x in top_meta[i][0]),
                                     -bmac[i],
                                     top_meta[i][1], top_meta[i][2]))
            sel = pool[0]
            print("one-SE: threshold=%.4f pool=%d/%d"
                  % (thr, len(pool), K), flush=True)
        print("bootstrap: chosen mean mac=%.4f (+-%.4f) acc=%.4f ; "
              "argmax-on-mean mac=%.4f"
              % (bmac[sel], bse[sel], bacc[sel], bmac[ibest]), flush=True)
    w, tau1, tau2, guard = top_meta[sel]
    cfg = ("blend", w, tau1, tau2, guard)
    print("\n== selected blend on val (%s) ==" % args.select)
    print("config", cfg, "val mac=%.4f acc=%.4f" % (top_mac[sel], top_acc[sel]))
    fv, ft = member_softs(tau1, tau2)
    sv = sum(w[m] * fv[m] for m in range(len(fv)))
    st = sum(w[m] * ft[m] for m in range(len(ft)))
    if guard:
        sv[:, guard_fit] = -1
        st[:, guard_full] = -1
    pred_val, pred_te = sv.argmax(1), st.argmax(1)
    print("VAL  acc=%.4f bal=%.4f macF1=%.4f micF1=%.4f" % tuple(met(yval, pred_val)))
    mte = met(yte, pred_te)
    print("TEST acc=%.4f bal=%.4f macF1=%.4f micF1=%.4f" % tuple(mte))

    # ---- (A) hard weighted majority for reference ----
    hvv, htt, cvv, ctt = member_hards(tau1, tau2)
    hard_v = np.stack(hvv, 1)
    hard_t = np.stack(htt, 1)
    conf_v = np.stack(cvv, 1)
    conf_t = np.stack(ctt, 1)
    # w ordering matches the voter stacking above (v3 first only if present)
    hv = hard_majority(hard_v, conf_v, w,
                       np.bincount(ytr, minlength=97) / N_FIT)
    ht = hard_majority(hard_t, conf_t, w,
                       np.bincount(tr.class_id.values, minlength=97) / 9244)
    print("\nhard weighted majority (blend weights) VAL  acc=%.4f bal=%.4f macF1=%.4f"
          % tuple(met(yval, hv)[:3]))
    print("hard weighted majority (blend weights) TEST acc=%.4f bal=%.4f macF1=%.4f"
          % tuple(met(yte, ht)[:3]))

    te_raw = pd.read_csv(BASE / "new_data/test_set_2.csv").anonymised_protein_id.astype(str).values
    pd.DataFrame({"anonymised_protein_id": te_raw, "predicted_label": pred_te}).to_csv(
        R / args.out, index=False)
    np.savez_compressed(R / args.out_arrays,
                        val_pred_blend=pred_val, test_pred_blend=pred_te,
                        val_pred_hard=hv, test_pred_hard=ht)
    print("saved results_final/%s" % args.out)


if __name__ == "__main__":
    main()
