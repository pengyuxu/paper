"""C3 / C3b Guerra topological classifiers on both versions (paper §4.6).

Raw 179-coordinate descriptors come from experiments/results/desc/
guerra_{train,test}.npz (built with the published Guerra code).  Here we
reproduce the released cleaning and learning protocol with two changes that
match the revised paper's discipline:

  * all StandardScaler / LinearRegression cleaning is fit ONLY on the 7,428
    training rows (the released script refits a second, separate scaler on the
    test matrix); the released residual formulas, including the Alpha block's
    self-predict residual, are otherwise replicated verbatim;
  * the 80/20 random early-stopping split is replaced by the fixed 1,816-row
    validation list.

Two artifacts are produced per version:
  --fit trainfit : transforms fit on 7,428 rows; validation + test reported.
  --fit fullrefit: transforms + MLP refit on all 9,244 rows (the paper's
                   test-submission artifact); test reported only.

c3  : 171 -> 111 -> 100 -> 97 -> nc MLP, dropout 0.25, Adam(4e-4,wd1e-4),
      CrossEntropy, batch 128, <=5000 epochs, patience 500 on validation loss;
      8 rotation-group views, hard majority vote, ties broken by training-set
      class frequency.
c3b : RBF-SVM C=32, gamma='scale' on the canonical (view-0) 171-vector.

Outputs: experiments/results/<ver>/descriptors/guerra_c3*_*.npz + json.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "methods", "guerra", "repo"))
sys.path.insert(0, REPO)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from src.descriptors.spherical_sectors import spherical_block_permutations

from experiments.common import paths as vp
from experiments.common import metrics_lib

DESC = vp.EXP_DIR / "results" / "desc"
EXCL = [8 + 13 * k for k in range(8)]


def load_raw(version):
    cfg = vp.version_cfg(version)
    tr = np.load(DESC / "guerra_train.npz")
    te = np.load(DESC / "guerra_test.npz")
    tdf = pd.read_csv(vp.csv_dir(version) / "train_set_2.csv")
    id2lab = dict(zip(tdf.protein_id.astype(str), tdf.class_id.astype(int)))
    tr_ids = tr["ids"].astype(str)
    ytr = np.array([id2lab[i] for i in tr_ids], np.int64)
    n = vp.TRAIN_ROWS

    gtdf = pd.read_csv(vp.csv_dir(version) / cfg["gt_file"])
    g = gtdf.set_index(cfg["gt_anon_col"])[cfg["gt_label_col"]].to_dict()
    gmap = {str(k)[:-4] if str(k).endswith(".vtk") else str(k): int(v)
            for k, v in g.items()}
    keep = [str(x)[:-4] if str(x).endswith(".vtk") else str(x)
            for x in pd.read_csv(cfg["dataset_dir"] / cfg["test_list_file"]
                                 )[cfg["test_id_col"]]]
    pos = {i: k for k, i in enumerate(te["ids"].astype(str))}
    Xte = te["X"][np.array([pos[i] for i in keep], np.int64)]
    yte = np.array([gmap[i] for i in keep], np.int64)
    return tr["X"], ytr, tr_ids, Xte, yte, np.array(keep)


def clean(Xfit, *others):
    """Released cleaning, fitted on Xfit; applies to all other matrices.
    Returns cleaned matrices for [Xfit, *others]."""
    mats = [np.array(Xfit, np.float64, copy=True)] + \
           [np.array(m, np.float64, copy=True) for m in others]

    # ---- alpha block (columns 104:179) ----
    cols = [slice(104, 179)]
    sc = StandardScaler().fit(mats[0][:, 104:179])
    for m in mats:
        m[:, 104:179] = sc.transform(m[:, 104:179])
    # released residual pattern (including self-predict on c1,c2,c4,c5)
    for i in range(5):
        c1, c2, cc, c4, c5 = [104 + 5 * i + q for q in range(5)]
        regs = {q: LinearRegression().fit(mats[0][:, [cc]], mats[0][:, t])
                for q, t in ((c1, c1), (c2, c2), (c4, c4), (c5, c5))}
        for m in mats:
            for t, qx in ((c1, c1), (c2, c2), (c4, c4), (c5, c5)):
                m[:, t] = m[:, t] - regs[t].predict(m[:, [qx]])

    # ---- sectors block (104 cols, drop 8 H2 zeros -> 96) ----
    mats = [np.delete(m, EXCL, axis=1) for m in mats]
    sc2 = StandardScaler().fit(mats[0][:, :96])
    for m in mats:
        m[:, :96] = sc2.transform(m[:, :96])
    for k in range(8):
        c1, c2, c3 = [12 * k + q for q in range(3)]
        q1, q2, q3 = [12 * k + 3 + q for q in range(3)]
        regs = {}
        for t, f in ((c1, c2), (c3, c2), (q1, q2), (q3, q2)):
            regs[t] = LinearRegression().fit(mats[0][:, [f]], mats[0][:, t])
        for m in mats:
            for t, f in ((c1, c2), (c3, c2), (q1, q2), (q3, q2)):
                m[:, t] = m[:, t] - regs[t].predict(m[:, [f]])
    return mats


def eight_views(Z):
    """Z: (N,171) canonical view -> (8N,171) rotation-group permutations."""
    out = np.zeros((len(Z) * 8, Z.shape[1]), np.float32)
    for r in range(len(Z)):
        pv = np.array(spherical_block_permutations(Z[r, :96], 12))  # (8,96)
        out[8 * r:8 * r + 8, :96] = pv
        out[8 * r:8 * r + 8, 96:] = Z[r, 96:]
    return out


def metrics_row(scores, y, nc):
    return {k: float(metrics_lib.evaluate(scores, y, nc)[k])
            for k in metrics_lib.METRIC_ORDER}


def train_mlp(Xv, yv, Xt, yt, nc, device, seed=7,
              epochs=5000, patience=500):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.dr = nn.Dropout(0.25)
            self.f1 = nn.Linear(171, 111); self.f2 = nn.Linear(111, 100)
            self.f3 = nn.Linear(100, 97); self.f4 = nn.Linear(97, nc)

        def forward(self, x):
            x = torch.relu(self.f1(x)); x = self.dr(x)
            x = torch.relu(self.f2(x)); x = self.dr(x)
            x = torch.relu(self.f3(x)); x = self.dr(x)
            return self.f4(x)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    XtT = torch.tensor(Xt, dtype=torch.float32)
    ytT = torch.tensor(yt, dtype=torch.long)
    XvT = torch.tensor(Xv, dtype=torch.float32, device=device)
    yvT = torch.tensor(yv, dtype=torch.long, device=device)
    bs = 128
    best, wait, best_state = 1e18, 0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(XtT))
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = crit(model(XtT[idx].to(device)), ytT[idx].to(device))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for s in range(0, len(XvT), 512):
                vl += float(crit(model(XvT[s:s + 512]),
                                 yvT[s:s + 512])) * min(512, len(XvT) - s)
            vl /= len(XvT)
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, ep + 1


def mlp_views(model, Z, nc, device, train_freq=None, batch=2048):
    """8-view majority vote over N proteins (Z is 8N views) -> (N, nc).

    Returns vote fractions; ties in the argmax are broken exactly as in the
    released Learning.py: by the a-priori training-set class frequency, added
    as a 1e-4 perturbation (strict majority gap is >= 1/8, so it never
    overrides a strict winner).
    """
    import torch
    N = len(Z) // 8
    votes = np.zeros((N, nc), np.float64)
    with torch.no_grad():
        for s in range(0, len(Z), batch):
            xb = torch.tensor(Z[s:s + batch], dtype=torch.float32,
                              device=device)
            pred = model(xb).argmax(1).cpu().numpy()
            qi = (np.arange(s, s + len(pred)) // 8)
            np.add.at(votes, (qi, pred), 1.0)
    scores = votes / 8.0
    if train_freq is not None:
        scores = scores + 1e-4 * train_freq[None, :]
    return scores.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["paper", "updated"], required=True)
    ap.add_argument("--fit", choices=["trainfit", "fullrefit"],
                    default="trainfit")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    import torch
    nc = vp.version_cfg(args.version)["num_classes"]
    outdir = vp.RESULTS_DIR / args.version / "descriptors"
    outdir.mkdir(parents=True, exist_ok=True)

    Xall, yall, tr_ids, Xte, yte, idte = load_raw(args.version)
    n = vp.TRAIN_ROWS
    summary = {}

    if args.fit == "trainfit":
        Ztr, Zval, Ztest = clean(Xall[:n], Xall[n:], Xte)
        ytr, yva = yall[:n], yall[n:]
        idva = tr_ids[n:]
    else:
        Ztr, Ztest = clean(Xall, Xte)[:2]
        ytr = yall
        Zval, yva, idva = None, None, None

    # ---------------- c3 MLP on 8 views ----------------
    Vtr = eight_views(Ztr)
    ytr8 = np.repeat(ytr, 8)
    if args.fit == "trainfit":
        Vva = eight_views(Zval)
        yva8 = np.repeat(yva, 8)
    else:
        # fullrefit still needs an early-stop monitor: hold out 10% stratified
        rng = np.random.RandomState(7)
        hold = np.zeros(len(Vtr), bool)
        for c in np.unique(ytr8):
            idx = np.where(ytr8 == c)[0]
            take = rng.choice(idx, max(1, len(idx) // 10), replace=False)
            hold[take] = True
        Vva, yva8 = Vtr[hold], ytr8[hold]
        Vtr_, ytr8_ = Vtr[~hold], ytr8[~hold]
        Vtr, ytr8 = Vtr_, ytr8_
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, best_loss, used_ep = train_mlp(Vva, yva8, Vtr, ytr8, nc, device)
    print("MLP trained epochs=%d best_val_loss=%.4f device=%s" %
          (used_ep, best_loss, device), flush=True)

    # a-priori training-set class frequency (Learning.py tie-break rule)
    _cnt = np.zeros(nc)
    for _c in ytr:
        _cnt[_c] += 1
    train_freq = _cnt / max(_cnt.sum(), 1.0)

    Vte = eight_views(Ztest)
    ste = mlp_views(model, Vte, nc, device, train_freq)
    summary["c3_test"] = metrics_row(ste, yte, nc)
    tag = "c3_%s" % args.fit
    np.savez_compressed(outdir / ("guerra_%s_test.npz" % tag),
                        ids=idte.astype(str), scores=ste, labels=yte)
    if args.fit == "trainfit":
        sva = mlp_views(model, eight_views(Zval), nc, device, train_freq)
        summary["c3_val"] = metrics_row(sva, yva, nc)
        np.savez_compressed(outdir / ("guerra_%s_val.npz" % tag),
                            ids=idva.astype(str), scores=sva, labels=yva)

    # ---------------- c3b RBF-SVM on view 0 ----------------
    svm = SVC(C=32.0, kernel="rbf", gamma="scale",
              decision_function_shape="ovr", cache_size=2000)
    svm.fit(Ztr, ytr)  # Ztr is one canonical 171-vector per protein (view 0)
    sm_te = svm.decision_function(Ztest)
    summary["c3b_test"] = metrics_row(sm_te, yte, nc)
    tagb = "c3b_%s" % args.fit
    np.savez_compressed(outdir / ("guerra_%s_test.npz" % tagb),
                        ids=idte.astype(str), scores=sm_te.astype(np.float32),
                        labels=yte)
    if args.fit == "trainfit":
        sm_va = svm.decision_function(Zval)
        summary["c3b_val"] = metrics_row(sm_va, yva, nc)
        np.savez_compressed(outdir / ("guerra_%s_val.npz" % tagb),
                            ids=idva.astype(str),
                            scores=sm_va.astype(np.float32), labels=yva)

    with open(outdir / ("guerra_%s_metrics.json" % args.fit), "w") as f:
        json.dump(summary, f, indent=2)
    for k2, m in summary.items():
        print("%-14s acc=%.4f f1m=%.4f Pm=%.4f Rm=%.4f f1w=%.4f"
              % (k2, m["acc"], m["f1_macro"], m["prec_macro"],
                 m["rec_macro"], m["f1_weighted"]), flush=True)


if __name__ == "__main__":
    main()
