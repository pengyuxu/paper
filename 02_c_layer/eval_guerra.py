"""C3 evaluation: Guerra_v1 topological descriptors.

Faithful re-implementation of marcoguerra192/ProteinClassifier2025 learning
pipeline, with our fixed validation protocol (no test-set fitting):
  - sector 104-d raw -> drop always-zero H2 column per block (->96)
  - StandardScaler + linear decorrelation, FIT ON TRAIN SPLIT ONLY
    (their published notebook refits the scaler on the test set; we do not)
  - 8 rotation-group sector permutations per protein; alpha replicated x8
  - SimpleNN (input->111->100->97->97, dropout .25), Adam 4e-4 wd 1e-4,
    CE, 5000 epochs, patience 500 on val loss
  - 8-view hard majority vote, ties broken by train class frequency
Config/early-stopping use v3 val (last 1816). Test model retrains on all
9244 with an 80/20 stratified internal split for early stopping only.
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE = Path(__file__).resolve().parent.parent
GUERRA = BASE / "external/guerra"
sys.path.insert(0, str(GUERRA))
from src.descriptors.spherical_sectors import spherical_block_permutations  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

D = BASE / "results_final/guerra"
NTR_FIT = 7428
SEED = 1


class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes=97):
        super().__init__()
        self.dr = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(input_dim, 111)
        self.fc2 = nn.Linear(111, 100)
        self.fc3 = nn.Linear(100, 97)
        self.fc4 = nn.Linear(97, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x)); x = self.dr(x)
        x = self.relu(self.fc2(x)); x = self.dr(x)
        x = self.relu(self.fc3(x)); x = self.dr(x)
        return self.fc4(x)


def load_descs():
    tr = pd.read_csv(BASE / "csv/train_set_2.csv")
    te = pd.read_csv(BASE / "new_data/test_set_2.csv")
    keys = list(tr.protein_id.astype(str)) + [str(p).replace(".vtk", "") for p in te.anonymised_protein_id]
    A, S, miss = [], [], []
    for k in keys:
        f = D / (k + ".npz")
        if not f.exists():
            miss.append(k); A.append(np.zeros(75, np.float32)); S.append(np.zeros(104, np.float32)); continue
        z = np.load(f); A.append(z["alpha"]); S.append(z["sector"])
    return np.asarray(A, np.float64), np.asarray(S, np.float64), tr.class_id.values, miss


def clean(alpha, sector, n_fit):
    """Scalers/decorrelation fit on the first n_fit rows only (train split)."""
    exclude = [8 + 13 * k for k in range(8)]
    keep = [j for j in range(104) if j not in exclude]
    S = sector[:, keep].copy()
    sc = StandardScaler().fit(S[:n_fit])
    S = sc.transform(S)
    for k in range(8):  # residualize against the median column of each 12-block
        c1, c2, c3 = 12 * k, 1 + 12 * k, 2 + 12 * k
        q1, q2, q3 = 3 + 12 * k, 4 + 12 * k, 5 + 12 * k
        for tgt, ref in ((c1, c2), (c3, c2), (q1, q2), (q3, q2)):
            reg = LinearRegression().fit(S[:n_fit, [ref]], S[:n_fit, tgt])
            S[:, tgt] = S[:, tgt] - reg.predict(S[:, [ref]])
    A = alpha.copy()
    sca = StandardScaler().fit(A[:n_fit])
    A = sca.transform(A)
    for i in range(5):  # their H0 persistent-image decorrelation groups
        c1, c2, cc, c4, c5 = [5 * i + j for j in range(5)]
        for tgt in (c1, c2, c4, c5):
            reg = LinearRegression().fit(A[:n_fit, [cc]], A[:n_fit, tgt])
            A[:, tgt] = A[:, tgt] - reg.predict(A[:, [cc]])
    return A, S


def augment(A, S):
    n = A.shape[0]
    Sa = np.zeros((n * 8, S.shape[1]))
    Aa = np.repeat(A, 8, axis=0)
    for r in range(n):
        Sa[8 * r:8 * r + 8] = np.asarray(spherical_block_permutations(S[r], S.shape[1] // 8))
    return np.concatenate([Sa, Aa], axis=1).astype(np.float32)


def train_net(Xtr, ytr, Xva, yva, device, epochs=5000, patience=500, seed=SEED, verbose=False):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SimpleNN(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=4e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva); yv = torch.tensor(yva, dtype=torch.long)
    best, bad, best_sd = float("inf"), 0, None
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = crit(model(xb.to(device)), yb.to(device)); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv.to(device)), yv.to(device)).item()
        if vl < best - 1e-5:
            best, bad, best_sd = vl, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 100 == 0:
            print("ep%d val_loss=%.4f best=%.4f" % (ep, vl, best), flush=True)
    model.load_state_dict(best_sd)
    return model, best


def soft_views(model, X, device, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            out.append(torch.softmax(model(torch.tensor(X[i:i + bs]).to(device)), 1).cpu().numpy())
    p = np.concatenate(out)                       # (n*8, 97)
    k = p.shape[1]
    return p.reshape(-1, 8, k)


def vote(view_prob, prior):
    """Hard majority over 8 views, ties broken by a-priori class frequency."""
    n = view_prob.shape[0]
    votes = np.zeros((n, 97))
    hard = view_prob.argmax(2)
    np.add.at(votes, (np.repeat(np.arange(n), 8), hard.reshape(-1)), 1)
    pred = np.zeros(n, int)
    for i in range(n):
        top = votes[i] == votes[i].max()
        cand = np.flatnonzero(top)
        pred[i] = cand[np.argmax(prior[cand])]
    return pred, votes


def met(y, p):
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0),
            f1_score(y, p, average="micro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-epochs", type=int, default=5000)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    A0, S0, ytr_all, miss = load_descs()
    print("descs loaded alpha=%s sector=%s missing=%d" % (A0.shape, S0.shape, len(miss)), flush=True)
    if miss[:5]:
        print("missing examples:", miss[:5])
    yval = ytr_all[NTR_FIT:]

    # ---- val protocol: transforms fit on first 7428 only ----
    Av, Sv = clean(A0, S0, NTR_FIT)
    Xv = augment(Av, Sv)
    prior = np.bincount(ytr_all[:NTR_FIT], minlength=97).astype(float) / NTR_FIT
    print("== val model (train=%d, val=%d proteins)" % (NTR_FIT, len(yval)), flush=True)
    t0 = time.time()
    m_val, best = train_net(Xv[:NTR_FIT * 8], ytr_all[:NTR_FIT].repeat(8),
                            Xv[NTR_FIT * 8:9244 * 8], yval.repeat(8),
                            device, args.max_epochs)
    print("trained %.0fs best_val_loss=%.4f" % (time.time() - t0, best), flush=True)
    pv = soft_views(m_val, Xv[NTR_FIT * 8:9244 * 8], device)
    hv_val, _ = vote(pv, prior)
    print("VAL acc=%.4f bal=%.4f macF1=%.4f micF1=%.4f" % met(yval, hv_val), flush=True)

    # ---- test model: transforms refit on ALL 9244 train proteins ----
    At, St = clean(A0, S0, 9244)
    X = augment(At, St)
    Xte_aug = X[9244 * 8:]
    ytr_aug = np.repeat(ytr_all, 8)
    print("== test model: retrain on all 9244 (transforms fit on 9244)", flush=True)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
    ti, vi = next(sss.split(X[:9244 * 8], ytr_aug))
    t0 = time.time()
    m_te, best = train_net(X[:9244 * 8][ti], ytr_aug[ti], X[:9244 * 8][vi], ytr_aug[vi],
                           device, args.max_epochs)
    print("trained %.0fs best_val_loss=%.4f" % (time.time() - t0, best), flush=True)
    pt = soft_views(m_te, Xte_aug, device)
    hv_te, votes_te = vote(pt, np.bincount(ytr_all, minlength=97) / 9244)

    gt = pd.read_csv(BASE / "csv/test_set_ground_truth.csv")
    yte = gt.class_id.values
    print("TEST acc=%.4f bal=%.4f macF1=%.4f micF1=%.4f" % met(yte, hv_te), flush=True)

    te_raw = pd.read_csv(BASE / "new_data/test_set_2.csv").anonymised_protein_id.astype(str).values
    pd.DataFrame({"anonymised_protein_id": te_raw, "predicted_label": hv_te}).to_csv(
        BASE / "results_final/test_c3_guerra.csv", index=False)
    np.savez_compressed(
        BASE / "results_final/c3_arrays.npz",
        val_soft=pv.mean(1).astype(np.float32), val_pred=hv_val, val_y=yval,
        test_soft=pt.mean(1).astype(np.float32), test_pred=hv_te, test_y=yte)
    print("saved results_final/test_c3_guerra.csv + c3_arrays.npz", flush=True)


if __name__ == "__main__":
    main()
