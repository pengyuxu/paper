"""Error-set overlap (Jaccard) between members on validation (n=1816).

For member predictions p, q and labels y, define binary error indicators
e = (p != y); rho = |e_p & e_q| / |e_p | e_q|. Lower rho = more
complementary errors (Eq. 5 in the paper).

All predictions used here are the exact signals that enter the frozen
system: s1/s2/v3 are MC-FPS K=25 mean-logit argmax from the cache;
classical members are the stored val_pred in results_final/c{1,1d,2,3}
_arrays.npz. Reproduces the numbers reported in Section V-B and V-E.

Members whose cache has not been produced are skipped (with a hint):
  s2, c1, c1d, c2, c3 : produced by the standard pipeline (stages 1-2)
  v3                 : dump_mcfps.py defaults on the downloadable v3
                       best_model.pth (auto in run_all stage 1 if present)
  s1                 : optional seed-1 run:
                         train_v5.py --seed 1 --log_dir riconv_large_v5_s1
                         dump_mcfps.py --log_dir riconv_large_v5_s1 \
                             --ckpt swa_model.pth --K 25 --prefix mcfps_v5s1
                         dump_v5.py --log_dir riconv_large_v5_s1 \
                             --ckpt swa_model.pth --tag s1_swa
"""
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RF = BASE / "results_final"
C = RF / "cache"

# validation labels: any of these files carries the identical y vector
yv = None
for f in [C / "v5_s1_swa_val.npz", C / "v5_s2_swa_val.npz",
          RF / "c3_arrays.npz"]:
    if f.exists():
        yv = np.load(f)["y" if f.suffix == ".npz" and "arrays" not in f.name
                       else "val_y"]
        break
if yv is None:
    raise SystemExit(
        "no validation-label cache found: run stage 1 (dump_v5) first")

pred = {}

def add_deep(name, path):
    if path.exists():
        pred[name] = np.load(path).astype(np.float64).mean(0).argmax(1)

add_deep("s1", C / "mcfps_v5s1_val_logp.npy")
add_deep("s2", C / "mcfps_v5s2_val_logp.npy")
add_deep("v3", C / "mcfps_val_logp.npy")

for name in ["c1", "c1d", "c2", "c3"]:
    f = RF / ("%s_arrays.npz" % name)
    if f.exists():
        pred[name] = np.load(f)["val_pred"]


def rho(a, b):
    ea = pred[a] != yv
    eb = pred[b] != yv
    return float((ea & eb).sum() / (ea | eb).sum())

print("loaded members:", sorted(pred))
print("validation error counts:",
      {k: int((p != yv).sum()) for k, p in pred.items()})

pairs = [("s2", "c1"), ("s2", "c1d"), ("s2", "c2"), ("s2", "v3"),
         ("s2", "c3"),
         ("s1", "c1"), ("s1", "c1d"), ("s1", "c2"), ("s1", "v3"),
         ("s1", "c3"), ("s1", "s2")]
skipped = []
for a, b in pairs:
    if a in pred and b in pred:
        print("rho(%s,%s) = %.4f" % (a, b, rho(a, b)))
    else:
        skipped.append("rho(%s,%s)" % (a, b))

if skipped:
    print("\nskipped (cache missing): %s" % ", ".join(skipped))
    print("see the module docstring for how to produce each cache.")
