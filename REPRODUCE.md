# Replication Runbook

> [中文版 (Chinese)](REPRODUCE_ZH.md)

> Step-by-step manual for reproducing the experimental data of the SHREC 2025
> protein shape classification paper. Package overview at
> [README.md](README.md); this document focuses on **how to run each step, what
> you should see, and how to fix problems**.
> **Run all commands from the `replication/` root** (`cd replication/`).

**Replication targets (test set)**

| Metric | Target | Source |
| --- | --- | --- |
| Accuracy | 94.79 % | `expected_outputs/test_full_fusion.csv` |
| Balanced accuracy | 91.48 % | same |
| Macro-F1 | 92.25 % | same |

---

## 0. Preflight checklist

Confirm each item before running:

- [ ] **OS**: Linux (verified). On Windows, the `csv -> 00_data/splits` symlink must be replaced by a real directory or junction.
- [ ] **GPU**: NVIDIA + driver supporting CUDA 12.1 (this machine: 2x A40, driver 535.161.07 / max CUDA 12.2). Train with `--gpu 0` (GPU1 is often busy); stage-4-only verification runs on pure CPU.
- [ ] **Python 3.8.10**: `python --version` (system `python`; the actual paper environment). `python3.11` also works with matching dependency versions.
- [ ] **Disk**: ~30 GB free for training outputs + point clouds.
- [ ] **Package directories**: `new_data/`, `log/`, `results_final/`, `figures/` exist (placeholders).
- [ ] **`csv` symlink**: `ls csv/` should list the 3 split CSVs (pointing to `00_data/splits/`).

Check commands:
```bash
cd replication
ls csv/                       # expect train_set_2.csv / test_set_2.csv / test_set_ground_truth.csv
test -d new_data && echo OK   # placeholder directory is enough
python --version              # 3.8.x
```

---

## 1. Environment setup

Dependencies are pinned to the actual training machine (torch 2.3.0+cu121, see
`requirements.txt`), open3d included:

```bash
cd replication
python -m pip install -r requirements.txt          # add --user if permissions require
```

**Checkpoint 1**: environment ready
```bash
python -c "import torch,sklearn,open3d,trimesh; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
# expect: torch 2.3.0+cu121 cuda True (training machine); with cuda False only stage 4 runs
```

---

## 2. Data acquisition and verification

See `00_data/DOWNLOAD.md` for details.

### 2.1 Split CSVs (committed, no download)
`00_data/splits/` already contains:
- `train_set_2.csv` — 9244 rows (first 7428 train / last 1816 val), columns `protein_id, class_id`
- `test_set_2.csv` — 2321 rows, column `anonymised_protein_id`
- `test_set_ground_truth.csv` — 2321 rows, columns `anonymised_protein_id, class_id`

```bash
wc -l 00_data/splits/*.csv
# expect: 9245 / 2322 / 2322 (incl. header)
```

### 2.2 Point clouds `new_data/`
Each file `{basename}_pc.npy`, shape `(8192, 8)`, channels
`[x,y,z, nx,ny,nz, potential, normal_potential]`.

**Already committed**: `new_data/` contains the 2321 test point clouds
(~1.2 GB), works out of the box.
For the training point clouds (9244 files):
- **Option A (recommended, fast)**: download the precomputed archive via the link in `00_data/DOWNLOAD.md`, extract into `new_data/`.
- **Option B (extract yourself)**:
  ```bash
  # first download and extract train_set.tar.xz / test_set.tar.xz from the SHREC official site
  python 00_data/extract_pc_fast.py --vtk_dir <extracted train_set> --out_dir new_data
  python 00_data/extract_pc_fast.py --vtk_dir <extracted test_set>  --out_dir new_data
  ```

**Checkpoint 2**: point-cloud count
```bash
find new_data -name '*_pc.npy' | wc -l
# as committed: 2321 (test set); after adding train: 11565 (9244 + 2321)
ls new_data | head -1 | xargs -I{} python -c "import numpy as np; a=np.load('new_data/{}'); print(a.shape, a.dtype)"
# expect: (8192, 8) floatXX
```

### 2.3 Pretrained checkpoints
- **v3 `best_model.pth`**: **already committed** at
  `log/classification_shrec2025/riconv_large_shrec_run2_8k_largest/checkpoints/best_model.pth`;
  the dump scripts locate it automatically. To re-download see `01_a_layer/checkpoints/DOWNLOAD.md`.
- **v5 `swa_model.pth`**: produced by stage-1a training at `log/classification_shrec2025/riconv_large_v5_s2/checkpoints/swa_model.pth`; copy an existing one into that path if available.

---

## 3. Replication pipeline (5 stages)

Pipeline: `00_data -> 01_a_layer -> 02_c_layer -> 03_fusion -> 04_analysis`.

> One-shot: `./run_all.sh` (all 5 stages, incl. training ~24h+); `SKIP_TRAIN=1 ./run_all.sh` (skip training when checkpoints exist); `STAGE=N ./run_all.sh` (single stage). The manual step-by-step version follows.

### Stage 1 — A layer (deep learning)

```bash
# 1a. Train v5 (seed 2, 250 epochs, single GPU ~24h)
python 01_a_layer/train_v5.py --seed 2 --log_dir riconv_large_v5_s2
# produces: log/classification_shrec2025/riconv_large_v5_s2/checkpoints/{best_macrof1.pth,swa_model.pth,...}

# 1b. Dump MC-FPS (25 clean forward passes, mean-logit then softmax)
python 01_a_layer/dump_mcfps.py --log_dir riconv_large_v5_s2 \
       --ckpt swa_model.pth --K 25 --prefix mcfps_v5s2
# produces: val/test log-prob caches mcfps_v5s2* under results_final/cache/

# 1c. Dump v5 single forward (comparison baseline)
python 01_a_layer/dump_v5.py --log_dir riconv_large_v5_s2 \
       --ckpt swa_model.pth --tag s2_swa

# 1d. (optional) v3 baseline caches: with the committed best_model.pth,
#     used by split_sensitivity and rho(*,v3) in stage 4;
#     run_all.sh runs this automatically when the v3 ckpt is detected
python 01_a_layer/dump_clean.py --splits train,val
python 01_a_layer/dump_mcfps.py --K 25
```

**Checkpoint 3**: checkpoints and caches
```bash
ls log/classification_shrec2025/riconv_large_v5_s2/checkpoints/swa_model.pth   # should exist
ls results_final/cache/ | grep mcfps_v5s2                                       # cache files expected
```
> v5 default hyperparameters (`configs/v5.yaml`): Adam lr 1e-3 / wd 1e-4,
> CosineAnnealing T_max 245 / eta_min 1e-5 + 5-epoch warmup, focal gamma 1.0,
> cutmix 0.2, sqrt-inverse class weights clipped to 10:1, SWA from epoch 200.
> **Do not change.**

**Skip-training option**: to avoid retraining v5, copy in an existing
`swa_model.pth` and run 1b/1c directly. The v3 `best_model.pth` is committed,
so v3 requires no training.

### Stage 2 — C layer (classical descriptors)

All run scripts write to `results_final/<method>/` by default and eval scripts
read caches to report metrics; everything runs without arguments.

```bash
# 2a. 3DZD extraction + 1-NN (c1: 3D Zernike Descriptor, volume gate [0.8,1.2])
python 02_c_layer/run_3dzd.py     # --workers 36 --out results_final/3dzd (defaults)
python 02_c_layer/eval_3dzd.py
# 2b. FPFH extraction + RBF-SVM (c2: FPFH L2-normalized + decision_function margin)
python 02_c_layer/run_fpfh_o3d.py # --workers 12 --out results_final/fpfh
python 02_c_layer/eval_fpfh.py
# 2c. Guerra topology + SimpleNN 8-view hard voting (c3)
python 02_c_layer/run_guerra.py   # --workers 24 --out results_final/guerra
python 02_c_layer/eval_guerra.py   # --device cpu --max-epochs 5000
# 2d. SVM members c1d / c4 (c4 = [363 ZDZ | 612 FPFH] block-standardized concat + RBF-SVM C16/gamma.003)
python 02_c_layer/tune_svm_members.py
```

**Checkpoint 4**: C-layer outputs and reference metrics (frozen in `configs/c_members.yaml`)
```bash
ls results_final/   # expect 3dzd/ fpfh/ guerra/ etc.
```
| Member | Method | test acc / bal / mac |
| --- | --- | --- |
| c1 | 3DZD 1-NN + volume gate | 0.9100 / 0.8914 / 0.8867 |
| c2 | FPFH + RBF-SVM | 0.9052 / 0.8570 / 0.8401 |
| c4 | [ZDZ\|FPFH] + RBF-SVM | 0.9324 / 0.8633 / 0.8822 |

### Stage 3 — Probability-space late fusion

```bash
python 03_fusion/fuse_mcfps2.py
```
`fuse_mcfps2.py`: one-SE nested-K selection (K=25) on val, paired bootstrap
against the frozen production val predictions, and a single test report for the
winner. Fusion formula (`configs/fusion_final.yaml`):
```
p = 2*softmax(v5_mc_mean_logp) + 2*softmax(c1_dist/0.5) + 2*softmax(c2_margin/1.0)
```
> **The final production predictions are frozen in
> `expected_outputs/test_full_fusion.csv`**; the script never overwrites it.

### Stage 4 — Analysis and verification

```bash
python 04_analysis/compute_final_metrics.py      # verify final test numbers
python 01_a_layer/calibration_analysis.py        # ECE/AURC/BALD (needs 25-draw cache)
python 04_analysis/make_calibration_fig.py       # reliability + risk-coverage figure
<py_numpy2> 01_a_layer/deterministic_init_eval.py  # D0/Ds/random/K25 (needs numpy>=2)
python 01_a_layer/split_sensitivity.py           # split sensitivity (needs 1d v3 clean cache)
python 01_a_layer/member_overlap_rho.py          # member error overlap rho (skips missing caches)
python 04_analysis/make_framework_fig.py         # Fig. 1 framework diagram (pure plotting, no deps)
python 04_analysis/make_variance_fig.py          # MC-FPS per-pass variance
python 04_analysis/make_k_curve_fig.py           # nested-K val curve
python 04_analysis/make_winners_curse_fig.py      # winner's-curse illustration
python 04_analysis/make_confusion_matrix.py      # 97x97 confusion matrix
python 04_analysis/make_per_class_table.py       # LaTeX per-class table
```
Figures go to `figures/`; the per-class table prints to stdout and writes
`figures/per_class_table.tex`; calibration/deterministic-init summaries go to
`results_final/calibration_summary.json` and
`results_final/deterministic_init_summary.json`.

### Stage 5 — Cross-architecture replication (optional, ModelNet40)

Two external resources are required (not distributed in this package):
- **ModelNet40 (normal_resampled)**: public mirror
  `https://hf-mirror.com/datasets/Pointcept/modelnet40_normal_resampled-compressed`
  — download `modelnet40_normal_resampled.tar.gz` (~1.7 GB; resume truncated
  downloads with `curl -C -`; the tarball has no top-level directory, so
  extract with `mkdir modelnet40_normal_resampled && tar xzf ... -C modelnet40_normal_resampled`).
- **PointNet++ code and released weights**:
  `git clone https://gh-proxy.com/https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git`
  (the repo already contains the SSG/MSG `best_model.pth` files).

Run (needs an environment with numpy>=2 + torch):
```bash
MODELNET40_ROOT=<modelnet40_normal_resampled> \
POINTNET2_REPO=<Pointnet_Pointnet2_pytorch> \
SHARP_PYTHON=<py_numpy2> STAGE=5 ./run_all.sh
```

**Checkpoint 5 (final)**:
```bash
python 04_analysis/compute_final_metrics.py
# expect:
#   accuracy      : 94.79
#   balanced acc  : 91.48
#   macro-F1      : 92.25
#   PASS: metrics match the pre-registered target (tol=0.05).
```

**Checkpoint 6 (stage 5)**: `05_cross_arch/routeb_summary.json` should be close to
SSG: single acc .9271+-.0017 -> K25 .9279, flip .0535, ECE .0197->.0169;
MSG: .9239+-.0017 -> .9263, flip .0527, ECE .0397->.0393 (MI ~0.0048 nats).

---

## 4. Minimal verification path (no training; verify frozen results only)

If you only want to confirm that the frozen predictions reproduce the paper
numbers (no GPU/data/training needed):
```bash
cd replication
python -m pip install -r requirements.txt      # sklearn/pandas needed even on CPU
STAGE=4 ./run_all.sh        # or directly: python 04_analysis/compute_final_metrics.py
```
This path only reads `expected_outputs/test_full_fusion.csv` +
`00_data/splits/test_set_ground_truth.csv` and should report PASS.

---

## 5. Troubleshooting matrix

| Symptom | Cause | Fix |
| --- | --- | --- |
| `[guard] no *_pc.npy found under new_data/` | point clouds not in place | download/extract clouds into `new_data/` per Sec. 2.2 |
| `FileNotFoundError: log/.../swa_model.pth` | not trained yet or wrong path | run stage 1a, or copy an existing ckpt in; `SKIP_TRAIN=1` only works with an existing ckpt |
| MC-FPS differs across runs | FPS randomness (`riconv2_utils_features.py:88` `torch.randint`) | **expected behaviour**; production uses K=25 averaging to remove variance — never report a single draw |
| NaN from softmax, fusion collapses | c1 1-NN distances contain `inf` (volume gate rejected everything) | replace `inf` with a large finite value (e.g. `1e6` or the max finite distance) before softmax |
| SVM member drags fusion down / macro drops hard | Platt scaling used by mistake | **never** use `SVC(probability=True)`; always use `decision_function` margins (see `configs/c_members.yaml`) |
| c3 test accuracy ~1 point low | refit not done on the full 9244 rows | `c3`/`c3b` must refit on the **full 9244-row** training set, not only the 7428-row internal split (`clean_refit_on_full_9244: true`) |
| training underfits, test < 87.5% | aggressive class weights | use sqrt-inverse-frequency weights clipped to 10:1; never Class-Balanced beta=0.999 (ratio 800+:1) |
| val high but test rejects | stratified split used by mistake | `train_set_2.csv` row-order 7428/1816; **stratified splitting is forbidden** |
| small val gain flips on test | winner's curse | use the robust one-SE nested-K + paired bootstrap protocol, not a single lucky draw |
| `run_fpfh_o3d.py` ImportError | open3d not installed | `pip install -r requirements.txt` (includes open3d 0.13.0; latest for py3.8) |
| abnormal LR after resume | scheduler advanced by repeated `scheduler.step()` | resume with `last_epoch` at scheduler construction; never loop step |
| Route B/C forward wrong or assertion exit | numpy 1.x loading numpy 2.x pickle (silently corrupts) / version assertion | use a separate numpy>=2 + torch env, passed via `SHARP_PYTHON` |

---

## 6. Script / config quick reference

| Path | Purpose | Key arguments |
| --- | --- | --- |
| `01_a_layer/train_v5.py` | v5 training | `--seed 2 --log_dir riconv_large_v5_s2` (all defaults, see `configs/v5.yaml`) |
| `01_a_layer/dump_mcfps.py` | MC-FPS K forward passes | `--log_dir <ld> --ckpt swa_model.pth --K 25 --prefix mcfps_v5s2` |
| `01_a_layer/dump_v5.py` | single forward pass | `--log_dir <ld> --ckpt swa_model.pth --tag s2_swa` |
| `01_a_layer/split_sensitivity.py` | Sec. III-D split sensitivity | no args (reads v3 clean cache, no retraining) |
| `01_a_layer/member_overlap_rho.py` | Sec. V-B/E error overlap rho | no args (skips missing caches) |
| `04_analysis/make_framework_fig.py` | Fig. 1 framework diagram | no args (pure plotting, no data/GPU) |
| `02_c_layer/run_3dzd.py` | 3DZD extraction | `--workers 36 --out results_final/3dzd` (defaults) |
| `02_c_layer/eval_3dzd.py` | 3DZD 1-NN + gate | no args |
| `02_c_layer/run_fpfh_o3d.py` | FPFH extraction | `--workers 12 --out results_final/fpfh` |
| `02_c_layer/eval_fpfh.py` | FPFH + SVM | no args |
| `02_c_layer/run_guerra.py` | Guerra topology | `--workers 24 --out results_final/guerra` |
| `02_c_layer/eval_guerra.py` | SimpleNN 8-view | `--device cpu --max-epochs 5000` |
| `02_c_layer/tune_svm_members.py` | c1d/c4 SVM | no args |
| `03_fusion/fuse_mcfps2.py` | fusion (production entry) | no args (reads `results_final/cache`) |
| `01_a_layer/calibration_analysis.py` | Sec. V-C ECE/AURC/BALD | no args (needs 25-draw cache + pandas/sklearn) |
| `01_a_layer/deterministic_init_eval.py` | Sec. V-D D0/Ds control | `--gpu 0` (needs numpy>=2) |
| `05_cross_arch/route_b_eval.py` | Sec. V-E cross-architecture 25-draw | `--data_root <mn40> --repo_root <pn2> --K 25` (needs numpy>=2) |
| `04_analysis/compute_final_metrics.py` | final verification | no args |

| Config | Contents |
| --- | --- |
| `configs/v5.yaml` | lr 1e-3, wd 1e-4, T_max 245, eta_min 1e-5, warmup 5, focal gamma 1.0, cutmix 0.2, class_weight sqrt_inverse clip 10, swa_start 200, seeds [1,2] |
| `configs/c_members.yaml` | c1 (3DZD 1-NN + volume gate [0.8,1.2]) / c1d / c2 (FPFH + SVM C64 gamma 12) / c3 (Guerra) / c4 ([363\|612] concat + SVM C16 gamma .003), decision_function: true |
| `configs/fusion_final.yaml` | K=25, weights 2:2:2, c1 tau=.5, c2 tau=1.0; p = 2*softmax(v5_mc) + 2*softmax(c1/0.5) + 2*softmax(c2/1.0) |

---

## 7. Mapping to paper sections / tables

| Paper content | Replication location |
| --- | --- |
| Final fusion metrics (experiments Sec. 7) | `expected_outputs/test_full_fusion.csv` + `compute_final_metrics.py` |
| Single-model / fusion-level ablation (tab:ablation) | `01_a_layer` dumps + `03_fusion/fuse_mcfps2.py` recomputed at the same protocol |
| Winner's-curse illustration | `04_analysis/make_winners_curse_fig.py` -> `figures/winners_curse.pdf` |
| MC-FPS per-pass variance | `04_analysis/make_variance_fig.py` |
| nested-K val curve | `04_analysis/make_k_curve_fig.py` |
| Per-class accuracy / confusion | `make_per_class_table.py` / `make_confusion_matrix.py` -> `figures/confusion_matrix.pdf` |
| Split sensitivity (Sec. III-D) | `01_a_layer/split_sensitivity.py` (needs 1d v3 clean cache) |
| Member error overlap rho (Sec. V-B/E) | `01_a_layer/member_overlap_rho.py` -> 0.281/0.313/0.338/0.672/0.198 etc. |
| Fig. 1 framework diagram | `04_analysis/make_framework_fig.py` -> `figures/framework.pdf` |
| ECE/AURC/BALD (Sec. V-C) | `01_a_layer/calibration_analysis.py` + `04_analysis/make_calibration_fig.py` -> `figures/calibration.pdf` |
| Deterministic-init control (Sec. V-D) | `01_a_layer/deterministic_init_eval.py` -> `results_final/deterministic_init_summary.json` |
| Cross-architecture ModelNet40 (Sec. V-E) | `05_cross_arch/route_b_eval.py` -> `05_cross_arch/routeb_summary.json` |
