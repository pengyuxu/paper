# SHREC 2025 Protein Shape Classification — Replication Package

> [中文版 (Chinese)](README_ZH.md)

A **self-contained** replication package for all experiments in the paper on the
SHREC 2025 protein shape classification track (97 classes).
Final fused-system **test-set** metrics:

| Metric | Value |
| --- | --- |
| Accuracy | **94.79 %** |
| Balanced accuracy | **91.48 %** |
| Macro-F1 | **92.25 %** |

The system is a three-way probability-space late fusion: one deep signal
(RIConv++ v5, SWA + Monte-Carlo FPS averaged over 25 draws) and two classical
signals (3DZD 1-NN + volume gate, FPFH + RBF-SVM margin). All frozen
hyperparameters are in `configs/`.

> **Step-by-step runbook**: see [REPRODUCE.md](REPRODUCE.md) (preflight
> checklist / checkpoints / troubleshooting matrix / script reference). This
> file is the package overview.

> **Self-contained design**: every script uses
> `BASE = two levels above the script = package root`.
> Therefore **run all commands from the package root `replication/`**
> (`cd replication/`). The data directories `new_data/`, `log/`,
> `results_final/` are already created inside the package; place generated or
> downloaded data there directly — no script path edits needed.

---

## 1. Directory layout

```
replication/
├── README.md                    # this file
├── run_all.sh                   # 5-stage end-to-end pipeline (with preflight guards, per-stage skipping)
├── requirements.txt             # pinned dependencies
├── configs/                     # frozen hyperparameters
│   ├── v5.yaml                  # deep training (lr 1e-3, T_max 245, focal gamma 1.0, cutmix 0.2, swa 200, seed 2)
│   ├── c_members.yaml           # c1-c4 classical member configs + test metrics
│   └── fusion_final.yaml        # K=25, weights 2:2:2, c1 tau=.5, c2 tau=1.0
├── 00_data/                     # data preparation
│   ├── DOWNLOAD.md              # VTK / point clouds / splits download notes
│   ├── extract_pc_fast.py       # extract 8192x8 point clouds from VTK (scipy cKDTree acceleration)
│   └── splits/                  # split CSVs (committed, do not move)
│       ├── train_set_2.csv      # 9244 rows: first 7428 train / last 1816 val
│       ├── test_set_2.csv       # 2321 rows: anonymised_protein_id
│       └── test_set_ground_truth.csv  # 2321 rows: anonymised_protein_id, class_id
├── csv -> 00_data/splits        # symlink: scripts read splits via BASE/csv/
├── 01_a_layer/                  # A layer (deep learning)
│   ├── train_v5.py              # v5 training (250 epochs, single GPU ~24h)
│   ├── train_v3.py              # v3 training (baseline / ablation reference)
│   ├── dump_v5.py               # single-pass log-prob cache
│   ├── dump_mcfps.py            # MC-FPS: K random forward passes, mean-logit averaging
│   ├── dump_clean.py            # v3 clean single dump (for split analysis)
│   ├── split_sensitivity.py     # Sec. III-D: stratified vs row-order split sensitivity (no retraining)
│   ├── calibration_analysis.py  # Sec. V-C: ECE / AURC / BALD MI (reads 25-draw cache; needs pandas/sklearn)
│   ├── deterministic_init_eval.py # Sec. V-D: D0/Ds/random/MC-FPS comparison (needs numpy>=2)
│   ├── member_overlap_rho.py    # Sec. V-B/E: member error-set Jaccard rho (skips missing caches)
│   ├── models/                  # riconv2_cls_v2_features2_largest.py + riconv2_utils_features.py
│   ├── data_utils/              # ProteinToPointCloudProcessor(.py / Test.py)
│   └── checkpoints/DOWNLOAD.md  # v3 best_model.pth info; v5 swa is produced by training
├── 02_c_layer/                  # C layer (classical descriptors + fusion logic)
│   ├── run_3dzd.py / eval_3dzd.py        # c1: 3DZD 1-NN + volume gate
│   ├── run_fpfh_o3d.py / eval_fpfh.py    # c2: FPFH + RBF-SVM (decision_function)
│   ├── run_guerra.py / eval_guerra.py    # c3: Guerra topology + SimpleNN 8-view hard voting
│   ├── tune_svm_members.py               # c1d / c4 SVM members
│   ├── tune_3dzd*.py / tune_fpfh.py / eval_*.py  # remaining tuning / evaluation
│   └── external/                        # third-party baseline sources (tehrani / guerra / shrec2025_protein / SHREC2025)
├── 03_fusion/                   # probability-space late fusion
│   ├── fuse_c.py                # classical member fusion + grid search
│   ├── fuse_v5.py               # v5 single-signal fusion
│   ├── fuse_mcfps.py            # MC-FPS deep-signal fusion
│   └── fuse_mcfps2.py           # one-SE nested-K selects K=25 + paired bootstrap (production entry)
├── 04_analysis/                 # analysis and figures
│   ├── compute_final_metrics.py # verify frozen predictions -> 94.79/91.48/92.25
│   ├── make_framework_fig.py    # Fig. 1 framework diagram (pure plotting, no data/GPU)
│   ├── make_variance_fig.py     # MC-FPS per-pass variance
│   ├── make_k_curve_fig.py      # nested-K validation curve
│   ├── make_winners_curse_fig.py# winner's-curse illustration
│   ├── make_calibration_fig.py  # Sec. V-C: reliability + risk-coverage single-column figure
│   ├── make_confusion_matrix.py # 97x97 confusion matrix
│   └── make_per_class_table.py  # LaTeX per-class table
├── 05_cross_arch/               # optional: cross-architecture replication (ModelNet40 / PointNet++ SSG+MSG)
│   ├── route_b_eval.py          # 25-draw single vs MC-FPS: acc/F1/flip/ECE/AURC/MI (numpy>=2)
│   └── routeb_summary.json      # frozen results (SSG/MSG)
├── expected_outputs/
│   └── test_full_fusion.csv     # frozen production predictions (2321 rows, verified)
├── new_data/                    # [COMMITTED] 2321 test point clouds (~1.2 GB); train clouds can be added via DOWNLOAD.md link
├── log/                         # [COMMITTED] v3 checkpoint + training logs (run1/run2, ~138 MB)
│   └── classification_shrec2025/
│       ├── riconv_large_shrec_run1_8k_largest/  # v2 baseline (best_model.pth + log + src)
│       └── riconv_large_shrec_run2_8k_largest/  # v3 best_model.pth + log + src (MC-FPS reference)
├── results_final/               # [placeholder] cache and CSVs produced by dumps/fusion
└── figures/                     # [placeholder] figure scripts write here
```

---

## 2. Environment

Dependencies are pinned to the **actual training machine** (see `requirements.txt`):

- **Python** 3.8.10 (system `python`; `python3.11` also works with matching deps)
- **PyTorch** 2.3.0 + **cu121** (matches the local NVIDIA driver 535.161.07 / max CUDA 12.2)
- **open3d** 0.13.0 (FPFH extraction in `run_fpfh_o3d.py`; latest version installable on py3.8)
- **scikit-learn** 1.3.2, **matplotlib** 3.7.5, **pandas** 2.0.3, **numpy** 1.24.4, **scipy** 1.10.1
- **trimesh** 4.12.2, **point-cloud-utils** 0.34.0, tqdm 4.66.4

> An earlier plan pinned torch 2.8.0+cu126, but that requires driver >=545
> (CUDA 12.6), which this machine does not support; all paper experiments were
> run under torch 2.3.0+cu121, so that is pinned here.
> **GPU**: 2x NVIDIA A40 (46 GB); train with `--gpu 0` (GPU1 is often busy).
>
> **Optional second environment (Routes B/C)**: the deterministic-init control
> (`deterministic_init_eval.py`) and cross-architecture replication
> (`05_cross_arch/route_b_eval.py`) require **numpy>=2** (the checkpoints are
> numpy 2.x pickles; numpy 1.x deserialization silently corrupts the forward
> pass). The scripts themselves only need numpy+torch. Use a separate
> environment (e.g. Python 3.13 + torch 2.8) and pass it via
> `SHARP_PYTHON=<interpreter>`.

---

## 3. Installation

Install with system Python 3.8 (open3d included, nothing extra needed):

```bash
cd replication
python -m pip install -r requirements.txt        # add --user if permissions require
# verify
python -c "import torch,sklearn,open3d,trimesh; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
```

> Or use a venv: `python3.8 -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt`.

---

## 4. Data preparation

See `00_data/DOWNLOAD.md` for details. Summary:

1. **VTK raw meshes**: download `train_set.tar.xz`, `test_set.tar.xz` from the SHREC official site.
2. **Point clouds** (`new_data/`, each file `{basename}_pc.npy` of shape `(8192, 8)`,
   channels `[x, y, z, nx, ny, nz, potential, normal_potential]`):
   **already committed** — 2321 test point clouds (~1.2 GB), works out of the box;
   train point clouds (9244) can be added via the link in `00_data/DOWNLOAD.md`.
3. **Split CSVs** are committed under `00_data/splits/`; the `csv/` symlink points there.
4. **Pretrained checkpoints**: v3 `best_model.pth` is **already committed** at
   `log/classification_shrec2025/riconv_large_shrec_run2_8k_largest/checkpoints/`;
   v5 `swa_model.pth` is produced by stage-1a training
   (see `01_a_layer/checkpoints/DOWNLOAD.md`).

---

## 5. Replication pipeline (5 stages)

Pipeline direction: `00_data -> 01_a_layer -> 02_c_layer -> 03_fusion -> 04_analysis`.
**Run all commands from the `replication/` root.** One-shot runs via
`run_all.sh` (per-stage skipping, see Sec. 6).

### Stage 0 — Data in place
`new_data/` has point clouds and `00_data/splits/` has the CSVs — nothing to run.

### Stage 1 — A layer (deep learning)
```bash
# 1a. Train v5 (seed 2, 250 epochs, single GPU ~24h)
python 01_a_layer/train_v5.py --seed 2 --log_dir riconv_large_v5_s2
# produces log/classification_shrec2025/riconv_large_v5_s2/checkpoints/swa_model.pth

# 1b. Dump MC-FPS (25 clean forward passes, mean-logit then softmax)
python 01_a_layer/dump_mcfps.py --log_dir riconv_large_v5_s2 \
       --ckpt swa_model.pth --K 25 --prefix mcfps_v5s2

# 1c. Dump v5 single forward (comparison baseline)
python 01_a_layer/dump_v5.py --log_dir riconv_large_v5_s2 \
       --ckpt swa_model.pth --tag s2_swa
```
> To skip retraining v5, copy an existing trained `log/.../swa_model.pth` in,
> or use the committed v3 `best_model.pth` via the v3 MC-FPS path.

### Stage 2 — C layer (classical descriptors)
```bash
# 2a. 3DZD extraction + 1-NN (c1)
python 02_c_layer/run_3dzd.py
python 02_c_layer/eval_3dzd.py
# 2b. FPFH extraction + RBF-SVM (c2)
python 02_c_layer/run_fpfh_o3d.py
python 02_c_layer/eval_fpfh.py
# 2c. Guerra topology + SimpleNN 8-view hard voting (c3)
python 02_c_layer/run_guerra.py
python 02_c_layer/eval_guerra.py
# 2d. SVM members (c1d, c4 = [363 ZDZ | 612 FPFH] block-standardized concat + RBF-SVM C16/gamma.003)
python 02_c_layer/tune_svm_members.py
```

### Stage 3 — Fusion
```bash
python 03_fusion/fuse_mcfps2.py
```
`fuse_mcfps2.py` runs one-SE nested-K selection (K=25) on val, a paired
bootstrap against the frozen production val predictions, and emits a single
test report for the winner. **The final production predictions are frozen in
`expected_outputs/test_full_fusion.csv`** — the script never overwrites it.

### Stage 4 — Analysis / verification
```bash
python 04_analysis/compute_final_metrics.py      # verify final test numbers
python 01_a_layer/calibration_analysis.py        # Sec. V-C: ECE/AURC/BALD (needs 25-draw cache)
python 04_analysis/make_calibration_fig.py       # reliability + risk-coverage figure
<py313> 01_a_layer/deterministic_init_eval.py    # Sec. V-D: D0/Ds/random/K25 (needs numpy>=2)
python 01_a_layer/split_sensitivity.py           # Sec. III-D split sensitivity (needs v3 clean cache)
python 01_a_layer/member_overlap_rho.py          # Sec. V-B/E member error overlap rho
python 04_analysis/make_framework_fig.py         # Fig. 1 framework diagram (no deps)
python 04_analysis/make_variance_fig.py          # MC-FPS per-pass variance
python 04_analysis/make_k_curve_fig.py           # nested-K val curve
python 04_analysis/make_winners_curse_fig.py      # winner's-curse illustration
python 04_analysis/make_confusion_matrix.py      # 97x97 confusion matrix
python 04_analysis/make_per_class_table.py       # LaTeX per-class table
```
Figures go to `figures/`; the per-class table prints to stdout. Calibration /
deterministic-init summaries go to `results_final/calibration_summary.json`
and `deterministic_init_summary.json`.

### Stage 5 — Cross-architecture replication (optional, ModelNet40)
```bash
# External resources (download notes in REPRODUCE.md):
#   MODELNET40_ROOT = modelnet40_normal_resampled directory
#   POINTNET2_REPO  = yanx27/Pointnet_Pointnet2_pytorch clone (contains released checkpoints)
MODELNET40_ROOT=<dir> POINTNET2_REPO=<dir> \
    SHARP_PYTHON=<py313> STAGE=5 ./run_all.sh
```
Results are written to `05_cross_arch/routeb_summary.json` (SSG/MSG: label-flip
~5.3%, MC-FPS consistently improves accuracy/ECE/AURC).

---

## 6. One-shot runs (run_all.sh)

```bash
cd replication
./run_all.sh                  # default: all 5 stages (incl. training, ~24h+)
SKIP_TRAIN=1 ./run_all.sh    # skip training (when a checkpoint already exists)
STAGE=4 ./run_all.sh          # only stage 4: verify frozen predictions
SHARP_PYTHON=<py313> STAGE=4 ./run_all.sh   # incl. deterministic-init control
MODELNET40_ROOT=<dir> POINTNET2_REPO=<dir> STAGE=5 ./run_all.sh  # cross-architecture
```
`run_all.sh` has preflight guards: it checks that `new_data/` contains point
clouds and that checkpoints exist, and warns instead of running blind.

---

## 7. Expected output verification

`expected_outputs/test_full_fusion.csv` holds the frozen production predictions
(2321 rows, columns `anonymised_protein_id, predicted_label`).
Running `04_analysis/compute_final_metrics.py` should give:

```
accuracy=94.79  balanced=91.48  macro-F1=92.25   PASS
```

---

## 8. Critical pitfalls (must follow)

- **FPS randomness**: farthest-point sampling uses a random start at every
  set-abstraction layer (`riconv2_utils_features.py` line 88, `torch.randint`),
  so a single clean forward pass is random — **this is expected behaviour**.
  The production signal averages 25 MC-FPS draws to remove the variance;
  **never report a single forward pass as the final result**.

- **Guerra clean refit**: `c3`/`c3b` must be **refit on the full 9244-row
  training set**, not only on the 7428-row internal train split. Missing the
  refit silently costs about 1 point of test accuracy.
  `clean_refit_on_full_9244: true` is recorded in `configs/c_members.yaml`.

- **c1 inf-fill of distances**: the 3DZD 1-NN distance array contains `inf`
  where the volume gate rejects every gallery item. Before
  `softmax(c1_dist / tau)`, replace `inf` with a large finite value (e.g. the
  largest finite distance or `1e6`), otherwise softmax yields NaN and the whole
  fusion collapses.

- **No Platt scaling for SVMs**: **do not** use `SVC(probability=True)`
  (Platt scaling). On this 97-class imbalanced task Platt probability
  calibration is poor and drags the fusion down. All SVM members use
  `decision_function` margins (`decision_function: true` in
  `configs/c_members.yaml`).

- **Mild class weights**: use sqrt-inverse-frequency weights clipped to a
  max/min ratio of 10:1. Aggressive weights (e.g. Class-Balanced beta=0.999,
  ratio 800+:1) underfit — test accuracy < 87.5%.

- **Row-order split**: `train_set_2.csv` first 7428 rows train, last 1816 val;
  **stratified splitting is forbidden** (stratification inflates validation
  class accuracy, producing optimistic bias that the test set rejects).

- **Guard against candidate-space inflation**: when the fusion candidate space
  grows exponentially with members, the "winner's curse" appears — small val
  advantages can flip on test. Hence the final protocol is one-SE nested-K +
  paired bootstrap, not a single lucky draw.

---

## 9. Mapping to paper tables/sections

- Final fusion metrics (Sec. 7) -> `expected_outputs/test_full_fusion.csv` + `compute_final_metrics.py`
- Single-model / fusion-level ablation (tab:ablation) -> `01_a_layer` dumps + `03_fusion/fuse_mcfps2.py` recomputed at the same protocol
- Winner's-curse illustration -> `04_analysis/make_winners_curse_fig.py`
- MC-FPS variance -> `04_analysis/make_variance_fig.py`
- Per-class accuracy -> `04_analysis/make_per_class_table.py` / `make_confusion_matrix.py`
- ECE / AURC / BALD (Sec. V-C) -> `01_a_layer/calibration_analysis.py` + `04_analysis/make_calibration_fig.py`
- Deterministic-init control (Sec. V-D) -> `01_a_layer/deterministic_init_eval.py`
- Cross-architecture ModelNet40 (Sec. V-E) -> `05_cross_arch/route_b_eval.py` + `05_cross_arch/routeb_summary.json`
