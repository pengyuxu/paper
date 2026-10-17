# Reproducibility checklist — SHREC 2025 dual-version rerun

This document itemises every experiment-level parameter that must be known to reproduce the numbers reported in `PR-Pattern-Recognition-revised.md`. Every parameter below is traceable either to a released cache file or to a specific line of the rebuild code. Numbers come in `paper / updated` pairs where the two dataset profiles differ.

Anonymous repository (code + caches): https://anonymous.4open.science/r/paper-0794

## 1. Dataset versions

| Item | Paper version | Updated version |
|---|---|---|
| Protein families | 97 | 99 (family 16 split → classes 97 + 98) |
| Training meshes | 9,244 | 9,244 (same IDs) |
| Validation split | 1,816 (row-order split of training list) | 1,816 (identical rows) |
| Test meshes | 2,311 | 2,308 (ids 840, 1277, 1311 removed) |
| Classes present in test | 88 of 97 | 90 of 99 |
| Classes absent from test | {26, 42, 44, 50, 58, 63, 72, 77, 95} | identical nine IDs |
| CSV id column | `anonymised_protein_id` | `anonymized_protein_id` |

Source: `experiments/results/{paper,updated}/deep/v3_noSynth_s2__best_model__seeded42_K25_{val,test}.npz` (labels arrays), `PR-Pattern-Recognition-revised.md` §3.1.

## 2. Preprocessing

- VTK → STL (Trimesh), centre of mass translate, unit-ball scale.
- 8,192 points, area-weighted face sampling, **fixed seed = 42** (`trimesh.sample.sample_surface`).
- Per-point channels: xyz (geom), nx/ny/nz (face normal), φ (Coulomb potential, min-max per mesh), ψ (normal potential, min-max per mesh).
- Source: `experiments/methods/deep_dump.py` data loading pipeline.

## 3. Deep backbone (RIConv++)

- Architecture: rotation-invariant convolutions, channel multiplier 2, 5 set-abstraction stages (Table 1 of paper).
- **Nondeterminism source**: `riconv2_utils_features.py:88` → `farthest = torch.randint(0, N, ...)` at every SA stage, every forward call.
- Parameters: ~5.75M.
- Batch size 16, Adam (lr 10^-3, weight decay 10^-4), cosine annealing 245 steps to 10^-5 after 5 warm-up epochs, focal loss γ = 1.0, label smoothing 0.1, PointCutMix p = 0.2, sqrt-inverse class weights clipped 10:1.
- Training time: ~60 h / v5 run on single NVIDIA A40 (48 GB).

## 4. MC-FPS (Section 4.4)

- K = 25 clean no-rotation forward passes per input.
- **Aggregation** (critical for exact reproduction): each pass's logits are log-softmaxed per-pass → softmax(t=1.0) → mean over K passes → log (via `np.log(p.mean(0) + 1e-300)`). Implemented in `experiments/common/stats_lib.py:mc_aggregate`.
- Final prediction: argmax of mean probability vector.
- Pass seed strategy: fixed ordered seeds per checkpoint (not bootstrapped).
- Dump time: ~20 min per version (25 passes × 2,300 meshes × 0.5 s / pass).

## 5. Classical descriptors

### 3DZD (c1 family)
- Order 20, canonical c = 0.5 normalisation, Map2Zernike / Novotni-Bonn binary (`zernike3d`).
- 121-coefficient blocks × 3 (whole / +potential / −potential surfaces) + volume scalar = 363-dim + volume.
- Classifiers:
  - c1: 1-NN on negative volume gates d, gated by V_q / V_g ∈ [0.8, 1.2] (out-of-gate → worst finite distance).
  - c1b: inverse-distance k-NN, k selected on validation.
  - c1c: block-z-scored weighted 1-NN.
  - c1d: RBF-SVM `StandardScaler + SVC(C=50, γ=0.003)`, `probability=False`.

### FPFH (c2 family)
- Point sampling: 30,000 points, area-weighted, Sobol sequence (`scipy.stats.qmc.Sobol`, fixed seed).
  - Note: original i8 compiled Sobol helper unavailable; this is a substitute, likely contributing to weaker c2 values (§6.4).
- Normals: radius search r_n = 0.1, oriented toward origin.
- FPFH: Open3D `compute_fpfh` at r_f = 0.2.
- Pooled 612-dim signature: 33 channel means + 561 upper-triangular covariance entries + 16-bin potential histogram + mean + variance.
- Classifiers:
  - c2: RBF-SVM `SVC(C=64, γ=12)` on L2-normalised features, `probability=False`.
  - c2b: ExtraTrees `n_estimators=500, max_features=None, class_weight='balanced'`.
  - c2d: cosine 1-NN.

### Guerra (c3 family)
- 75-dim alpha block + 104-dim spherical-sector block.
- Cleaning: 8 always-zero columns dropped; block standardised; block members residualised against median column.
- After cleaning: 171 coordinates; 8 rotation-group sector permutations = 8 views.
- Refit discipline: validation models fit on 7,428 training rows only; test submission retrained from scratch with cleaning transforms refit on all 9,244 rows (including the 1,816 validation rows as unlabelled covariates).
- Classifiers:
  - c3: small MLP (171→111→100→C, C = 97 / 99, dropout 0.25, Adam 4×10^-4), 8-view hard majority vote.
  - c3b: RBF-SVM `SVC(C=32)` (default scale), `probability=False`.
- Note: Platt calibration (`probability=True`) was **never implemented** in the rebuild — excluded by design after the pre-rerun qualitative finding that it collapses under SHREC's severe class imbalance (Section 4.6.1).

### Concatenation (c4)
- 363 Zernike + 612 FPFH, block-standardised, concatenated.
- RBF-SVM `C=16, γ=0.003`, no Guerra block.

## 6. Fusion protocol

- Temperature softmax over each member's raw score (margins for SVMs, negative distances for 1-NN): `q_m = softmax(s_m / τ_m)`.
- Fused prediction: `argmax Σ_m w_m q_m(x)`, w_m ∈ {0, 1, 2}.
- Weight search: coarse regularised exhaustive enumeration via a single CPU GEMM (weight matrix × stacked member probability tensor), no continuous weight fitting.
- Temperature grids: τ_3DZD ∈ {0.5, 1, 2, 4, 8}, τ_FPFH ∈ {0.5, 1, 2}.
- All selections **exclusively** on 1,816 validation rows; test labels never consulted.

### Validated panel weights and temperatures (source: table78_fusion.csv)

| Panel | paper weights | paper τ_dzd/τ_fpfh | updated weights | updated τ_dzd/τ_fpfh |
|---|---|---|---|---|
| P7 argmax | [1,1,2,1,2,2,2] | 8.0 / 0.5 | [0,1,1,2,2,1,2] | 8.0 / 0.5 |
| P7 one-SE | [0,1,0,1,2,0,2] | 8.0 / 0.5 | [0,0,0,1,2,0,1] | 2.0 / 0.5 |
| P10 argmax | [1,1,2,2,2,2,2,1,0,0] | 8.0 / 0.5 | [1,1,2,2,2,1,2,0,0,0] | 8.0 / 0.5 |
| P10 one-SE | [0,1,0,1,2,2,2,0,0,0] | 8.0 / 0.5 | [0,0,0,1,2,0,1,0,0,0] | 8.0 / 2.0 |
| P11 argmax | [0,0,2,0,0,0,1,1,0,0,1] | 0.5 / 1.0 | [2,0,2,0,0,0,1,0,0,0,2] | 0.5 / 2.0 |
| P11 one-SE | [0,0,2,0,0,0,1,0,0,0,1] | 0.5 / 1.0 | [0,0,1,0,0,0,1,0,0,0,1] | 0.5 / 2.0 |
| **Final (frozen)** | **[2,2,2]** | **0.5 / 1.0** | **[2,2,2]** | **0.5 / 1.0** |

Note on Final: this panel was **pre-registered** before the dual-version rerun, so weights and τ are **identical** across paper and updated versions. P7/P10/P11 are **validation-select panels** whose weights/τ correctly differ between versions (different data → different validation argmax).

### Top-30 fusion-scan configs (paper version): τ_FPFH = 0.5 appears 30 / 30 times.

## 7. Evaluation metrics

All computed with scikit-learn on the fixed 97-class / 99-class label grid.

| Metric | Scikit-learn call | Note |
|---|---|---|
| Accuracy | `accuracy_score(y_true, y_pred)` | — |
| Balanced accuracy | `balanced_accuracy_score(..., adjusted=False)` | — |
| Macro-F1 | `f1_score(..., average='macro')` with **auto-discovered labels** | ⚠️ Averages over classes **actually present** in y_true, not over all 97/99 possible classes. paper test has 88/97, updated test has 90/99 — macro-F1 denominator is 88 or 90, not 97/99. |
| ECE | 15 equal-width bins | `sklearn.metrics.expected_calibration_error` |
| AURC | Risk-coverage curve integral | Custom (Section 5.3) |
| BALD MI | `H(p̄) − 1/K Σ_k H(p_k)` nats | Custom |

Macro-F1 auto-discovered labels was verified: direct call to `evaluate(logp, y, nc)` from `experiments/common/metrics_lib.py` with nc=97 returns **acc=.9394, bal=.8986, mac=.8884** for paper v3 K25 val — matches Table 2. Calling with `labels=list(range(97))` forces all 97 classes and gives mac=.7877, a 10pp lower value that would not match the paper.

## 8. Statistical tests

| Test | Parameter | Implementation | Source |
|---|---|---|---|
| K-scan (Section 4.4) | B = 300 bootstrap resamples of 1,816 validation rows | For each K ∈ {1,2,3,5,10,25}: resample K FPS passes with replacement from 25 available passes; compute metric; report mean ± SE. Best K = max; chosen K = smallest K within 1 SE. | `stats_lib.py:one_se_k`, seed=67890 |
| Paired bootstrap (§4.10, §5.10) | B = 500 paired resamples of validation indices | Recompute paired difference δ* of macro-F1 / accuracy. Report P(δ*>0) and 95% CI (percentile method). Adoption threshold: P ≥ 0.8 and lower bound ≥ −0.002. | `make_manuscript_tables_dual.py` inline |
| Final test acc CI (§5.12) | B = 500 resamples | 95% CI half-width | Renderer stdout: halfwidth 0.0104 both versions |
| FPS-gap bootstrap | Same | Per-sample max logit gap across draws | Section 4.3 |

## 9. Software and hardware

| Item | Value |
|---|---|
| Hardware | 2 × NVIDIA A40 (48 GB each), 48 CPU cores, 62 GB RAM |
| Python | 3.8 |
| PyTorch | 2.3.0 + CUDA 12.1 |
| scikit-learn | 1.3.2 |
| NumPy | 1.24.4 |
| Open3D | 0.13.0 |
| VTK | 9.2 |
| Trimesh | 4.5 |
| SVM Platt | Never implemented — hardcoded `probability=False` in classical descriptor codes |

## 10. Output artifact map

| File | Purpose |
|---|---|
| `experiments/results/{paper,updated}/deep/*.npz` | Per-checkpoint MC-FPS log-softmax predictions (K, N, C) float16 + labels + ids |
| `experiments/results/{paper,updated}/fusion/fusion_summary.json` | P11 argmax / P11 one-SE / P11 SE / Final panel metrics |
| `experiments/results/{paper,updated}/tables/table78_fusion.csv` | All panel weights, taus, val/test metrics |
| `experiments/results/{paper,updated}/tables/table17_groupsize.csv` | Class-size group accs |
| `experiments/results/{paper,updated}/fusion/p7_scan_top2000.csv` | Top-2000 p7 scan configs |
| `experiments/results/manuscript_tables_dual.md` | Authoritative rendered tables (single source of truth) |
| `experiments/results/manuscript_stats.json` | Bootstrap values, τ_fpfh top-30 half, ρ overlaps |
| `experiments/results/paper_numbers_audit.md` | Audit report — consistency: **PASS** |
| `experiments/results/number_map.json` | 29 headline metrics with exact cache values |

## 11. Items NOT rerun (qualitative or excluded)

| Item | Reason | Reference |
|---|---|---|
| Split-sensitivity (stratified 80/20 vs row-order) | Original `a_layer/split_sensitivity.py` unavailable | Section 3.4.1 (rewritten qualitative) |
| Platt margin-vs-probability comparison | Never implemented in rebuild | Section 4.6.1 (rewritten qualitative) |
| ModelNet40 PointNet++ audit | External benchmark, unaffected by SHREC versioning | Table 5 note N1 |
| Block A integer track values | Read from organisers' bar charts, no per-sample release | Table 11 Block A (decision A5) |

## 12. Verification history (this round)

| Verification | Method | Result |
|---|---|---|
| All 319 manuscript_tables_dual pairs appear in paper | regex extraction + set diff | missing: 0 |
| ρ overlaps (5 values) | `rho_overlaps()` from cache | all 6 decimals match |
| v3 K25 val acc/bal/mac | `mc_aggregate` + `evaluate` from cache | exact match .9394/.8986/.8884 |
| Final panel weights/τ | direct CSV read | [2,2,2] / (0.5, 1.0) both versions |
| P10 weights differ between versions | direct CSV read | paper [1,1,2,2,2,2,2,1,0,0], updated [1,1,2,2,2,1,2,0,0,0] |
| Final−P11 / Final−v3K25 bootstrap | renderer stdout | exact match |
| Audit | `make_number_audit` | **consistency: PASS** |
| Platt in rebuild | grep `platt\|probability.*True` + table scans | zero hits — design-excluded |
