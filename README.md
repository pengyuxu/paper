# Reproduction Package

**Anonymous repository:** https://anonymous.4open.science/r/paper-0794

Reproduction package for *"Seed Sensitivity of Farthest-Point Sampling in
Set-Abstraction Point-Cloud Networks: Prevalence, Consequences, and a
Training-Free Remedy"* (SHREC 2025 protein shape classification track; two
dataset versions).

Everything in this package is **zero-transcription**: every number in every
table and figure is recomputed from the per-sample prediction caches
(`.npz` / `.json` / `.csv`) shipped here. No value is hand-entered.

---

## 1. What can be reproduced from this package alone

| Output | Command (from package root) | Runtime |
|---|---|---|
| All manuscript tables (paper + updated versions) | `python3 experiments/methods/make_manuscript_tables_dual.py` | < 1 min, CPU |
| Figs. 1, 2 (data-driven figures) | `python3 experiments/methods/regenerate_figures.py` | < 1 min, CPU |
| Fig. S1 (framework diagram) | `python3 experiments/methods/make_fig1_framework.py` | seconds |
| Fig. 3 (baseline comparison) | `python3 experiments/methods/make_fig3_method_comparison.py` | seconds |
| Graphical abstract | `python3 experiments/methods/make_graphical_abstract.py` | seconds |
| Headline-number audit | `python3 experiments/methods/make_number_audit.py` | seconds |
| Independent 108-row verification (npz → metrics, diff = 0) | `python3 experiments/methods/final_verification.py` | < 2 min, CPU |

Run all commands from the **package root** (scripts use
`experiments/...`-relative paths; `paths.py` resolves the repo root from its
own location).

`experiments/results/verification_report.csv` (shipped) is the reference
output of the verification stage: 108 rows, `PASS` on every row, expected vs
computed identical to 6 decimal places.

## 2. Package layout

```
repro_package/
├── README.md                        this file
├── requirements.txt                 pinned Python dependencies
├── experiments/
│   ├── common/                      metrics_lib, stats_lib (mc_aggregate),
│   │                                deterministic_fps, rot_tta, paths, diagnostics
│   ├── preprocess/                  VTK mesh -> 8192-pt cloud (build + verify)
│   ├── descriptors/                 3DZD / FPFH / Guerra-potential builders
│   ├── methods/                     dumps, member training/eval, fusion panels,
│   │                                table renderers, figures, audit, verification
│   ├── scheduler/                   training queue helper
│   ├── tests/                       FPS-determinism probe and GPU helpers
│   ├── run_env.sh                   CUDA/cuDNN LD_LIBRARY_PATH fix + seeds
│   └── results/                     ALL CACHES (see §4) + expected outputs
├── results/                         external baselines' per-sample submissions
│                                    (Tatsuma, Kagaya, Guerra, He, Tehrani, Yang,
│                                    Li, Peng + Barisin; Table 8 Block B source)
├── work/clouds/txt8/0_pc.npy        one illustrative 8192-pt cloud (graphical abstract)
└── figures/                         rendered PNGs (expected outputs)
```

## 3. Environment

Python 3.8; the pinned versions we used are in `requirements.txt`
(numpy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2, scipy 1.10.1,
matplotlib 3.7.5). Stages A–G below are CPU-only and need no torch;
torch 2.3.0+cu121 is only required if you want to regenerate the `.npz`
caches from checkpoints (Stage D needs a CUDA GPU).

If your machine has a system CUDA whose cuDNN mismatches the pip torch
build, `source experiments/run_env.sh` first (it strips the offending
`LD_LIBRARY_PATH` entries and fixes seeds).

## 4. Cache formats (the provenance contract)

All metrics in the paper are defined on these caches:

* `experiments/results/{paper,updated}/deep/*.npz` — per-checkpoint
  Monte-Carlo FPS dumps:
  * `scores`: float16, shape `(K, N, C)` — **per-pass log-softmax**
    (log-probabilities, NOT raw logits), K = number of FPS draws (25),
    N = samples, C = 97 or 99.
  * `labels`: int64 ground truth; `ids`: sample identifiers.
  * Aggregation (`experiments/common/stats_lib.py::mc_aggregate`):
    `logsumexp(pass_logp[:K], b=1.0/K, axis=0)` — the log-mean-exp of the
    per-pass log-softmax (probability-space averaging).
  * Metrics (`experiments/common/metrics_lib.py::evaluate`): sklearn
    accuracy / balanced accuracy / macro-F1 with
    `f1_score(average="macro", zero_division=0)` — averaged over the labels
    **actually present in `y_true`** (88 of 97 on the paper test split,
    90 of 99 on the updated test split). Note: sklearn's default macro-F1
    also averages over classes that appear only in `y_pred`; on the updated
    validation split one such phantom class (id 72) exists, which is why
    `final_verification.py` passes `labels=sorted(set(y_true))` explicitly.
* `experiments/results/{paper,updated}/tables/table78_fusion.csv` — the
  exact member weights and temperatures of every fusion panel (P7, P10, P11,
  Final). The Final panel is pre-registered and frozen: weights `[2,2,2]`,
  `tau_dzd = 0.5`, `tau_fpfh = 1.0`, identical on both versions; P7/P10/P11
  are validation-selected and therefore differ between versions.
* `experiments/results/{paper,updated}/deep/deep_analysis.json` — per-run
  metric curves (K = 1..25) for every deep member and split.
* `experiments/results/desc/`, `.../{paper,updated}/descriptors/` — 3DZD /
  FPFH / potential-feature caches for the classical members.
* `results/metrics_all_authors.csv` — the released per-author metric summary
  of the track (source of Table 8 Block A and of the reconciliation in
  Supplementary Table S4); `results/*/` — the released per-sample submission files of the
  external baselines, on the fixed class grid (Table 8 Block B).

## 5. Pipeline map

| Stage | Script(s) | Inputs | Outputs | In package? |
|---|---|---|---|---|
| A. Preprocess | `experiments/preprocess/build_pointclouds.py` | raw VTK meshes (`data_raw/`, not included) | `work/clouds/txt8/*_pc.npy` | code only; one illustrative cloud (`0_pc.npy`) is shipped for the graphical abstract |
| B. Descriptors | `experiments/descriptors/{dzd_build,fpfh,guerra_build}.py` | clouds from A | `experiments/results/desc/`, `.../descriptors/` | code + caches |
| C. Classical members | `experiments/methods/{dzd_c1,fpfh_c2,guerra_c3}.py` | B caches | member `.npz` | code + caches |
| D. Deep dumps | `experiments/methods/deep_dump.py` (+ RIConv++ code and checkpoints, third-party, not included) | checkpoints | `.../deep/*.npz` | code + caches |
| E. Fusion panels | `experiments/methods/fusion_panel.py` | C + D npz, weight/tau tables | `.../fusion/`, `table78_fusion.csv` | code + caches |
| F. Manuscript tables | `experiments/methods/make_manuscript_tables_dual.py` | C–E caches | `experiments/results/manuscript_tables_dual.md` (+ CSVs) | **fully reproducible** |
| G. Figures / audit / verification | see §1 | same caches | `figures/*.png`, `number_map.json`, `verification_report.csv` | **fully reproducible** |

Stages F–G run from the shipped caches with no external data. Stages A–D
regenerate the caches themselves and require the SHREC 2025 track release
(raw meshes; the updated-version mapping is in
`results/corr_prot_id_testset.csv` and
`experiments/results/preprocess_{train,test}_meta.csv`).

## 6. Table / figure map (manuscript numbering)

Post-review layout: the main text is the single-thread FPS diagnosis/remedy;
the fusion study is Supplementary Section S1 and reproducibility/provenance is Supplementary Section S2.

| Manuscript item | Renderer / source |
|---|---|
| Table 1 (dataset-version differences) | in-text, from the release notes |
| Table 2 (set-abstraction stage configuration) | in-text, from the backbone definition |
| Tables 3–5 (deep members single-draw/MC-FPS, calibration, deterministic control) | `make_manuscript_tables_dual.py` |
| Table 6 (ModelNet40 cross-architecture audit) | `make_table6.py` (from the ModelNet40 per-sample caches; carried over, see note N1) |
| Table 7 (rotational TTA) | `make_table_tta.py` |
| Table 8 (external baselines, Blocks A/B/C) | `make_table10.py` from top-level `results/` + `results/metrics_all_authors.csv` |
| Supplementary Tables S1–S3 (classical members, fusion panels, winner's-curse evidence) | `make_table78.py` / renderer, configs in `table78_fusion.csv` |
| Supplementary Table S4 (value-source reconciliation) | in-text, from `results/metrics_all_authors.csv` + per-sample recomputation |
| Supplementary Table S5 (bootstrap B = 500 vs B = 2,000) | `review_round8_boot.py` |
| Supplementary Table S6 (draw-level ordering stability) | `review_round8_fragility.py` |
| Supplementary Table S7 (training-recipe ablation, incl. the v3 identity of Section 4.2) | `make_table9.py` → `table9_ablation.csv`, then `make_manuscript_tables_dual.py` (from per-sample caches) |
| Supplementary Table S8 (class-size groups) | `make_table17.py` from per-sample caches |
| Figs. 1, 2 | `regenerate_figures.py` (from caches) — AURC = trapz(risk, coverage), no −0.5 offset; must equal Table 4 |
| Fig. 3 | `make_fig3_method_comparison.py` (from caches + `results/`) |
| Fig. S1 | `make_fig1_framework.py` (static diagram) |
| Fig. S2 | `regenerate_figures.py` (from Final-panel per-sample predictions) |

## 7. Known provenance limits (disclosed in manuscript Supplementary Section S2)

* The original compiled i8 Sobol helper and the predecessor fusion/analysis
  layer were not part of the released code; their substitutes
  (scipy Sobol, rebuilt descriptors) and consequences are documented in the
  manuscript (Supplementary Section S2) and in
  `experiments/results/reproducibility_checklist.md`.
* Block A integers of Table 8 come from the released per-author metric
  summary (approximate, decision A5; not regenerable from per-sample files);
  Supplementary Table S4 reconciles it with our recomputation per method.
* The deep checkpoints and the third-party RIConv++ training code are not
  redistributed here; Stage D therefore cannot be run from this package
  alone, but every downstream number is reproducible from the shipped dumps.
  The RIConv++ training code is available from its authors' public release
  (see manuscript reference [2]); our checkpoints are available from the
  corresponding author on reasonable request.
* This anonymized snapshot mirrors the reviewed manuscript state; a public,
  DOI-minted repository will replace it upon acceptance.

## 8. One-command sanity check

```bash
cd repro_package
python3 experiments/methods/final_verification.py
# -> experiments/results/verification_report.csv : 108 rows, PASS=108, FAIL=0
```
