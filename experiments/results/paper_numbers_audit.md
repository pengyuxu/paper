# Paper number <-> cache cross-reference (Task 18)

Every number in the revised manuscript is generated from a cache; no result is hand-typed. Rounding in the prose is to 3-4 decimals of the values below.

## Table 3 (single/MC-FPS K sweep)
- artifact: `experiments/results/<version>/tables/table2_mc.csv`
- generator: `python -m experiments.methods.deep_analysis --version {v}`
- cache lineage: deep/*seeded42_K25*.npz, deep/*first42_K1*.npz

## Table 4 (ECE/AURC/BALD/flip)
- artifact: `experiments/results/<version>/tables/table3_diag.csv`
- generator: `python -m experiments.methods.deep_analysis --version {v}`
- cache lineage: deep/*seeded42_K25*.npz via diagnostics.all_diagnostics

## Table 5 (D0/Ds/random/K25 modes)
- artifact: `experiments/results/<version>/tables/table4_modes.csv`
- generator: `python -m experiments.methods.deep_analysis --version {v}`
- cache lineage: deep/*first42_K1 (D0), deep/*gen42_K1 (Ds), deep/*seeded42_K25

## Table 7 (rotational TTA-25)
- artifact: `experiments/results/<version>/tables/table_tta.csv`
- generator: `python -m experiments.methods.make_table_tta --version {v}`
- cache lineage: deep/v3_noSynth_s2__best_model__tta25_{val,test}.npz

## Supplementary Table S1 (classical members)
- artifact: `experiments/results/<version>/tables/table6_classical.csv`
- generator: `python -m experiments.methods.make_table6 --version {v}`
- cache lineage: descriptors/{dzd_c1,fpfh_c2,c4,guerra_*}_metrics.json

## Supplementary Tables S2-S3 (fusion panels / winner's curse)
- artifact: `experiments/results/<version>/tables/table78_fusion.csv`
- generator: `make_table78 + shrec-table-closeout/refresh_table78.py`
- cache lineage: fusion/fusion_summary.json + 2025-10-01 classical panels

## Supplementary Table S7 (training-recipe ablation)
- artifact: `experiments/results/<version>/tables/table9_ablation.csv`
- generator: `python -m experiments.methods.make_table9 --version {v}`
- cache lineage: 4 abl/v3 best_model K25 npz + dzd_c1 + fpfh_c2

## Table 8 (cross-method comparison)
- artifact: `experiments/results/<version>/tables/table10_crossmethod.csv`
- generator: `python -m experiments.methods.make_table10 --version {v}`
- cache lineage: history/*grid.csv + barisin json + fusion_summary.json

## Supplementary Table S8 (accuracy by class size)
- artifact: `experiments/results/<version>/tables/table17_groupsize.csv`
- generator: `python -m experiments.methods.make_table17 --version {v}`
- cache lineage: v3 K1/K25 + v5 K25 + c1/c2 + train_set_2.csv

## Fig. S2 (per-class / confusion)
- artifact: `experiments/results/<version>/tables/perclass_final.csv, confusion_final.csv`
- generator: `python -m experiments.methods.make_table17 --version {v}`
- cache lineage: Final panel = v5_s2_swa K25 + dzd_c1 + fpfh_c2

## Headline numbers (abstract / conclusions)

| key | paper | updated | original manuscript (baseline only) |
|---|---|---|---|
| flip_val | 0.0286 | 0.0281 | 0.1590 |
| flip_test | 0.0281 | 0.0269 | — |
| v3_val_singlemac | 0.8672 | 0.8731 | 0.8620 |
| v3_val_k25mac | 0.8884 | 0.8902 | 0.8940 |
| v3_test_singlemac | 0.8791 | 0.8958 | — |
| v3_test_k25acc | 0.9381 | 0.9437 | — |
| v3_test_k25bal | 0.8762 | 0.8975 | — |
| v3_test_k25mac | 0.8808 | 0.9026 | — |
| bald_val | 0.0327 | 0.0312 | 0.0350 |
| bald_test | 0.0322 | 0.0305 | — |
| ece_val_k25 | 0.0176 | 0.0095 | — |
| ece_test_k25 | 0.0118 | 0.0148 | — |
| aurc_val_k25 | 0.0087 | 0.0101 | — |
| aurc_test_k25 | 0.0100 | 0.0069 | — |
| v5s2_k25_test_acc | 0.9455 | 0.9445 | — |
| v5s2_k25_test_bal | 0.9100 | 0.9028 | — |
| v5s2_k25_test_mac | 0.9174 | 0.9079 | 0.8991 |
| final_val_acc | 0.9427 | 0.9433 | — |
| final_val_bal | 0.8893 | 0.8923 | — |
| final_val_mac | 0.8841 | 0.8892 | — |
| final_test_acc | 0.9347 | 0.9376 | 0.9479 |
| final_test_bal | 0.8631 | 0.8683 | 0.9148 |
| final_test_mac | 0.8738 | 0.8781 | 0.9225 |
| p11_argmax_test_mac | 0.8899 | 0.9073 | — |
| p11_onse_test_mac | 0.9047 | 0.8970 | — |
| p7_test_mac | 0.8495 | 0.8409 | 0.8968 |
| tta_test_id_mac | 0.8835 | 0.8988 | — |
| tta_test_k25_mac | 0.8976 | 0.9047 | — |
| tta_test_k25_acc | 0.9429 | 0.9445 | — |

## Cross-file consistency assertions

PASS: final panel identical across fusion_summary / table78 / table10; v3 K25 mac identical across deep_analysis / table9 for both versions.

## Reproduction deltas (see manuscript Supplementary Section S2)
The original results_final caches and the a_layer classical code were unavailable; all rerun numbers come from regenerated 8,192-point clouds, retrained checkpoints, and rebuilt FPFH/3DZD descriptors. Qualitative findings reproduce; absolute levels differ and are disclosed in Sec. 6.4.
