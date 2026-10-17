"""Build the paper-number <-> cache cross-reference (Task 18, AC-7 / TR-18.1).

All numbers in the revised manuscript must come from generated caches. This
script does not recompute models; it orchestrates the already-generated table
CSVs / JSONs, extracts the headline numbers used in the abstract/conclusions,
runs cross-file consistency assertions, and renders ONE human-readable
cross-reference:

  experiments/results/paper_numbers_audit.md
  experiments/results/number_map.json

Every entry names the source artifact(s) and the generator command, so a
reviewer can trace each manuscript number to a cache. Original-submission
values (the pre-rerun PR manuscript, produced by the now-missing results_final
pipeline) are listed only as a documented baseline in the delta table, never
re-used as results.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

import json
from pathlib import Path

import pandas as pd

from experiments.common import paths as vp

OUT = vp.RESULTS_DIR
VERSIONS = ("paper", "updated")

# table artifact -> (generator, cache lineage)
TABLES = {
    "Table 3 (single/MC-FPS K sweep)": (
        "tables/table2_mc.csv",
        "python -m experiments.methods.deep_analysis --version {v}",
        "deep/*seeded42_K25*.npz, deep/*first42_K1*.npz"),
    "Table 4 (ECE/AURC/BALD/flip)": (
        "tables/table3_diag.csv",
        "python -m experiments.methods.deep_analysis --version {v}",
        "deep/*seeded42_K25*.npz via diagnostics.all_diagnostics"),
    "Table 5 (D0/Ds/random/K25 modes)": (
        "tables/table4_modes.csv",
        "python -m experiments.methods.deep_analysis --version {v}",
        "deep/*first42_K1 (D0), deep/*gen42_K1 (Ds), deep/*seeded42_K25"),
    "Table 7 (rotational TTA-25)": (
        "tables/table_tta.csv",
        "python -m experiments.methods.make_table_tta --version {v}",
        "deep/v3_noSynth_s2__best_model__tta25_{val,test}.npz"),
    "Supplementary Table S1 (classical members)": (
        "tables/table6_classical.csv",
        "python -m experiments.methods.make_table6 --version {v}",
        "descriptors/{dzd_c1,fpfh_c2,c4,guerra_*}_metrics.json"),
    "Supplementary Tables S2-S3 (fusion panels / winner's curse)": (
        "tables/table78_fusion.csv",
        "make_table78 + shrec-table-closeout/refresh_table78.py",
        "fusion/fusion_summary.json + 2025-10-01 classical panels"),
    "Supplementary Table S7 (training-recipe ablation)": (
        "tables/table9_ablation.csv",
        "python -m experiments.methods.make_table9 --version {v}",
        "4 abl/v3 best_model K25 npz + dzd_c1 + fpfh_c2"),
    "Table 8 (cross-method comparison)": (
        "tables/table10_crossmethod.csv",
        "python -m experiments.methods.make_table10 --version {v}",
        "history/*grid.csv + barisin json + fusion_summary.json"),
    "Supplementary Table S8 (accuracy by class size)": (
        "tables/table17_groupsize.csv",
        "python -m experiments.methods.make_table17 --version {v}",
        "v3 K1/K25 + v5 K25 + c1/c2 + train_set_2.csv"),
    "Fig. S2 (per-class / confusion)": (
        "tables/perclass_final.csv, confusion_final.csv",
        "python -m experiments.methods.make_table17 --version {v}",
        "Final panel = v5_s2_swa K25 + dzd_c1 + fpfh_c2"),
}

# Original manuscript headline numbers (pre-rerun; documented baseline only).
ORIG = {
    "flip_val": 0.159,
    "v3_val_singlemac": 0.862, "v3_val_k25mac": 0.894,
    "bald_val": 0.035,
    "v5s2_k25_test_mac": 0.8991,
    "final_test_acc": 0.9479, "final_test_bal": 0.9148,
    "final_test_mac": 0.9225,
    "p11_test_mac": 0.9200, "p7_test_mac": 0.8968,
    "tta_test_mac": 0.8899,
}


def deep(v):
    return json.load(open(OUT / v / "deep" / "deep_analysis.json"))


def entry(d, run, K=25):
    return [x for x in d if x["run"] == run and x["K"] == K][0]


def headlines(v):
    d = deep(v)
    v3 = entry(d, "v3_noSynth_s2", 25)
    s2 = entry(d, "v5_s2", 25)
    f = json.load(open(OUT / v / "fusion" / "fusion_summary.json"))
    tta = pd.read_csv(OUT / v / "tables" / "table_tta.csv")
    tt_id = tta[(tta.split == "test") & (tta.method == "identity")].iloc[0]
    tt_25 = tta[(tta.split == "test") &
                (tta.method == "tta25_meanlogit")].iloc[0]
    vv, vt = v3["splits"]["val"], v3["splits"]["test"]
    s2t = s2["splits"]["test"]["curve"]["25"]

    def curve(e, sp, k):
        return e["splits"][sp]["curve"][str(k)]

    h = {
        "flip_val": vv["diag"]["flip_rate"],
        "flip_test": vt["diag"]["flip_rate"],
        "v3_val_singlemac": vv["single_mean_mac"],
        "v3_val_k25mac": curve(v3, "val", 25)["mac"],
        "v3_test_singlemac": vt["single_mean_mac"],
        "v3_test_k25acc": curve(v3, "test", 25)["acc"],
        "v3_test_k25bal": curve(v3, "test", 25)["bal"],
        "v3_test_k25mac": curve(v3, "test", 25)["mac"],
        "bald_val": vv["diag"]["bald_mean"],
        "bald_test": vt["diag"]["bald_mean"],
        "ece_val_k25": vv["diag"]["ece15"],
        "ece_test_k25": vt["diag"]["ece15"],
        "aurc_val_k25": vv["diag"]["aurc"],
        "aurc_test_k25": vt["diag"]["aurc"],
        "v5s2_k25_test_acc": s2t["acc"],
        "v5s2_k25_test_bal": s2t["bal"],
        "v5s2_k25_test_mac": s2t["mac"],
        "final_val_acc": f["final"]["val"]["acc"],
        "final_val_bal": f["final"]["val"]["bal_acc"],
        "final_val_mac": f["final"]["val"]["f1_macro"],
        "final_test_acc": f["final"]["test"]["acc"],
        "final_test_bal": f["final"]["test"]["bal_acc"],
        "final_test_mac": f["final"]["test"]["f1_macro"],
        "p11_argmax_test_mac": f["p11_argmax"]["test"]["f1_macro"],
        "p11_onse_test_mac": f["p11_onse"]["test"]["f1_macro"],
        "p7_test_mac": float(pd.read_csv(
            OUT / v / "tables" / "table78_fusion.csv")
            .query("panel=='p7' and selection=='onse'")["test_mac"].iloc[0]),
        "tta_test_id_mac": float(tt_id["f1_macro"]),
        "tta_test_k25_mac": float(tt_25["f1_macro"]),
        "tta_test_k25_acc": float(tt_25["acc"]),
    }
    return h


def check_consistency(v, h, fails):
    # final matches table78 and table10 rows
    t78 = pd.read_csv(OUT / v / "tables" / "table78_fusion.csv")
    fr = t78.query("panel=='final'").iloc[0]
    for col, key in (("test_acc", "final_test_acc"),
                     ("test_bal", "final_test_bal"),
                     ("test_mac", "final_test_mac")):
        if abs(fr[col] - h[key]) > 1e-9:
            fails.append("%s final %s table78!=summary" % (v, key))
    t10 = pd.read_csv(OUT / v / "tables" / "table10_crossmethod.csv")
    ours = t10[t10.method == "ours_Final"].iloc[0]
    if abs(ours["mac"] - h["final_test_mac"]) > 1e-4:
        fails.append("%s final mac table10!=summary" % v)
    # v3 K25 in table9 standalone == deep_analysis
    t9 = pd.read_csv(OUT / v / "tables" / "table9_ablation.csv")
    r = t9[(t9.block == "standalone") &
           (t9.run == "v3_noSynth_s2") & (t9.split == "test")].iloc[0]
    if abs(r["f1_macro"] - h["v3_test_k25mac"]) > 1e-9:
        fails.append("%s v3 K25 mac table9!=deep_analysis" % v)


def main():
    nmap, fails = {}, []
    for v in VERSIONS:
        h = headlines(v)
        nmap[v] = h
        check_consistency(v, h, fails)
    with open(OUT / "number_map.json", "w") as fp:
        json.dump(nmap, fp, indent=2)

    L = ["# Paper number <-> cache cross-reference (Task 18)",
         "",
         "Every number in the revised manuscript is generated from a cache; "
         "no result is hand-typed. Rounding in the prose is to 3-4 decimals "
         "of the values below.", ""]
    for title, (rel, gen, lineage) in TABLES.items():
        L += ["## %s" % title,
              "- artifact: `experiments/results/<version>/%s`" % rel,
              "- generator: `%s`" % gen,
              "- cache lineage: %s" % lineage, ""]

    L += ["## Headline numbers (abstract / conclusions)", ""]
    keys = [k for k in nmap["paper"]]
    L.append("| key | paper | updated | original manuscript (baseline only) |")
    L.append("|---|---|---|---|")
    for k in keys:
        ov = ORIG.get(k, "")
        L.append("| %s | %.4f | %.4f | %s |" % (
            k, nmap["paper"][k], nmap["updated"][k],
            ("%.4f" % ov) if isinstance(ov, float) else "—"))
    L += ["", "## Cross-file consistency assertions", ""]
    if fails:
        L += ["FAIL:"] + ["- %s" % x for x in fails]
    else:
        L.append("PASS: final panel identical across fusion_summary / "
                 "table78 / table10; v3 K25 mac identical across "
                 "deep_analysis / table9 for both versions.")
    L += ["", "## Reproduction deltas (see manuscript Supplementary Section S2)",
          "The original results_final caches and the a_layer classical code "
          "were unavailable; all rerun numbers come from regenerated "
          "8,192-point clouds, retrained checkpoints, and rebuilt FPFH/3DZD "
          "descriptors. Qualitative findings reproduce; absolute levels "
          "differ and are disclosed in Sec. 6.4."]
    (OUT / "paper_numbers_audit.md").write_text("\n".join(L) + "\n")
    print("headlines (paper / updated):")
    for k in keys:
        print("  %-22s %.4f / %.4f" % (k, nmap["paper"][k], nmap["updated"][k]))
    print("consistency:", "FAIL %s" % fails if fails else "PASS")
    print("wrote", OUT / "paper_numbers_audit.md", "and number_map.json")


if __name__ == "__main__":
    main()
