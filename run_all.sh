#!/bin/bash
# SHREC 2025 蛋白质形状分类 —— 端到端复现流水线
# 用法：
#   ./run_all.sh                 # 跑全部 5 阶段（含训练，单 GPU ~24h+）
#   SKIP_TRAIN=1 ./run_all.sh    # 已有 checkpoint，跳过阶段 1a 训练
#   STAGE=4 ./run_all.sh         # 只跑某一阶段（1/2/3/4/5）
#   PYTHON=python3.11 ./run_all.sh
#   SHARP_PYTHON=/path/to/numpy2/python  # Route C 确定性对照需 numpy>=2 的环境
#   MODELNET40_ROOT=<dir> POINTNET2_REPO=<dir> STAGE=5 ./run_all.sh   # 跨架构复现
# 所有命令在 replication/ 根目录执行（脚本会自动 cd）。
set -euo pipefail

PROJ=$(cd "$(dirname "$0")" && pwd)
cd "$PROJ"
PYTHON=${PYTHON:-python}
STAGE=${STAGE:-0}          # 0 = 全部
SKIP_TRAIN=${SKIP_TRAIN:-0}
LOG_DIR=riconv_large_v5_s2
CKPT=swa_model.pth

c_red()    { printf '\033[31m%s\033[0m\n' "$*"; }
c_green()  { printf '\033[32m%s\033[0m\n' "$*"; }
c_yellow() { printf '\033[33m%s\033[0m\n' "$*"; }
banner()   { printf '\n\033[1;36m======== %s ========\033[0m\n' "$*"; }

# ---- 前置守卫 ---------------------------------------------------------------
guard_data() {
    local n
    n=$(find new_data -name '*_pc.npy' 2>/dev/null | head -1)
    if [[ -z "$n" ]]; then
        c_red "[守卫] new_data/ 下未找到 *_pc.npy 点云。请先按 README §4 准备数据。"
        c_red "       下载预计算点云或运行: python 00_data/extract_pc_fast.py --vtk_dir <dir> --out_dir new_data"
        exit 1
    fi
    c_green "[守卫] new_data/ 点云就位: $n ..."
}
guard_ckpt() {
    local p="log/classification_shrec2025/$LOG_DIR/checkpoints/$CKPT"
    if [[ ! -f "$p" ]]; then
        c_red "[守卫] 缺少 $p"
        c_red "       请先训练 v5（阶段 1a），或从 01_a_layer/checkpoints/DOWNLOAD.md 获取。"
        exit 1
    fi
    c_green "[守卫] checkpoint 就位: $p"
}
run() { c_yellow "$ $*"; "$@"; }

should_run() { [[ $STAGE -eq 0 || $STAGE -eq $1 ]]; }

# ---- 阶段 0：数据 -----------------------------------------------------------
banner "阶段 0: 数据准备"
# 仅阶段 1/2/3 需要点云；阶段 4（分析）只读 expected_outputs + splits，无需 new_data
if [[ $STAGE =~ ^(4|5)$ ]]; then :; else guard_data; fi
if [[ $STAGE =~ ^(4|5)$ ]]; then c_yellow "[阶段 $STAGE] 跳过数据守卫（不需点云）"; fi

# ---- 阶段 1：A 层（深度学习）-----------------------------------------------
if should_run 1; then
    banner "阶段 1: A 层（深度学习）"
    if [[ $SKIP_TRAIN -eq 1 ]]; then
        c_yellow "[跳过训练] SKIP_TRAIN=1，直接用已有 checkpoint"
        guard_ckpt
    else
        run $PYTHON 01_a_layer/train_v5.py --seed 2 --log_dir "$LOG_DIR"
    fi
    guard_ckpt
    run $PYTHON 01_a_layer/dump_mcfps.py --log_dir "$LOG_DIR" --ckpt "$CKPT" --K 25 --prefix mcfps_v5s2
    run $PYTHON 01_a_layer/dump_v5.py    --log_dir "$LOG_DIR" --ckpt "$CKPT" --tag s2_swa

    # 1d. 可选：v3 基线缓存（供 split_sensitivity 与 rho(*,v3) 复现）
    V3_CKPT="log/classification_shrec2025/riconv_large_shrec_improved_v3/checkpoints/best_model.pth"
    if [[ -f "$V3_CKPT" ]]; then
        c_yellow "[v3] 检测到 best_model.pth，dump v3 单次 / MC-FPS 缓存"
        run $PYTHON 01_a_layer/dump_clean.py --splits train,val
        run $PYTHON 01_a_layer/dump_mcfps.py --K 25
    else
        c_yellow "[v3] 未提供 best_model.pth（见 01_a_layer/checkpoints/DOWNLOAD.md），跳过 v3 缓存"
    fi
fi

# ---- 阶段 2：C 层（经典描述子）---------------------------------------------
if should_run 2; then
    banner "阶段 2: C 层（经典描述子）"
    run $PYTHON 02_c_layer/run_3dzd.py
    run $PYTHON 02_c_layer/eval_3dzd.py
    run $PYTHON 02_c_layer/run_fpfh_o3d.py
    run $PYTHON 02_c_layer/eval_fpfh.py
    run $PYTHON 02_c_layer/run_guerra.py
    run $PYTHON 02_c_layer/eval_guerra.py
    run $PYTHON 02_c_layer/tune_svm_members.py
fi

# ---- 阶段 3：融合 -----------------------------------------------------------
if should_run 3; then
    banner "阶段 3: 概率空间后融合"
    run $PYTHON 03_fusion/fuse_mcfps2.py
fi

# ---- 阶段 4：分析 / 验证 ----------------------------------------------------
if should_run 4; then
    banner "阶段 4: 分析与验证"
    run $PYTHON 04_analysis/compute_final_metrics.py
    run $PYTHON 01_a_layer/calibration_analysis.py  || c_yellow "[warn] calibration_analysis 失败（缺 25-draw cache，跳过）"
    run $PYTHON 04_analysis/make_calibration_fig.py || c_yellow "[warn] make_calibration_fig 失败（跳过）"
    if [[ -n "${SHARP_PYTHON:-}" ]]; then
        run "$SHARP_PYTHON" 01_a_layer/deterministic_init_eval.py || c_yellow "[warn] deterministic_init_eval 失败（跳过）"
    else
        c_yellow "[skip] deterministic_init_eval 需 numpy>=2 环境；设 SHARP_PYTHON=<python> 后重跑 STAGE=4"
    fi
    run $PYTHON 01_a_layer/split_sensitivity.py    || c_yellow "[warn] split_sensitivity 失败（缺 v3 clean cache，跳过）"
    run $PYTHON 01_a_layer/member_overlap_rho.py   || c_yellow "[warn] member_overlap_rho 失败（缺 cache，跳过）"
    run $PYTHON 04_analysis/make_framework_fig.py
    run $PYTHON 04_analysis/make_variance_fig.py       || c_yellow "[warn] make_variance_fig 失败（可能缺 cache，跳过）"
    run $PYTHON 04_analysis/make_k_curve_fig.py        || c_yellow "[warn] make_k_curve_fig 失败（跳过）"
    run $PYTHON 04_analysis/make_winners_curse_fig.py  || c_yellow "[warn] make_winners_curse_fig 失败（跳过）"
    run $PYTHON 04_analysis/make_confusion_matrix.py  || c_yellow "[warn] make_confusion_matrix 失败（跳过）"
    run $PYTHON 04_analysis/make_per_class_table.py    || c_yellow "[warn] make_per_class_table 失败（跳过）"
fi

# ---- 阶段 5：跨架构复现（ModelNet40，可选）----------------------------------
if should_run 5; then
    banner "阶段 5: 跨架构种子敏感性（ModelNet40 / PointNet++ SSG+MSG）"
    if [[ -z "${MODELNET40_ROOT:-}" || -z "${POINTNET2_REPO:-}" ]]; then
        c_yellow "[skip] 需要外部资源：MODELNET40_ROOT=modelnet40_normal_resampled 目录，"
        c_yellow "       POINTNET2_REPO=Pointnet_Pointnet2_pytorch 克隆（见 REPRODUCE.md）"
        c_yellow "       该阶段还需要 numpy>=2 + torch 的环境（可用 SHARP_PYTHON）。"
    else
        RP="${SHARP_PYTHON:-$PYTHON}"
        run "$RP" 05_cross_arch/route_b_eval.py --data_root "$MODELNET40_ROOT" \
            --repo_root "$POINTNET2_REPO" --out_dir 05_cross_arch
        c_green "期望输出见 05_cross_arch/routeb_summary.json（SSG/MSG flip≈5.3%）"
    fi
fi

banner "流水线完成"
c_green "期望最终：TEST acc=94.79, bal=91.48, macF1=92.25"
