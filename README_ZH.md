# SHREC 2025 蛋白质形状分类 —— 复现指南

本目录是一个**自包含**的复现包，用于复现 SHREC 2025 蛋白质形状分类赛道（97 类）论文的全部实验数据。
最终融合系统在 **测试集** 上的指标为：

| 指标 | 数值 |
| --- | --- |
| Accuracy（准确率） | **94.79 %** |
| Balanced accuracy（平衡准确率） | **91.48 %** |
| Macro-F1 | **92.25 %** |

系统为三路概率空间后融合：一路深度信号（RIConv++ v5，SWA + Monte-Carlo FPS 25 次平均）
与两路经典信号（3DZD 1-NN + 体积门限，FPFH + RBF-SVM margin）。所有冻结超参见 `configs/`。

> **逐步操作手册**见 [REPRODUCE.md](REPRODUCE.md)（前置清单 / 校验点 / 故障排查矩阵 / 脚本速查表）。本文档为包总览。

> **自包含设计说明**：本包所有脚本用 `BASE = 脚本所在目录的上两级 = 本包根目录`。
> 因此**所有命令都请在本包根目录 `replication/` 下执行**（`cd replication/`）。
> 数据目录 `new_data/`、`log/`、`results_final/` 已在本包内建好占位，生成/下载的数据直接放进即可，
> 无需改动任何脚本路径。

---

## 1. 目录结构

```
replication/
├── README.md                    # 本文件
├── run_all.sh                   # 5 阶段端到端流水线（带前置守卫，可按阶段跳过）
├── requirements.txt             # pin 依赖
├── configs/                     # 冻结超参
│   ├── v5.yaml                  # 深层训练（lr1e-3, T_max245, focalγ1.0, cutmix0.2, swa200, seed2）
│   ├── c_members.yaml           # c1–c4 经典成员配置 + 各自 test 指标
│   └── fusion_final.yaml        # K=25, 权重 2:2:2, c1 τ=.5, c2 τ=1.0
├── 00_data/                     # 数据准备
│   ├── DOWNLOAD.md              # VTK / 点云 / 划分下载说明
│   ├── extract_pc_fast.py       # 从 VTK 提取 8192×8 点云（scipy cKDTree 加速）
│   └── splits/                  # 划分 CSV（已内置，勿移动）
│       ├── train_set_2.csv      # 9244 行：前 7428 训练 / 后 1816 验证
│       ├── test_set_2.csv       # 2321 行：anonymised_protein_id
│       └── test_set_ground_truth.csv  # 2321 行：anonymised_protein_id, class_id
├── csv -> 00_data/splits        # 软链：脚本按 BASE/csv/ 读划分，此处统一指向 splits
├── 01_a_layer/                  # A 层（深度学习）
│   ├── train_v5.py              # v5 训练（250 epoch，单 GPU ~24h）
│   ├── train_v3.py              # v3 训练（基线 / 消融参照）
│   ├── dump_v5.py               # 单次前向 log-prob 缓存
│   ├── dump_mcfps.py            # MC-FPS K 次随机前向，mean-logit 平均
│   ├── dump_clean.py            # v3 干净单次 dump（供 split 分析）
│   ├── split_sensitivity.py     # §III-D：分层 vs 行序划分敏感性（不重训）
│   ├── calibration_analysis.py  # §V-C：ECE / AURC / BALD MI（读 25-draw cache，需 pandas/sklearn）
│   ├── deterministic_init_eval.py # §V-D：D0/Ds/随机/MC-FPS 对照（需 numpy>=2）
│   ├── member_overlap_rho.py    # §V-B/E：成员错误集 Jaccard ρ（缺缓存自动跳过）
│   ├── models/                  # riconv2_cls_v2_features2_largest.py + riconv2_utils_features.py
│   ├── data_utils/              # ProteinToPointCloudProcessor(.py / Test.py)
│   └── checkpoints/DOWNLOAD.md # v3 best_model.pth 下载链接；v5 swa 由训练产出
├── 02_c_layer/                  # C 层（经典描述子 + 融合逻辑）
│   ├── run_3dzd.py / eval_3dzd.py        # c1: 3DZD 1-NN + 体积门限
│   ├── run_fpfh_o3d.py / eval_fpfh.py   # c2: FPFH + RBF-SVM(decision_function)
│   ├── run_guerra.py / eval_guerra.py   # c3: Guerra 拓扑 + SimpleNN 8 视角硬投票
│   ├── tune_svm_members.py              # c1d / c4 SVM 成员
│   ├── tune_3dzd*.py / tune_fpfh.py / eval_*.py  # 其余调参 / 评估
│   └── external/                        # 第三方 baseline 源码（tehrani / guerra / shrec2025_protein / SHREC2025）
├── 03_fusion/                   # 概率空间后融合
│   ├── fuse_c.py                # 经典成员融合 + 网格搜索
│   ├── fuse_v5.py               # v5 单次信号融合
│   ├── fuse_mcfps.py            # MC-FPS 深度信号融合
│   └── fuse_mcfps2.py           # one-SE nested-K 选 K=25 + 配对 bootstrap（生产入口）
├── 04_analysis/                 # 分析与出图
│   ├── compute_final_metrics.py # 验证冻结预测 → 应得 94.79/91.48/92.25
│   ├── make_framework_fig.py    # Fig.1 系统框架图（纯绘制，无需数据/GPU）
│   ├── make_variance_fig.py     # MC-FPS 各 pass 方差图
│   ├── make_k_curve_fig.py      # nested-K val 曲线
│   ├── make_winners_curse_fig.py# 赢者诅咒示意
│   ├── make_calibration_fig.py  # §V-C：可靠性图 + risk-coverage 单栏图
│   ├── make_confusion_matrix.py # 97×97 混淆矩阵
│   └── make_per_class_table.py  # LaTeX 每类表
├── 05_cross_arch/               # 可选：跨架构复现（ModelNet40 / PointNet++ SSG+MSG）
│   ├── route_b_eval.py          # 25-draw 单模型 vs MC-FPS：acc/F1/flip/ECE/AURC/MI（numpy>=2）
│   └── routeb_summary.json      # 冻结结果（SSG/MSG）
├── expected_outputs/
│   └── test_full_fusion.csv     # 冻结生产预测（2321 行，已验证）
├── new_data/                    # 【已内置】2321 个测试集点云（~1.2 GB）；训练集点云可从 DOWNLOAD.md 链接补充
├── log/                         # 【已内置】v3 checkpoint + 训练日志（run1/run2，共 ~138 MB）
│   └── classification_shrec2025/
│       ├── riconv_large_shrec_run1_8k_largest/  # v2 baseline（best_model.pth + log + src）
│       └── riconv_large_shrec_run2_8k_largest/  # v3 best_model.pth + log + src（MC-FPS 参照）
├── results_final/               # 【占位】dump/融合产出的 cache 与 CSV
└── figures/                     # 【占位】出图脚本写图于此
```

---

## 2. 环境

复现依赖按**本机真实训练环境** pin（见 `requirements.txt`）：

- **Python** 3.8.10（系统 `python`；亦可用 `python3.11` 但需另配同版本依赖）
- **PyTorch** 2.3.0 + **cu121**（与本机 NVIDIA 驱动 535.161.07 / 最高 CUDA 12.2 匹配）
- **open3d** 0.13.0（`run_fpfh_o3d.py` 提取 FPFH 用；py3.8 能装到的最新版）
- **scikit-learn** 1.3.2、**matplotlib** 3.7.5、**pandas** 2.0.3、**numpy** 1.24.4、**scipy** 1.10.1
- **trimesh** 4.12.2、**point-cloud-utils** 0.34.0、tqdm 4.66.4

> 原方案曾 pin torch 2.8.0+cu126，但需驱动 ≥545（CUDA 12.6），本机驱动不支持；
> 论文实验实际在 torch 2.3.0+cu121 下完成，故以此为准。
> **GPU**：2× NVIDIA A40（46 GB）；训练用 `--gpu 0`（GPU1 常被占用）。
>
> **可选的第二环境（Route B/C）**：确定性初始点对照（`deterministic_init_eval.py`）
> 与跨架构复现（`05_cross_arch/route_b_eval.py`）要求 **numpy>=2**（checkpoint 由
> numpy 2.x pickle，numpy 1.x 反序列化会静默损坏前向），脚本本身只依赖 numpy+torch。
> 可用独立环境（如 Python 3.13 + torch 2.8）并通过 `SHARP_PYTHON=<解释器>` 传给流水线。

---

## 3. 安装

直接用系统 Python 3.8 装依赖（已含 open3d，无需单独装）：

```bash
cd replication
python -m pip install -r requirements.txt        # 若权限不足加 --user
# 校验
python -c "import torch,sklearn,open3d,trimesh; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
```

> 也可建独立环境：`python3.8 -m venv .venv && source .venv/bin/activate` 后再 `pip install -r requirements.txt`。

---

## 4. 数据准备

详见 `00_data/DOWNLOAD.md`，要点：

1. **VTK 原始网格**：从 SHREC 官网下载 `train_set.tar.xz`、`test_set.tar.xz`。
2. **点云**（`new_data/`，每个文件 `{basename}_pc.npy` 形状 `(8192, 8)`，
   通道 `[x, y, z, nx, ny, nz, potential, normal_potential]`）：
   **已内置** 2321 个测试集点云（~1.2 GB），开箱即用；
   训练集点云（9244 个）可从 `00_data/DOWNLOAD.md` 链接补充。
3. **划分 CSV** 已内置在 `00_data/splits/`，无需再下；`csv/` 软链已指向它。
4. **预训练 checkpoint**：v3 `best_model.pth` **已内置**于
   `log/classification_shrec2025/riconv_large_shrec_run2_8k_largest/checkpoints/`；
   v5 `swa_model.pth` 由阶段 1a 训练产出（见 `01_a_layer/checkpoints/DOWNLOAD.md`）。

---

## 5. 复现操作过程（5 阶段流水线）

流水线方向：`00_data → 01_a_layer → 02_c_layer → 03_fusion → 04_analysis`。
**所有命令在 `replication/` 根目录执行**。一键运行见 `run_all.sh`（可按阶段跳过，见 §6）。

### 阶段 0 —— 数据就位
`new_data/` 有点云、`00_data/splits/` 划分就位即可，无需运行脚本。

### 阶段 1 —— A 层（深度学习）
```bash
# 1a. 训练 v5（seed 2，250 epoch，单 GPU ~24h）
python 01_a_layer/train_v5.py --seed 2 --log_dir riconv_large_v5_s2
# 训练产出 log/classification_shrec2025/riconv_large_v5_s2/checkpoints/swa_model.pth

# 1b. dump MC-FPS（25 次干净前向，mean-logit 平均后 softmax）
python 01_a_layer/dump_mcfps.py --log_dir riconv_large_v5_s2 \
       --ckpt swa_model.pth --K 25 --prefix mcfps_v5s2

# 1c. dump v5 单次前向（对比用）
python 01_a_layer/dump_v5.py --log_dir riconv_large_v5_s2 \
       --ckpt swa_model.pth --tag s2_swa
```
> 若不想重训 v5，可从已训练好的 `log/.../swa_model.pth` 直接拷入，或下载 v3 `best_model.pth` 走 v3 MC-FPS 路径。

### 阶段 2 —— C 层（经典描述子）
```bash
# 2a. 3DZD 提取 + 1-NN（c1）
python 02_c_layer/run_3dzd.py
python 02_c_layer/eval_3dzd.py
# 2b. FPFH 提取 + RBF-SVM（c2）
python 02_c_layer/run_fpfh_o3d.py
python 02_c_layer/eval_fpfh.py
# 2c. Guerra 拓扑 + SimpleNN 8 视角硬投票（c3）
python 02_c_layer/run_guerra.py
python 02_c_layer/eval_guerra.py
# 2d. SVM 成员（c1d, c4 = [363 ZDZ | 612 FPFH] 块标准化拼接 + RBF-SVM C16/γ.003）
python 02_c_layer/tune_svm_members.py
```

### 阶段 3 —— 融合
```bash
python 03_fusion/fuse_mcfps2.py
```
`fuse_mcfps2.py` 在 val 上做 one-SE nested-K 选 K=25、配对 bootstrap 对照冻结生产 val 预测，
并对赢家出一次 test 报告。**最终生产预测已冻结在 `expected_outputs/test_full_fusion.csv`**，
脚本不会自动覆盖该文件。

### 阶段 4 —— 分析 / 验证
```bash
python 04_analysis/compute_final_metrics.py      # 验证最终 test 数字
python 01_a_layer/calibration_analysis.py        # §V-C：ECE/AURC/BALD（需 25-draw cache）
python 04_analysis/make_calibration_fig.py       # 可靠性图 + risk-coverage 图
<py313> 01_a_layer/deterministic_init_eval.py    # §V-D：D0/Ds/随机/K25（需 numpy>=2）
python 01_a_layer/split_sensitivity.py           # §III-D 划分敏感性（需 v3 clean cache）
python 01_a_layer/member_overlap_rho.py          # §V-B/E 成员错误重叠 ρ
python 04_analysis/make_framework_fig.py         # Fig.1 框架图（无依赖）
python 04_analysis/make_variance_fig.py          # MC-FPS pass 方差图
python 04_analysis/make_k_curve_fig.py           # nested-K val 曲线
python 04_analysis/make_winners_curse_fig.py      # 赢者诅咒示意
python 04_analysis/make_confusion_matrix.py      # 97×97 混淆矩阵
python 04_analysis/make_per_class_table.py       # LaTeX 每类表
```
图写入 `figures/`；每类表打印到 stdout。标定/确定性结果写入
`results_final/calibration_summary.json`、`deterministic_init_summary.json`。

### 阶段 5 —— 跨架构复现（可选，ModelNet40）
```bash
# 外部资源（下载方式见 REPRODUCE.md §7）：
#   MODELNET40_ROOT = modelnet40_normal_resampled 目录
#   POINTNET2_REPO = yanx27/Pointnet_Pointnet2_pytorch 克隆（含发布的 checkpoints）
MODELNET40_ROOT=<dir> POINTNET2_REPO=<dir> \
    SHARP_PYTHON=<py313> STAGE=5 ./run_all.sh
```
结果写入 `05_cross_arch/routeb_summary.json`（SSG/MSG：label-flip≈5.3 %，
MC-FPS 在准确率/ECE/AURC 上一致改善）。

---

## 6. 一键运行（run_all.sh）

```bash
cd replication
./run_all.sh                  # 默认跑全部 5 阶段（含训练，~24h+）
SKIP_TRAIN=1 ./run_all.sh    # 跳过训练（已有 checkpoint 时）
STAGE=4 ./run_all.sh          # 只跑阶段 4 验证冻结预测
SHARP_PYTHON=<py313> STAGE=4 ./run_all.sh   # 含确定性初始点对照
MODELNET40_ROOT=<dir> POINTNET2_REPO=<dir> STAGE=5 ./run_all.sh  # 跨架构
```
`run_all.sh` 带前置守卫：会检查 `new_data/` 是否有点云、checkpoint 是否存在，缺则提示而非盲跑。

---

## 7. 期望输出验证

`expected_outputs/test_full_fusion.csv` 为冻结生产预测（2321 行，列 `anonymised_protein_id, predicted_label`）。
运行 `04_analysis/compute_final_metrics.py` 应得到：

```
accuracy=94.79  balanced=91.48  macro-F1=92.25   PASS
```

---

## 8. 关键避坑（务必遵守）

- **FPS 随机性**：最远点采样在每一层 set-abstraction 都用随机起点（`riconv2_utils_features.py` 第 88 行 `torch.randint`），
  单次干净前向是随机的——**这是预期行为**。生产信号用 25 次 MC-FPS 平均消除方差，
  **绝不可把单次前向当最终结果上报**。

- **Guerra 干净 refit**：`c3`/`c3b` 必须**在全 9244 行训练集上 refit**，不能只在 7428 行内部训练划分上训练。
  漏 refit 会静默掉约 1 个点 test 精度。`configs/c_members.yaml` 中 `clean_refit_on_full_9244: true` 已记录。

- **c1 距离 inf 填充**：3DZD 1-NN 距离数组在体积门限拒绝所有 gallery 项处会出现 `inf`。
  送入 `softmax(c1_dist / tau)` 前须把 `inf` 换成大有限值（如最大有限距离或 `1e6`），
  否则 softmax 出 NaN 致整个融合崩塌。

- **SVM 禁用 Platt**：**不要**用 `SVC(probability=True)`（Platt scaling）。在 97 类不平衡任务上 Platt 概率校准很差，
  会拖垮融合。所有 SVM 成员一律用 `decision_function` margin（见 `configs/c_members.yaml` 的 `decision_function: true`）。

- **类别权重温和策略**：必须用 sqrt-inverse-frequency 权重，clip 到最大/最小比 10:1。
  激进权重（如 Class-Balanced β=0.999，权重比 800+:1）会欠拟合，test 准确率 < 87.5%。

- **数据划分按行序**：`train_set_2.csv` 前 7428 行训练、后 1816 行验证，**禁止分层划分**
  （分层会抬高验证集 class acc 造成乐观假象，被 test 否定）。

- **候选空间防膨胀**：融合候选空间随成员指数膨胀时易现"赢者诅咒"——val 上小幅优势在 test 可能反转。
  因此最终采用 one-SE nested-K + 配对 bootstrap 的稳健口径，而非单次幸运 draw。

---

## 9. 与论文表的对应

- 最终融合指标（§7）→ `expected_outputs/test_full_fusion.csv` + `compute_final_metrics.py`
- 单模型 / 融合级消融表（论文 tab:ablation）→ `01_a_layer` 的 dump 结果 + `03_fusion/fuse_mcfps2.py` 同口径重算
- 赢者诅咒示意 → `04_analysis/make_winners_curse_fig.py`
- MC-FPS 方差 → `04_analysis/make_variance_fig.py`
- 每类精度 → `04_analysis/make_per_class_table.py` / `make_confusion_matrix.py`
- ECE / AURC / BALD（§V-C）→ `01_a_layer/calibration_analysis.py` + `04_analysis/make_calibration_fig.py`
- 确定性初始点对照（§V-D）→ `01_a_layer/deterministic_init_eval.py`
- 跨架构 ModelNet40（§V-E）→ `05_cross_arch/route_b_eval.py` + `05_cross_arch/routeb_summary.json`
