# 复现操作文档（Runbook）

> 本文档是 SHREC 2025 蛋白质形状分类论文实验数据的**逐步复现操作手册**。
> 包结构与速查见 [README.md](README.md)；本文聚焦「**怎么一步步跑、每步看到什么、出问题怎么修**」。
> **所有命令均在 `replication/` 根目录执行**（`cd replication/`）。

**复现目标（测试集）**

| 指标 | 目标值 | 来源 |
| --- | --- | --- |
| Accuracy | 94.79 % | `expected_outputs/test_full_fusion.csv` |
| Balanced accuracy | 91.48 % | 同上 |
| Macro-F1 | 92.25 % | 同上 |

---

## 0. 前置清单（Preflight Checklist）

执行前逐项确认：

- [ ] **操作系统**：Linux（已验证）。Windows 下 `csv -> 00_data/splits` 软链需改为真实目录或 junction。
- [ ] **GPU**：NVIDIA + 驱动支持 CUDA 12.1（本机 2× A40，驱动 535.161.07 / 最高 CUDA 12.2）。训练用 `--gpu 0`（GPU1 常被占用）；纯阶段 4 验证可纯 CPU。
- [ ] **Python 3.8.10**：`python --version`（系统 `python`；本机论文实验实际环境）。`python3.11` 亦可用但需另配同版本依赖。
- [ ] **磁盘**：训练产出 + 点云约 ~30 GB 可用空间。
- [ ] **包内目录**：`new_data/`、`log/`、`results_final/`、`figures/` 已存在（占位）。
- [ ] **`csv` 软链**：`ls csv/` 应列出 3 个 split CSV（指向 `00_data/splits/`）。

校验命令：
```bash
cd replication
ls csv/                       # 应见 train_set_2.csv / test_set_2.csv / test_set_ground_truth.csv
test -d new_data && echo OK   # 占位目录在即可
python --version              # 3.8.x
```

---

## 1. 环境搭建

依赖按本机真实训练环境 pin（torch 2.3.0+cu121，见 `requirements.txt`），已含 open3d：

```bash
cd replication
python -m pip install -r requirements.txt          # 权限不足加 --user
```

**校验点 1**：环境就绪
```bash
python -c "import torch,sklearn,open3d,trimesh; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
# 期望：torch 2.3.0+cu121 cuda True（训练机）；cuda False 时仅阶段 4 可跑
```

---

## 2. 数据获取与校验

详见 `00_data/DOWNLOAD.md`。

### 2.1 划分 CSV（已内置，无需下载）
`00_data/splits/` 已含：
- `train_set_2.csv` — 9244 行（前 7428 训练 / 后 1816 验证），列 `protein_id, class_id`
- `test_set_2.csv` — 2321 行，列 `anonymised_protein_id`
- `test_set_ground_truth.csv` — 2321 行，列 `anonymised_protein_id, class_id`

```bash
wc -l 00_data/splits/*.csv
# 期望：9245 / 2322 / 2322（含表头）
```

### 2.2 点云 `new_data/`
每个文件 `{basename}_pc.npy`，形状 `(8192, 8)`，通道 `[x,y,z, nx,ny,nz, potential, normal_potential]`。

**已内置**：`new_data/` 已含 2321 个测试集点云（~1.2 GB），开箱即用。
如需训练集点云（9244 个）：
- **方式 A（推荐，快）**：从 `00_data/DOWNLOAD.md` 链接下载预计算压缩包，解压到 `new_data/`。
- **方式 B（自行提取）**：
  ```bash
  # 先从 SHREC 官网下载并解压 train_set.tar.xz / test_set.tar.xz
  python 00_data/extract_pc_fast.py --vtk_dir <train_set解压目录> --out_dir new_data
  python 00_data/extract_pc_fast.py --vtk_dir <test_set解压目录>  --out_dir new_data
  ```

**校验点 2**：点云数量
```bash
find new_data -name '*_pc.npy' | wc -l
# 内置后：2321（测试集）；下载训练集后：11565（9244 + 2321）
ls new_data | head -1 | xargs -I{} python -c "import numpy as np; a=np.load('new_data/{}'); print(a.shape, a.dtype)"
# 期望：(8192, 8) floatXX
```

### 2.3 预训练 checkpoint
- **v3 `best_model.pth`**：**已内置**于
  `log/classification_shrec2025/riconv_large_shrec_run2_8k_largest/checkpoints/best_model.pth`，
  dump 脚本自动定位。如需重新下载见 `01_a_layer/checkpoints/DOWNLOAD.md`。
- **v5 `swa_model.pth`**：由阶段 1a 训练产出在 `log/classification_shrec2025/riconv_large_v5_s2/checkpoints/swa_model.pth`；若已有可拷入该路径。

---

## 3. 复现流程（5 阶段）

流水线：`00_data → 01_a_layer → 02_c_layer → 03_fusion → 04_analysis`。

> 一键运行：`./run_all.sh`（全 5 阶段，含训练 ~24h+）；`SKIP_TRAIN=1 ./run_all.sh`（已有 ckpt 跳训练）；`STAGE=N ./run_all.sh`（只跑某阶段）。下文为分步手动执行。

### 阶段 1 — A 层（深度学习）

```bash
# 1a. 训练 v5（seed 2，250 epoch，单 GPU ~24h）
python 01_a_layer/train_v5.py --seed 2 --log_dir riconv_large_v5_s2
# 产出：log/classification_shrec2025/riconv_large_v5_s2/checkpoints/{best_macrof1.pth,swa_model.pth,...}

# 1b. dump MC-FPS（25 次干净前向，mean-logit 平均后 softmax）
python 01_a_layer/dump_mcfps.py --log_dir riconv_large_v5_s2 \
       --ckpt swa_model.pth --K 25 --prefix mcfps_v5s2
# 产出：results_final/cache/ 下 mcfps_v5s2* 的 val/test log-prob 缓存

# 1c. dump v5 单次前向（对比基准）
python 01_a_layer/dump_v5.py --log_dir riconv_large_v5_s2 \
       --ckpt swa_model.pth --tag s2_swa

# 1d.（可选）v3 基线缓存：下载 best_model.pth 后执行，供阶段 4 的
#     split_sensitivity 与 rho(*,v3) 使用；run_all.sh 检测到 v3 ckpt 会自动执行
python 01_a_layer/dump_clean.py --splits train,val
python 01_a_layer/dump_mcfps.py --K 25
```

**校验点 3**：checkpoint 与缓存
```bash
ls log/classification_shrec2025/riconv_large_v5_s2/checkpoints/swa_model.pth   # 应存在
ls results_final/cache/ | grep mcfps_v5s2                                       # 应有缓存文件
```
> v5 训练默认超参（`configs/v5.yaml`）：Adam lr1e-3/wd1e-4，CosineAnnealing T_max245/eta_min1e-5 + 5 epoch warmup，focal γ1.0，cutmix 0.2，sqrt-inverse 类权重 clip 10:1，SWA 起 epoch 200。**勿改**。

**跳过训练选项**：若不想重训 v5，可拷入已有 `swa_model.pth`，直接执行 1b/1c。v3 `best_model.pth` 可下载免训。

### 阶段 2 — C 层（经典描述子）

所有 run 脚本默认输出到 `results_final/<方法>/`，eval 脚本读缓存出指标；可无参运行。

```bash
# 2a. 3DZD 提取 + 1-NN（c1：3D Zernike Descriptor，体积门限 [0.8,1.2]）
python 02_c_layer/run_3dzd.py     # --workers 36 --out results_final/3dzd（默认）
python 02_c_layer/eval_3dzd.py
# 2b. FPFH 提取 + RBF-SVM（c2：FPFH L2-normalized + decision_function margin）
python 02_c_layer/run_fpfh_o3d.py # --workers 12 --out results_final/fpfh
python 02_c_layer/eval_fpfh.py
# 2c. Guerra 拓扑 + SimpleNN 8 视角硬投票（c3）
python 02_c_layer/run_guerra.py   # --workers 24 --out results_final/guerra
python 02_c_layer/eval_guerra.py   # --device cpu --max-epochs 5000
# 2d. SVM 成员 c1d / c4（c4 = [363 ZDZ | 612 FPFH] 块标准化拼接 + RBF-SVM C16/γ.003）
python 02_c_layer/tune_svm_members.py
```

**校验点 4**：C 层产出与参考指标（`configs/c_members.yaml` 冻结值）
```bash
ls results_final/   # 应见 3dzd/ fpfh/ guerra/ 等
```
| 成员 | 方法 | test acc / bal / mac |
| --- | --- | --- |
| c1 | 3DZD 1-NN+体积门限 | 0.9100 / 0.8914 / 0.8867 |
| c2 | FPFH+RBF-SVM | 0.9052 / 0.8570 / 0.8401 |
| c4 | [ZDZ\|FPFH]+RBF-SVM | 0.9324 / 0.8633 / 0.8822 |

### 阶段 3 — 概率空间后融合

```bash
python 03_fusion/fuse_mcfps2.py
```
`fuse_mcfps2.py`：在 val 上做 one-SE nested-K 选 K=25、配对 bootstrap 对照冻结生产 val 预测，并对赢家出一次 test 报告。融合公式（`configs/fusion_final.yaml`）：
```
p = 2·softmax(v5_mc_mean_logp) + 2·softmax(c1_dist/0.5) + 2·softmax(c2_margin/1.0)
```
> **最终生产预测已冻结在 `expected_outputs/test_full_fusion.csv`**，脚本不会自动覆盖该文件。

### 阶段 4 — 分析与验证

```bash
python 04_analysis/compute_final_metrics.py      # 验证最终 test 数字
python 01_a_layer/calibration_analysis.py        # ECE/AURC/BALD（需 25-draw cache）
python 04_analysis/make_calibration_fig.py       # 可靠性图 + risk-coverage 图
<py_numpy2> 01_a_layer/deterministic_init_eval.py  # D0/Ds/随机/K25（需 numpy>=2）
python 01_a_layer/split_sensitivity.py           # 划分敏感性（需 1d v3 clean cache）
python 01_a_layer/member_overlap_rho.py          # 成员错误重叠 ρ（缺缓存自动跳过项）
python 04_analysis/make_framework_fig.py         # Fig.1 框架图（纯绘制无依赖）
python 04_analysis/make_variance_fig.py          # MC-FPS 各 pass 方差图
python 04_analysis/make_k_curve_fig.py           # nested-K val 曲线
python 04_analysis/make_winners_curse_fig.py      # 赢者诅咒示意
python 04_analysis/make_confusion_matrix.py      # 97×97 混淆矩阵
python 04_analysis/make_per_class_table.py       # LaTeX 每类表
```
图写入 `figures/`；每类表打印 stdout 并写 `figures/per_class_table.tex`；
标定/确定性汇总写 `results_final/calibration_summary.json`、
`results_final/deterministic_init_summary.json`。

### 阶段 5 — 跨架构复现（可选，ModelNet40）

需要两个外部资源（本包不分发）：
- **ModelNet40（normal\_resampled）**：公开镜像
  `https://hf-mirror.com/datasets/Pointcept/modelnet40_normal_resampled-compressed`
  下载 `modelnet40_normal_resampled.tar.gz`（约 1.7 GB；若截断用 `curl -C -` 续传，
  压缩包无顶层目录，解压时 `mkdir modelnet40_normal_resampled && tar xzf ... -C modelnet40_normal_resampled`）。
- **PointNet++ 代码与发布权重**：
  `git clone https://gh-proxy.com/https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git`
  （仓库内已含 SSG/MSG 的 `best_model.pth`）。

运行（需要 numpy>=2 + torch 的环境）：
```bash
MODELNET40_ROOT=<modelnet40_normal_resampled> \
POINTNET2_REPO=<Pointnet_Pointnet2_pytorch> \
SHARP_PYTHON=<py_numpy2> STAGE=5 ./run_all.sh
```

**校验点 5（最终）**：
```bash
python 04_analysis/compute_final_metrics.py
# 期望输出：
#   accuracy      : 94.79
#   balanced acc  : 91.48
#   macro-F1      : 92.25
#   PASS: metrics match the pre-registered target (tol=0.05).
```

**校验点 6（阶段 5）**：`05_cross_arch/routeb_summary.json` 应近似
SSG：single acc .9271±.0017 → K25 .9279，flip .0535，ECE .0197→.0169；
MSG：.9239±.0017 → .9263，flip .0527，ECE .0397→.0393（MI≈0.0048 nats）。

---

## 4. 最小验证路径（不训练，仅验证冻结结果）

若只想确认冻结预测能复现论文数字（无需 GPU/数据/训练）：
```bash
cd replication
python -m pip install -r requirements.txt      # 纯 CPU 也需 sklearn/pandas
STAGE=4 ./run_all.sh        # 或直接 python 04_analysis/compute_final_metrics.py
```
此路径只读 `expected_outputs/test_full_fusion.csv` + `00_data/splits/test_set_ground_truth.csv`，应得 PASS。

---

## 5. 故障排查矩阵

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `[守卫] new_data/ 下未找到 *_pc.npy` | 点云未就位 | 按 §2.2 下载/提取点云到 `new_data/` |
| `FileNotFoundError: log/.../swa_model.pth` | 未训练或路径不对 | 跑阶段 1a 训练，或拷入已有 ckpt；`SKIP_TRAIN=1` 仅在已有 ckpt 时用 |
| MC-FPS 多次结果不同 | FPS 随机（`riconv2_utils_features.py:88` `torch.randint`） | **预期行为**；生产用 K=25 平均消除方差，勿报单次结果 |
| `softmax` 出 NaN，融合崩塌 | c1 1-NN 距离含 `inf`（体积门限全拒） | 送入 softmax 前把 `inf` 换大有限值（如 `1e6` 或最大有限距离） |
| SVM 成员拖垮融合 / macro 大跌 | 误用 Platt scaling | **禁用** `SVC(probability=True)`；一律用 `decision_function` margin（见 `configs/c_members.yaml`） |
| c3 test 精度偏低 ~1 点 | 未在全 9244 行 refit | `c3`/`c3b` 必须在**全 9244 行**训练集 refit，非仅 7428 行内部划分（`clean_refit_on_full_9244: true`） |
| 训练欠拟合，test<87.5% | 激进类权重 | 必用 sqrt-inverse-frequency，clip 到 10:1；勿用 Class-Balanced β=0.999（权重比 800+:1） |
| val 偏高但 test 否定 | 误用分层划分 | `train_set_2.csv` 按行序 7428/1816，**禁止分层划分** |
| val 小幅领先但 test 反转 | 赢者诅咒 | 用 one-SE nested-K + 配对 bootstrap 稳健口径，勿取单次幸运 draw |
| `run_fpfh_o3d.py` ImportError | open3d 未装 | `pip install -r requirements.txt`（已含 open3d 0.13.0；py3.8 装到此版） |
| 学习率异常（续训） | 多次 `scheduler.step()` 推进 | 续训时用 `last_epoch` 直接定位断点，勿循环 step |
| Route B/C 前向结果错误或脚本断言退出 | numpy 1.x 加载 numpy2 pickle（静默损坏）/ 版本断言 | 用 numpy>=2 + torch 的独立环境，通过 `SHARP_PYTHON` 指定 |

---

## 6. 脚本 / 配置速查

| 路径 | 用途 | 关键参数 |
| --- | --- | --- |
| `01_a_layer/train_v5.py` | v5 训练 | `--seed 2 --log_dir riconv_large_v5_s2`（全默认超参，见 `configs/v5.yaml`） |
| `01_a_layer/dump_mcfps.py` | MC-FPS K 次前向 | `--log_dir <ld> --ckpt swa_model.pth --K 25 --prefix mcfps_v5s2` |
| `01_a_layer/dump_v5.py` | 单次前向 | `--log_dir <ld> --ckpt swa_model.pth --tag s2_swa` |
| `01_a_layer/split_sensitivity.py` | §III-D 划分敏感性 | 无参（读 v3 clean cache，不重训） |
| `01_a_layer/member_overlap_rho.py` | §V-B/E 错误重叠 ρ | 无参（缺缓存自动跳过对应项） |
| `04_analysis/make_framework_fig.py` | Fig.1 框架图 | 无参（纯绘制，无需数据/GPU） |
| `02_c_layer/run_3dzd.py` | 3DZD 提取 | `--workers 36 --out results_final/3dzd`（默认） |
| `02_c_layer/eval_3dzd.py` | 3DZD 1-NN+门限 | 无参 |
| `02_c_layer/run_fpfh_o3d.py` | FPFH 提取 | `--workers 12 --out results_final/fpfh` |
| `02_c_layer/eval_fpfh.py` | FPFH+SVM | 无参 |
| `02_c_layer/run_guerra.py` | Guerra 拓扑 | `--workers 24 --out results_final/guerra` |
| `02_c_layer/eval_guerra.py` | SimpleNN 8 视角 | `--device cpu --max-epochs 5000` |
| `02_c_layer/tune_svm_members.py` | c1d/c4 SVM | 无参 |
| `03_fusion/fuse_mcfps2.py` | 融合（生产入口） | 无参（读 `results_final/cache`） |
| `01_a_layer/calibration_analysis.py` | §V-C ECE/AURC/BALD | 无参（需 25-draw cache + pandas/sklearn） |
| `01_a_layer/deterministic_init_eval.py` | §V-D D0/Ds 对照 | `--gpu 0`（需 numpy>=2） |
| `05_cross_arch/route_b_eval.py` | §V-E 跨架构 25-draw | `--data_root <mn40> --repo_root <pn2> --K 25`（需 numpy>=2） |
| `04_analysis/compute_final_metrics.py` | 最终验证 | 无参 |

| 配置 | 内容 |
| --- | --- |
| `configs/v5.yaml` | lr1e-3, wd1e-4, T_max245, eta_min1e-5, warmup5, focal γ1.0, cutmix0.2, class_weight sqrt_inverse clip10, swa_start200, seeds[1,2] |
| `configs/c_members.yaml` | c1(3DZD 1-NN+体积门限[0.8,1.2]) / c1d / c2(FPFH+SVM C64 γ12) / c3(Guerra) / c4([363\|612] concat + SVM C16 γ.003)，decision_function:true |
| `configs/fusion_final.yaml` | K=25, 权重 2:2:2, c1 τ=.5, c2 τ=1.0；p=2·softmax(v5_mc)+2·softmax(c1/0.5)+2·softmax(c2/1.0) |

---

## 7. 与论文章节 / 表的对应

| 论文内容 | 复现位置 |
| --- | --- |
| 最终融合指标（实验 §7） | `expected_outputs/test_full_fusion.csv` + `compute_final_metrics.py` |
| 单模型 / 融合级消融表（tab:ablation） | `01_a_layer` dump 结果 + `03_fusion/fuse_mcfps2.py` 同口径重算 |
| 赢者诅咒示意 | `04_analysis/make_winners_curse_fig.py` → `figures/winners_curse.pdf` |
| MC-FPS pass 方差 | `04_analysis/make_variance_fig.py` |
| nested-K val 曲线 | `04_analysis/make_k_curve_fig.py` |
| 每类精度 / 混淆 | `make_per_class_table.py` / `make_confusion_matrix.py` → `figures/confusion_matrix.pdf` |
| 划分敏感性（§III-D） | `01_a_layer/split_sensitivity.py`（需 1d v3 clean cache） |
| 成员错误重叠 ρ（§V-B/E） | `01_a_layer/member_overlap_rho.py` → 0.281/0.313/0.338/0.672/0.198 等 |
| Fig.1 系统框架图 | `04_analysis/make_framework_fig.py` → `figures/framework.pdf` |
| ECE/AURC/BALD（§V-C） | `01_a_layer/calibration_analysis.py` + `04_analysis/make_calibration_fig.py` → `figures/calibration.pdf` |
| 确定性初始点对照（§V-D） | `01_a_layer/deterministic_init_eval.py` → `results_final/deterministic_init_summary.json` |
| 跨架构 ModelNet40（§V-E） | `05_cross_arch/route_b_eval.py` → `05_cross_arch/routeb_summary.json` |
