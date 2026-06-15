# DeepVul-Lab 🔍

**面向 C/C++ 函数级漏洞检测的严谨评估与组合提升流水线**

DeepVul-Lab 在 Devign 与 DiverseVul 两个公开基准上，提供一条可复现的端到端实验流水线：从去重与防泄漏划分，到多骨干微调、阈值校准与多种子集成，系统审计常见"提升手段"的真实有效性，并构建稳健超越 CodeBERT 基线的组合检测系统。

---

## ✨ Key Features

- **严谨数据治理** — 精确去重（SHA-1）+ 近重复去重（MinHash LSH），DiverseVul 采用项目级 80/10/10 划分，杜绝记忆泄漏与风格泄漏
- **统一实验协议** — 两数据集归一化为同一 JSONL 模式，所有骨干共享一致的微调超参与评价指标（F1、MCC、PR-AUC 等）
- **多骨干横向对比** — 支持 CodeBERT、VulBERTa、GraphCodeBERT、UniXcoder 及关键上下文切片输入
- **不平衡鲁棒训练** — 逆频率类别加权、Focal Loss、随机过采样，适配 DiverseVul 约 7% 正样本率
- **零额外训练的后处理** — 验证集阈值校准与 3 种子软投票集成，降低种子方差
- **可复现结果汇总** — 一键编译对比表、最优系统提升阶梯与可视化图表

---

## 🏗 Architecture

```
DeepVul-Lab/
├── code/
│   ├── src/                        # 核心库
│   │   ├── data/                   # 数据加载、去重、划分、切片
│   │   ├── training/experiment.py  # 统一微调入口（Trainer 封装）
│   │   ├── metrics.py              # 评价指标与阈值搜索
│   │   └── utils/gpu.py            # 自动选取空闲 GPU
│   ├── scripts/                    # 命令行工具
│   │   ├── build_dataset.py        # ① 构建去重 + 划分后的 JSONL
│   │   ├── build_sliced.py         # ② 生成关键上下文切片字段
│   │   ├── train.py                # ③ 多种子微调
│   │   ├── tune_threshold.py       # ④ 验证集阈值校准
│   │   ├── ensemble.py             # ⑤ 多种子软投票集成
│   │   ├── compile_results.py      # 汇总全部实验对比表
│   │   └── build_best_system.py    # 构建最优系统提升阶梯
│   ├── data/processed/             # 处理后数据集（JSONL + stats.json）
│   └── outputs/                    # 实验结果、图表与报告
├── report/                         # 期末实验报告
└── LICENSE
```

**流水线概览：**

```
原始数据集 → 去重 & 防泄漏划分 → [可选] 关键上下文切片
          → 多种子微调 → 阈值校准 / 软投票集成 → 结果汇总
```

---

## 🚀 Getting Started

### 1. 克隆仓库

```bash
git clone https://github.com/<your-org>/DeepVul-Lab.git
cd DeepVul-Lab
```

### 2. 创建环境并安装依赖

```bash
conda create -n syssec_env python=3.10 -y
conda activate syssec_env

pip install torch transformers datasets numpy scipy matplotlib datasketch
```

### 3. 预下载数据集与预训练模型

脚本默认以离线模式运行（`HF_HUB_OFFLINE=1`），需提前缓存至本地：

```bash
# 数据集
huggingface-cli download DetectVul/devign
huggingface-cli download claudios/DiverseVul

# 预训练骨干（按需下载）
huggingface-cli download microsoft/codebert-base
huggingface-cli download claudios/VulBERTa-mlm
huggingface-cli download microsoft/graphcodebert-base
huggingface-cli download microsoft/unixcoder-base
```

### 4. 快速运行示例

```bash
# 构建处理后数据集（无需 GPU）
python code/scripts/build_dataset.py

# 在 Devign 上微调 CodeBERT 基线（自动选取空闲 GPU）
python code/scripts/train.py \
    --model microsoft/codebert-base \
    --dataset devign \
    --seeds 42 1 2 \
    --tag codebert_fullfunc
```

---

## 📖 Usage

所有命令均在仓库根目录、已激活 `syssec_env` 的环境下执行。

### 数据准备

```bash
# 去重 + 划分，输出至 code/data/processed/<dataset>/
python code/scripts/build_dataset.py

# 为每条记录追加 func_sliced / func_marked 字段（RQ2 关键上下文实验）
python code/scripts/build_sliced.py
```

### 模型训练

```bash
# 基线：CodeBERT + 类别加权 CE
python code/scripts/train.py \
    --model microsoft/codebert-base \
    --dataset diversevul \
    --seeds 42 1 2 \
    --tag codebert_fullfunc

# 结构感知骨干：UniXcoder
python code/scripts/train.py \
    --model microsoft/unixcoder-base \
    --dataset diversevul \
    --seeds 42 1 2 \
    --tag unixcoder_fullfunc

# 更强不平衡处理：Focal Loss
python code/scripts/train.py \
    --model microsoft/unixcoder-base \
    --dataset diversevul \
    --loss-type focal \
    --seeds 42 1 2 \
    --tag unixcoder_focal

# 关键上下文切片输入（需先运行 build_sliced.py）
python code/scripts/train.py \
    --model microsoft/codebert-base \
    --dataset diversevul \
    --text-field func_sliced \
    --seeds 42 1 2 \
    --tag codebert_sliced

# VulBERTa 需信任远程代码
python code/scripts/train.py \
    --model claudios/VulBERTa-mlm \
    --dataset diversevul \
    --trust-remote-code \
    --seeds 42 1 2 \
    --tag vulberta_fullfunc
```

### 后处理与结果汇总

```bash
# 验证集阈值校准（不重训练）
python code/scripts/tune_threshold.py \
    --run-dir code/outputs/runs/diversevul_codebert_fullfunc \
    --model microsoft/codebert-base \
    --dataset diversevul \
    --metric mcc

# 3 种子软投票集成
python code/scripts/ensemble.py \
    --run-dir code/outputs/runs/diversevul_unixcoder_focal

# 汇总全部实验为对比表（Markdown / CSV / 柱状图）
python code/scripts/compile_results.py

# 构建最优系统提升阶梯（S0→S4）
python code/scripts/build_best_system.py --threshold-metric mcc

# 绘制提升阶梯图
python code/scripts/plot_best_system.py
```

### 主要 CLI 参数速查

| 脚本 | 关键参数 | 说明 |
|---|---|---|
| `train.py` | `--model`, `--dataset`, `--tag`, `--seeds` | 微调骨干并聚合多种子指标 |
| `train.py` | `--loss-type`, `--sampler` | `focal` / `oversample` 等不平衡策略 |
| `train.py` | `--text-field` | 输入字段：`func`（默认）或 `func_sliced` |
| `tune_threshold.py` | `--metric` | 验证集上优化的目标：`f1` 或 `mcc` |
| `ensemble.py` | `--tune-metric` | 可选：在集成概率上再调阈值 |

训练产物保存在 `code/outputs/runs/<dataset>_<tag>/`，包含 `summary.json`、各种子 `metrics.json` 与 `test_logits.npy`。

---

## 💻 Requirements

### 硬件

| 组件 | 建议配置 |
|---|---|
| GPU | NVIDIA GPU，显存 ≥ 20 GB（RoBERTa-base 级骨干，batch size 32） |
| 内存 | ≥ 32 GB（DiverseVul 去重与分词缓存） |
| 磁盘 | ≥ 50 GB（数据集缓存 + 分词缓存 + 检查点） |

### 软件

| 依赖 | 用途 |
|---|---|
| Python ≥ 3.10 | 运行环境 |
| PyTorch | 模型训练与推理 |
| Hugging Face `transformers` / `datasets` | 预训练模型与数据集加载 |
| NumPy / SciPy | 数值计算与概率变换 |
| Matplotlib | 混淆矩阵与对比图表 |
| datasketch | MinHash LSH 近重复去重 |

### 数据集

| 数据集 | Hugging Face 仓库 | 特点 |
|---|---|---|
| Devign | `DetectVul/devign` | FFmpeg + QEMU，近均衡（正样本率 ~46%） |
| DiverseVul | `claudios/DiverseVul` | 310 个项目，强不平衡（正样本率 ~7%） |

---

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。
