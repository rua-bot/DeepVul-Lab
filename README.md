# DeepVul-Lab

> 🔬 面向 C/C++ 函数级漏洞检测的受控实证研究 —— 在严格评估协议下，系统比较领域预训练、结构感知骨干、关键上下文抽取与阈值校准等提升路径的真实效果。

---

## ✨ 核心亮点

- **方法学严谨**：统一数据模式、精确 + 近重复去重、项目级划分防泄漏，杜绝"记忆样本"带来的虚高指标
- **多骨干横向对比**：支持 CodeBERT、VulBERTa、GraphCodeBERT、UniXcoder 等预训练模型在同一协议下公平评测
- **类别不平衡友好**：逆频率类别加权交叉熵 + F1 / MCC / PR-AUC 等稳健指标，避免 Accuracy 误导
- **多种子统计聚合**：默认 3 个随机种子（42 / 1 / 2），报告均值 ± 标准差，区分真实提升与运行噪声
- **关键上下文抽取**：轻量静态启发式切片，无需外部解析器，缓解 512-token 截断问题
- **决策阈值校准**：复用已训练检查点，在验证集上搜索最优 F1 / MCC 阈值，无需重训练
- **端到端可复现**：离线模式运行、分词结果磁盘缓存、逐种子指标与聚合摘要完整留存

---

## 🏗️ 项目架构

```
DeepVul-Lab/
├── code/                              # 核心代码与实验产物
│   ├── scripts/                       # 可执行脚本（流水线入口）
│   │   ├── user_setup.sh              # 环境初始化：安装依赖、下载数据集与模型
│   │   ├── build_dataset.py           # 数据治理：加载 → 去重 → 划分 → 写出 JSONL
│   │   ├── build_sliced.py            # 关键上下文抽取：生成 func_sliced / func_marked 字段
│   │   ├── analyze_lengths.py         # 函数 token 长度分布与截断率分析
│   │   ├── train.py                   # 多随机种子微调与指标聚合
│   │   ├── tune_threshold.py          # 验证集阈值搜索 + 测试集报告
│   │   ├── compile_results.py         # 汇总所有实验 run，生成对比表与柱状图
│   │   ├── smoke_train.py             # 端到端冒烟测试（验证训练栈可用性）
│   │   ├── vulberta_load_test.py      # VulBERTa 兼容性检测
│   │   └── tok_diag.py                # Tokenizer 诊断工具
│   ├── src/                           # 可复用库模块
│   │   ├── data/
│   │   │   ├── loaders.py             # Devign / DiverseVul 统一加载器
│   │   │   ├── dedup.py               # 精确去重（SHA-1）+ 近重复去重（MinHash LSH）
│   │   │   ├── splitting.py           # 项目级划分与标签统计
│   │   │   ├── slicing.py             # 关键上下文抽取与结构标记
│   │   │   └── jsonl_dataset.py       # 处理后 JSONL 数据读取
│   │   ├── training/
│   │   │   └── experiment.py          # 单次微调实验（分词缓存、类别加权、Trainer 封装）
│   │   ├── metrics.py                 # F1 / MCC / PR-AUC 等指标计算
│   │   └── utils/
│   │       └── gpu.py                 # 共享服务器动态 GPU 选择
│   ├── data/
│   │   └── processed/                 # 去重 + 划分后的 JSONL 数据集
│   └── outputs/
│       ├── runs/                      # 各实验配置的多种子结果（summary.json、metrics.json）
│       ├── results_table.md           # 跨实验对比表
│       └── length_stats.json          # 截断分析统计
├── report/
│   └── 期末实验报告.md                 # 完整实验报告（研究问题、方法、结果与分析）
└── outputs/
    └── logs/                          # 训练日志
```

### 流水线概览

```mermaid
flowchart LR
    A[原始数据集] --> B[统一模式加载]
    B --> C[去重 + 防泄漏划分]
    C --> D[关键上下文抽取<br/>可选]
    D --> E[类别加权微调<br/>3 seeds]
    E --> F[阈值校准<br/>可选]
    F --> G[指标聚合与对比]
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<your-org>/DeepVul-Lab.git
cd DeepVul-Lab
```

### 2. 创建并激活 Conda 环境

```bash
conda create -n deepvul_env python=3.10 -y
conda activate deepvul_env
```

### 3. 安装核心依赖

```bash
pip install torch transformers datasets huggingface_hub \
    scikit-learn scipy numpy matplotlib datasketch libclang
```

> 推荐使用 `transformers >= 5.1.0`。若使用 HuggingFace 镜像，可设置：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 4. 一键初始化（下载数据集与模型）

```bash
bash code/scripts/user_setup.sh
```

该脚本将自动完成：
- 安装 `libclang`（VulBERTa 自定义 tokenizer 所需）与 `datasketch`（近重复去重）
- 下载 [DiverseVul](https://huggingface.co/datasets/claudios/DiverseVul) 数据集
- 下载 [VulBERTa-mlm](https://huggingface.co/claudios/VulBERTa-mlm) 预训练权重

### 5. 验证环境

```bash
# 冒烟测试：验证数据 → 分词 → 训练 → 评估链路
python code/scripts/smoke_train.py

# VulBERTa 兼容性检测
python code/scripts/vulberta_load_test.py
```

---

## 📖 使用方法

### 数据预处理

```bash
# 构建去重 + 划分后的数据集（输出至 code/data/processed/）
python code/scripts/build_dataset.py

# 可选：生成关键上下文切片字段 func_sliced
python code/scripts/build_sliced.py

# 分析函数长度与截断率
python code/scripts/analyze_lengths.py
```

### 模型训练

以下命令均在项目根目录下执行，且需先 `conda activate deepvul_env`。

**CodeBERT 基线（完整函数）**

```bash
python code/scripts/train.py \
    --model microsoft/codebert-base \
    --dataset devign \
    --seeds 42 1 2 \
    --tag codebert_fullfunc
```

**UniXcoder（结构感知预训练）**

```bash
python code/scripts/train.py \
    --model microsoft/unixcoder-base \
    --dataset diversevul \
    --seeds 42 1 2 \
    --tag unixcoder_fullfunc
```

**关键上下文切片变体**

```bash
python code/scripts/train.py \
    --model microsoft/codebert-base \
    --dataset diversevul \
    --text-field func_sliced \
    --seeds 42 1 2 \
    --tag codebert_sliced
```

**VulBERTa（领域自适应，需 trust-remote-code）**

```bash
python code/scripts/train.py \
    --model claudios/VulBERTa-mlm \
    --dataset devign \
    --trust-remote-code \
    --tokenize-num-proc 4 \
    --seeds 42 1 2 \
    --tag vulberta_fullfunc
```

训练结果保存在 `code/outputs/runs/<dataset>_<tag>/`，包含：
- `summary.json`：多种子聚合指标（均值 ± 标准差）
- `seed*/metrics.json`：逐种子测试集指标
- `confusion_seed_first.png`：混淆矩阵可视化

### 决策阈值校准

```bash
python code/scripts/tune_threshold.py \
    --run-dir code/outputs/runs/diversevul_codebert_fullfunc \
    --model microsoft/codebert-base \
    --dataset diversevul \
    --text-field func \
    --metric f1
```

### 汇总实验结果

```bash
python code/scripts/compile_results.py
```

输出 `code/outputs/results_table.md`、`results_table.csv` 及分组柱状图，便于跨实验横向对比。

### 主要命令行参数

| 脚本 | 关键参数 | 说明 |
|---|---|---|
| `train.py` | `--model` | HuggingFace 模型 ID 或本地路径 |
| | `--dataset` | `devign` 或 `diversevul` |
| | `--tag` | 实验标签，用于输出目录命名 |
| | `--text-field` | 输入字段：`func`（默认）/ `func_sliced` / `func_marked` |
| | `--seeds` | 随机种子列表，默认 `42 1 2` |
| | `--max-len` | 最大序列长度，默认 512 |
| | `--epochs` | 训练轮数，默认 4 |
| `tune_threshold.py` | `--run-dir` | 已训练实验的输出目录 |
| | `--metric` | 优化目标：`f1` 或 `mcc` |

---

## 📊 评价指标

鉴于漏洞数据集普遍存在类别不平衡，本项目以以下指标为主：

| 指标 | 含义 |
|---|---|
| **F1** | 查准率与查全率的调和平均 |
| **MCC** | Matthews 相关系数，对混淆矩阵四象限均敏感 |
| **PR-AUC** | 平均精度，刻画与阈值无关的排序能力 |
| **ROC-AUC** | 受试者工作特征曲线下面积 |

类别加权损失函数基于训练集逆频率：

$$\mathcal{L} = -\sum_{i} w_{y_i} \log p(y_i \mid x_i), \quad w_c = \frac{N}{K \cdot n_c}$$

其中 $N$ 为训练样本总数，$K$ 为类别数，$n_c$ 为类别 $c$ 的样本数。

---

## 💻 环境要求

### 硬件

| 组件 | 最低要求 | 推荐配置 |
|---|---|---|
| **GPU** | NVIDIA GPU，显存 ≥ 16 GB | NVIDIA A100（32 GB+） |
| **内存** | 32 GB RAM | 64 GB+ |
| **磁盘** | 50 GB 可用空间 | 100 GB+（含数据集与模型缓存） |

### 软件

| 依赖 | 版本建议 |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+（含 CUDA 支持） |
| transformers | ≥ 5.1.0 |
| datasets | ≥ 2.14 |
| scikit-learn | ≥ 1.3 |
| scipy | ≥ 1.11 |
| libclang | 最新版（VulBERTa 必需） |
| datasketch | 最新版（近重复去重） |
| CUDA | 11.8+ / 12.x |

### 数据集

| 数据集 | 来源 | 特点 |
|---|---|---|
| **Devign** | [DetectVul/devign](https://huggingface.co/datasets/DetectVul/devign) | FFmpeg + QEMU，类别近似均衡（~45% 正样本） |
| **DiverseVul** | [claudios/DiverseVul](https://huggingface.co/datasets/claudios/DiverseVul) | 310 个项目，强不平衡（~7% 正样本） |

> 所有脚本默认以离线模式运行（`HF_HUB_OFFLINE=1`），请确保数据集与模型已预先下载至本地 HuggingFace 缓存。

---

## 📄 实验报告

完整的实验设计、结果分析与结论请参阅 [`report/期末实验报告.md`](report/期末实验报告.md)。

核心发现摘要：
- 在去重与项目级划分的硬协议下，**结构感知预训练（UniXcoder）** 带来跨数据集一致的稳健提升
- 领域自适应预训练（VulBERTa）未能优于通用 CodeBERT 基线
- 关键上下文抽取虽降低截断率，但未转化为指标提升
- 决策阈值校准仅移动工作点，无法改善排序能力（PR-AUC）

---

## 📜 License

本项目采用 [MIT License](LICENSE) 开源协议。

```
MIT License

Copyright (c) 2026 DeepVul-Lab Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

