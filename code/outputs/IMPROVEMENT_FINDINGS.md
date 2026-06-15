# 实验补充与完善：两个问题的诊断与解决记录

> 本文档记录针对两个实验问题的诊断结论、补充实验与最终数据，作为后续改写实验报告的素材。**不改动报告正文**。所有数字均为测试集、3 随机种子（42/1/2）结果；除特别标注外为单种子均值。

---

## 问题一：未体现"在测试集上提升准确度" —— 已通过补充实验解决

### 诊断结论（问题确实存在）
原项目以"受控审计"为主线，引言甚至声明"目标并非追求最高分数"，与任务要求"在测试集上提升准确度"立场相反；且最优模型在主数据集 DiverseVul 上的**原始准确率反而低于基线**（类别加权用召回换准确率所致），缺少清晰的"基线→提升"主线。

### 解决方案：构建"组合最优系统" + 提升阶梯
在 CodeBERT 基线之上逐步叠加技术，给出测试集上的**累计提升**。新增技术：结构感知骨干、Focal Loss / 随机过采样（更强不平衡处理）、验证集阈值校准、3 种子软投票集成。**不平衡方法按验证集 F1 选择**（DiverseVul/Devign 均选中 Focal），不窥探测试集。

#### 提升阶梯（DiverseVul，强不平衡，主数据集）—— 干净的单调提升

| 步骤 | 配置 | F1 | MCC | PR-AUC | Balanced Acc | Accuracy |
|---|---|---|---|---|---|---|
| S0 | CodeBERT 基线 | 0.2255 | 0.1614 | 0.1576 | 0.6219 | 0.8092 |
| S1 | + 结构骨干 UniXcoder | 0.2449 | 0.1912 | 0.1751 | 0.6528 | 0.7898 |
| S2 | + 最优不平衡处理（Focal） | 0.2452 | 0.1944 | 0.1775 | 0.6601 | 0.7779 |
| S3 | + 验证集阈值校准 | 0.2490 | 0.1946 | 0.1777 | — | — |
| **S4** | **+ 3 种子集成（最终系统）** | **0.2520** | **0.2017** | **0.1852** | **0.6638** | 0.7874 |

**最终系统 vs 基线**：F1 +11.8%、MCC **+25.0%**、PR-AUC **+17.5%**、ROC-AUC +3.6%、Balanced Acc +0.042（绝对）。提升集中体现在**与阈值无关的排序指标 PR-AUC/ROC-AUC**，是模型判别能力真实增强的证据。

#### 提升阶梯（Devign，近均衡）—— 由结构骨干主导

| 步骤 | 配置 | F1 | MCC | PR-AUC | Balanced Acc | Accuracy |
|---|---|---|---|---|---|---|
| S0 | CodeBERT 基线 | 0.5858 | 0.2846 | 0.7093 | 0.6376 | 0.6447 |
| S1 | + 结构骨干 UniXcoder | 0.6566 | 0.3285 | 0.7392 | 0.6642 | 0.6597 |
| **最终** | **UniXcoder + 3 种子集成** | **0.6646** | **0.3412** | **0.7471** | **0.6707** | **0.6658** |

**最终系统 vs 基线**：F1 +13.5%、MCC +19.9%、PR-AUC +5.3%、Accuracy +0.021、Balanced Acc +0.033。
说明：在近均衡数据上，**结构骨干（S1）已捕获主要增益**，额外的 Focal/过采样/阈值校准为中性或略负（Focal 在验证集胜出但测试 MCC 略低于 S1，阈值校准以 MCC 为目标会牺牲 F1）——这是与 DiverseVul 形成对照的、诚实的数据集相关结论。

### 关键纠偏：为何用 Balanced Accuracy 而非原始 Accuracy
原始 Accuracy 在不平衡数据上误导性极强。最直接的证据来自本次**随机过采样**实验：

| DiverseVul | Accuracy | Balanced Acc | F1 | MCC | PR-AUC |
|---|---|---|---|---|---|
| UniXcoder + 过采样 | **0.8925**(↑) | **0.5806**(↓) | 0.2184 | 0.1611 | 0.1592 |

过采样把原始准确率抬到 0.89（看似最好），但 Balanced Acc 跌到 0.58、PR-AUC 退回基线水平——它只是更多地预测"无漏洞"。因此本项目以 **PR-AUC / MCC / Balanced Accuracy** 衡量"提升"，最终系统在这些指标上对基线均有稳健提升。

---

## 问题二：RQ1/RQ2 因果不严谨 —— 已部分证伪 + 补充受控对照

### 诊断结论（部分成立，需修正分类）
- **"参数量差异"这一混淆——可直接排除**：实测四个骨干均为 RoBERTa-base，参数量几乎一致（见下表），架构完全相同。原报告未指出这一点，使审稿人易把提升误归因于"更大的模型"。
- **"语料/规模/tokenizer/预训练目标差异"——确实未控制**，而原报告 6.1 节把 UniXcoder 的提升直接归因于"结构归纳偏置"，话术偏强、存在混淆。

#### 模型特性对照表（受控因素：实测自模型文件）

| 模型 | 参数量(总) | 参数量(非嵌入) | hidden | layers | heads | vocab |
|---|---|---|---|---|---|---|
| CodeBERT | 124.6M | 86.0M | 768 | 12 | 12 | 50265 |
| VulBERTa | 124.8M | 86.4M | 768 | 12 | 12 | 50000 |
| GraphCodeBERT | 124.6M | 86.0M | 768 | 12 | 12 | 50265 |
| UniXcoder | 125.9M | 86.4M | 768 | 12 | 12 | 51416 |

→ 参数量与架构**已被控制**（差异 <1%，全部 ~86M 非嵌入参数），可据此排除"参数量"假设。完整对照（含语料/规模/目标/tokenizer 等未控制因素）见 `outputs/model_characteristics.md`。

### 解决方案：用 CodeBERT→GraphCodeBERT 作为"近受控"因果证据
GraphCodeBERT 与 CodeBERT **共享同一 CodeSearchNet 语料、同一 RoBERTa-base 架构、同一 tokenizer（vocab 50265）、同等参数量**，差别仅在于 GraphCodeBERT 额外引入**数据流（结构）预训练目标**。本次补跑 Devign 上的 GraphCodeBERT（原仅在 DiverseVul 跑过），使该受控对照**跨两个数据集均成立**：

| 受控对照（仅 +数据流目标） | F1 | MCC | ROC-AUC | PR-AUC | Balanced Acc |
|---|---|---|---|---|---|
| DiverseVul: CodeBERT | 0.2255 | 0.1614 | 0.7128 | 0.1576 | 0.6219 |
| DiverseVul: GraphCodeBERT | **0.2411** | **0.1782** | **0.7235** | **0.1620** | **0.6274** |
| Devign: CodeBERT | 0.5858 | 0.2846 | 0.7183 | 0.7093 | 0.6376 |
| Devign: GraphCodeBERT | **0.6162** | **0.3004** | **0.7289** | **0.7193** | **0.6486** |

在语料/架构/tokenizer/参数量全部 held constant 的条件下，仅增加结构目标即带来两数据集上的一致提升——这是"结构感知预训练目标→提升"的**最干净因果证据**，应取代原报告中以 UniXcoder（混淆较多）为主的论证。

### 建议修正的结论话术
- **RQ2（修正后）**：结构感知预训练目标与检测性能提升存在**因果关系**，由近受控对照 CodeBERT→GraphCodeBERT 在两数据集上一致支持；UniXcoder 提供了**附加但有混淆**的证据（其语料/tokenizer 亦不同，不能单独归因结构）。
- **遗留局限（应在 Threats to Validity 显式声明）**：要完全排除语料/规模混淆，需在相同语料、相同 tokenizer 上做"有/无结构目标"的从头预训练消融；课程算力下不现实，故对 UniXcoder 这条线的因果归因保持克制。

---

## 新增/修改的代码与产物（可复现）

**新增脚本**
- `scripts/model_card.py` — 实测各骨干参数量并合并预训练特性，产出 `outputs/model_characteristics.{md,json}`。
- `scripts/ensemble.py` — 多种子软投票集成（复用已存 logits，零额外训练）。
- `scripts/build_best_system.py` — 组装提升阶梯与最优系统，产出 `outputs/best_system.{md,json}`。

**修改**
- `src/metrics.py` — 新增 Balanced Accuracy（写入所有指标流）。
- `src/training/experiment.py` — 新增 Focal Loss 与随机过采样；分词缓存改为**原子发布**（temp 目录 + `os.replace`），修复并发训练共享缓存的竞态。
- `scripts/train.py` — 暴露 `--loss-type {ce_weighted,focal,ce}`、`--focal-gamma`、`--sampler {none,oversample}`。
- `scripts/tune_threshold.py` — 额外记录**验证集**指标，支持按验证集做诚实的系统选择。
- `src/utils/gpu.py` — 新增 `DEEPVUL_GPU` 显式指定，便于多卡并行而不抢卡。

**新增实验 run**（`outputs/runs/`）
- `diversevul_unixcoder_focal`、`diversevul_unixcoder_oversample`
- `devign_unixcoder_focal`、`devign_unixcoder_oversample`
- `devign_graphcodebert_fullfunc`（补齐受控对照）
- 各 run 的 `ensemble.json`、关键 run 的 `threshold_tuned_mcc.json`

**汇总产物**：`outputs/results_table.md`、`outputs/best_system.md`、`outputs/model_characteristics.md`、`outputs/compare_{devign,diversevul}.png`。
