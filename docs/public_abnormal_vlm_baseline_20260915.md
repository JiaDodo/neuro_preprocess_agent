# 公开异常数据 VLM 零样本基线（2026-09-15）

## 测试目的

使用真实运动伪影和独立专家评分，检查本地 `Qwen2.5-VL-3B-Instruct` 能否区分
`pass / review / fail`。测试保持 shadow mode，不影响流水线 QC gate 或 MySQL 入库。

## 数据

- 专家集：35 例。
  - MR-ART：15 例，5 名被试的 standard/headmotion1/headmotion2 配对 T1w。
  - ds004332：20 例，2 名被试的静止、点头、摇头及运动校正组合 T1w。
- 弱标签观察池：30 例 ABIDE/PCP 评分者分歧样本，不参与准确率计算。
- 所有 65 例均成功完成结构化推理，无缺失参考标签。

## 第一轮：错误复用 fMRIPrep 提示词

第一轮直接使用面向 fMRIPrep 掩膜和配准面板的提示词评审原始 MRI：

- 专家集 accuracy：0.2857
- macro-F1：0.2013
- 异常敏感度：1.0000
- 正常特异度：0.0000
- 模型没有预测任何 pass。

模型在不存在红色轮廓时仍描述了红色 mask overlay，说明输入任务和提示词不匹配。
该轮结果保留为接口错误基线。

## 第二轮：原始 MRI 专用提示词

第二轮明确限定为 source MRI，只允许判断运动、ghosting/dropout 和覆盖问题；禁止判断
脑掩膜、配准、标准化和分割，并明确说明面部脱敏不是异常。

### 总体专家集

| 指标 | 结果 |
|---|---:|
| 样本数 | 35 |
| accuracy | 0.2857 |
| macro-F1 | 0.2035 |
| 异常敏感度 | 1.0000 |
| 正常特异度 | 0.0000 |
| balanced accuracy | 0.5000 |

混淆矩阵（行是真值，列是预测）：

| 真值 | pass | review | fail |
|---|---:|---:|---:|
| pass | 0 | 1 | 16 |
| review | 0 | 1 | 8 |
| fail | 0 | 0 | 9 |

### 分数据集

- MR-ART：accuracy 0.4667，fail recall 1.0，但 pass recall 0。
- ds004332：accuracy 0.15，20 例全部预测为 fail。
- ABIDE 弱标签池：30 例全部预测为 fail；由于其真值只是评分分歧，不计算准确率。

## 结论

当前 3B 通用 VLM 存在极强的过度报警偏差。它能覆盖所有明显异常，却完全不能识别正常图像，
因此没有实际分流价值。提示词修正消除了“红色轮廓”幻觉，但没有改善正常特异度，继续微调
提示词的收益预计有限。

另外，模型原始 `model_decision` 在 65 例中全部为 `review`；当前后处理规则会把包含 high
severity 异常观察的样本升级为 `fail`，因而最终产生 63 个 fail。即使不升级，模型仍不会产生
pass，所以问题主体是模型判别能力而非后处理阈值。

## 下一步

1. 保持 VLM 为 shadow mode，不接入自动阻断。
2. 扩充正常样本，并保证按被试划分数据，避免同一被试泄漏。
3. 优先训练三分类或二阶段模型：第一阶段区分 pass/abnormal，第二阶段区分 review/fail。
4. 将专家分数作为有序标签，尝试 LoRA 微调或视觉 embedding 分类器。
5. 单独评估 MR-ART 与 ds004332，避免模型只学习扫描协议或站点差异。

## 结果文件

- 第一轮：`data/visual_qc/vlm_reviews/public_abnormal_qwen25vl3b/metrics.json`
- 第二轮：`data/visual_qc/vlm_reviews/public_abnormal_qwen25vl3b_source_prompt/metrics.json`
- 第二轮完整预测：`data/visual_qc/vlm_reviews/public_abnormal_qwen25vl3b_source_prompt/predictions.jsonl`
