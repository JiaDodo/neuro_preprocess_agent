# Embedding QC 首轮基线

## 实现

固定本地 DINOv2-small，使用源影像三视图生成 384 维 L2 归一化 embedding。
分类器为 StandardScaler + Logistic Regression，C=1、class_weight=balanced。
任务是 pass 与 abnormal（review/fail）二分类，固定概率阈值 0.5。
不参与现有流水线的 QC gate，未自动写入或排除数据。

## 数据与评测

仅使用 35 个专家标注 T1w，来自 MR-ART 的 5 人和 ds004332 的 2 人。
ABIDE 评分分歧样本不参与训练。

- 7 折按被试留一验证，同一个人的所有采集条件都在同一折。
- StandardScaler 和分类器仅在每折训练集拟合。
- 两个跨数据集验证方向：在 MR-ART 训练、ds004332 测试，以及反向测试。
- 未使用测试数据选择 C 或阈值。

| 评测 | ROC-AUC | 敏感度 | 特异度 | Balanced accuracy |
|---|---:|---:|---:|---:|
| 按被试留一 | 0.8791 | 0.7222 | 0.6471 | 0.6846 |
| MR-ART 作为测试集 | 0.9000 | 0.6000 | 0.8000 | 0.7000 |
| ds004332 作为测试集 | 0.9479 | 0.8750 | 0.5833 | 0.7292 |

留一验证计数：TP=13、TN=11、FP=6、FN=5。
VLM 的二分类 balanced accuracy 为 0.5、正常特异度为 0；embedding 基线已能识别一部分正常样本。
VLM 三分类 accuracy 不应直接与本实验的二分类指标比较。

## 结果文件与复现

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/train_qc_embedding_baseline.py
```

输出目录：`data/visual_qc/embedding_baseline/`

- `embedded_manifest.jsonl` 与 `embeddings/`：专家样本及 embedding。
- `predictions.jsonl`：每例 out-of-fold 概率，不是训练集预测。
- `metrics.json`：分组、指标及跨数据集结果。
- `classifier.joblib`：使用全部 35 例拟合的实验模型，仅用于后续 shadow 试验。

模型文件只能从可信本地来源加载，joblib 反序列化不适合不可信上传文件。

## 局限与下一步

样本只有 7 人，多个扫描并非独立样本，跨数据集结果也不是大型外部验证。
当前目标是原始 T1w 采集伪影，并不覆盖 fMRIPrep 脑掩膜、配准或功能像 QC。
三张中央切片可能遗漏局部异常；DINOv2 处理 montage 时也会压缩细节。

下一步先增加被试数、保留独立测试被试，并改为多切片/三平面分别编码。
待样本充足后，在训练集内部做分组阈值校准和超参数选择，再测试固定外部集。
在此之前不接入自动阻断，也不将当前数字作为生产性能承诺。
