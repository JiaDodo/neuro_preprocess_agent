# 视觉 QC 设计与使用

## 1. 当前边界

视觉 QC 运行在 `shadow` 模式：生成证据、模型特征和候选异常分数，但不修改规则 QC 的 `passed`、人工复核路由或数据库门禁。文件完整性、NIfTI header、shape、fMRIPrep 日志和运动阈值仍是硬约束。

DINOv2 是预训练视觉编码器，不是现成的神经影像 QC 分类器。没有人工确认的正常参考库或训练标签时，系统只保存 embedding，并返回 `needs_reference`，不会伪造 pass/fail 概率。

Qwen2.5-VL-3B-Instruct 作为第一版可解释视觉审查器。它读取独立 QC 面板并输出
`pass / review / fail`、置信度、异常类别和可见证据。模型结论固定为 shadow evidence，
`affects_gate=false`；即使模型报错或输出不符合 JSON schema，也不会改变确定性 QC 和数据库门禁。

## 2. QC packet

每名被试生成一张总览 PNG 和若干独立面板：

- T1w 与 brain mask 边界三视图；
- BOLD reference 与 mask 边界三视图；
- framewise displacement 曲线；
- fMRIPrep tissue segmentation reportlet；
- T1w 标准空间配准 reportlet；
- BOLD-T1w 配准 reportlet；
- functional ROI/coverage reportlet；
- carpet plot。

总览用于人工复核，独立面板用于模型 embedding，避免把一张很长的报告直接缩放到模型输入尺寸。

```bash
.venv/bin/python scripts/visual_qc_dataset.py packets \
  --bids-dir /path/to/raw/bids \
  --fmriprep-dir /path/to/derivatives/fmriprep \
  --output-dir data/visual_qc/manual_validation
```

提供 `--bids-dir` 后，每名被试额外生成原始 T1w 和原始 BOLD 时间均值面板，并保存维度、
体素大小、时间点和非零占比。原始面板用于人工前后对照，暂不加入现有 VLM 输入，避免在
没有重新建立基线前改变模型输入分布。

## 3. 标签

采用被试检查项级多标签：

```text
brain_mask_error
normalization_error
coregistration_error
coverage_error
ghosting_dropout
segmentation_error
severe_motion
```

总体决定为 `pass / uncertain / fail`。人工记录采用 append-only JSONL，保留标注者、说明和时间：

```bash
.venv/bin/python scripts/visual_qc_dataset.py label \
  --manifest data/visual_qc/manual_validation/manifest.jsonl \
  --sample-id MR0066073 \
  --labels-path data/visual_qc/labels.jsonl \
  --decision uncertain \
  --anatomical-input-state defaced \
  --labels severe_motion \
  --annotator dodo \
  --note "FD 存在多个明显峰值，需结合原始报告复核"
```

推荐使用本地标注界面。界面默认隐藏 VLM 结论，并对合成样本匿名化，避免异常类型从
`sample_id` 或文件名泄漏给标注者：

```bash
scripts/run_labeling_ui.sh \
  --manifest data/visual_qc/ten_subject_validation/manifest.jsonl \
  --predictions data/visual_qc/vlm_reviews/ten_subject_qwen25vl3b_v4.jsonl \
  --report runs/reports/run_20260914_160650_e449685f_summary.json
```

浏览器访问 `http://127.0.0.1:8502`。详细规则见 `docs/human_annotation.md`。

## 4. 合成异常

当前可生成 brain mask 偏移、覆盖裁剪、功能像 ghosting/dropout、功能像与 mask 偏移、异常运动曲线五类样本：

```bash
.venv/bin/python scripts/visual_qc_dataset.py synthetic \
  --manifest data/visual_qc/manual_validation/manifest.jsonl \
  --output-dir data/visual_qc/synthetic \
  --seed 42
```

合成记录强制带有：

```json
{"synthetic": true, "simulation_scope": "component_proxy"}
```

这些样本用于工程测试、预训练辅助和敏感性实验，不能替代真实失败样本，也不能作为最终临床或科研有效性证据。合成异常采用单检查项面板，避免将修改后的影像与未修改的 fMRIPrep reportlet 拼在同一个训练样本中。

## 5. DINOv2 embedding

GPU 环境建议先使用一张 3090。模型下载和运行是显式操作，流水线默认不会在计算节点上自动联网下载：

```bash
.venv/bin/python scripts/visual_qc_dataset.py embed \
  --manifest data/visual_qc/manual_validation/manifest.jsonl \
  --output data/visual_qc/embedded_manifest.jsonl \
  --model-id facebook/dinov2-small \
  --device cuda:0 \
  --allow-model-download
```

当前 `.venv` 使用 Python 3.10、PyTorch 2.4.0+cu124、TorchVision 0.19.0+cu124
和 Transformers 4.x，默认使用第一张 GPU。Transformers 固定在 5.0 以下，因为 5.x 要求
PyTorch 2.5 及以上。官方权重保存在忽略版本控制的
`data/models/facebook_dinov2-small`。本地验证命令为：

```bash
.venv/bin/python scripts/visual_qc_dataset.py embed \
  --manifest data/visual_qc/manual_validation/manifest.jsonl \
  --output data/visual_qc/manual_validation/embedded_manifest_gpu.jsonl \
  --model-id data/models/facebook_dinov2-small \
  --device cuda:0
```

当前服务器使用本地已有的 PyTorch/CUDA 二进制链接，其余依赖安装在隔离环境中。
这只是本机配置，不是公开安装方式。原服务器重建脚本保存于被忽略的本地目录：

```bash
bash local/scripts/rebuild_gpu_venv.sh
```

本机链接依赖原环境，因此不要删除被链接的环境。新机器应独立安装兼容其驱动的 CUDA
PyTorch，不需要原服务器环境。默认使用 `cuda:0`。

当前 GPU 基础环境是 Python 3.10。LangGraph API Server 依赖 Python 3.11，因此该环境支持
CLI、StateGraph、MySQL、fMRIPrep 和视觉 QC，但不安装 `langgraph dev` 所需的 server extra。
需要本地 API Server 时，应单独准备 Python 3.11 + CUDA PyTorch 环境。

人工确认至少两个正常样本后建立正常参考库：

```bash
.venv/bin/python scripts/visual_qc_dataset.py fit-reference \
  --manifest data/visual_qc/embedded_manifest.jsonl \
  --labels-path data/visual_qc/labels.jsonl \
  --output data/visual_qc/dinov2_normal_reference.npz \
  --percentile 99
```

正式实验不能只用两个样本。这里的最小数量只是防止计算退化；参考库应覆盖多个被试、站点和扫描协议，并按站点留出外部测试集。

## 6. LangGraph 集成

在配置中启用 `qc.visual` 后，现有 `qc` worker 会追加：

```text
qc_result.visual_model_qc
```

其 `affects_gate` 固定为 `false`。模型结果随运行报告和 MySQL `payload_json` 保存，被试级预测也进入 `subject_qc_results.metrics_json.visual_model_qc`。

### Qwen2.5-VL 本地审查

模型固定存放在版本控制忽略目录，下载支持续传：

```bash
HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60 \
  .venv/bin/hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir data/models/Qwen2.5-VL-3B-Instruct
```

先对已有 packet manifest 离线运行，不必重跑 fMRIPrep：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/visual_qc_dataset.py vlm-review \
  --manifest data/visual_qc/packets/<run_id>/manifest.jsonl \
  --output data/visual_qc/vlm_reviews/<run_id>.jsonl \
  --model-id data/models/Qwen2.5-VL-3B-Instruct \
  --device cuda:0
```

完整 LangGraph 集成配置见 `configs/visual_qc_vlm_shadow.json`。每个被试最多输入 8 个独立面板，
默认视觉 token 像素范围为 200704 至 401408，输出最多 384 token。模型只允许七类预处理
异常；无法解析和字段越界输出按被试标记为不可用，重复类别保留最严重的一条观察。

第一轮验证应比较 VLM 与确定性指标及人工复核，而不是直接计算模型“准确率”。十被试案例中
`sub-04` 是规则告警、待复核案例，不能在没有人工标签时直接视为视觉异常阳性。
首轮 3B 零样本结果及失败分析见 `docs/vlm_baseline_20260914.md`。

对带标签的 manifest 运行统一评分：

```bash
.venv/bin/python scripts/visual_qc_dataset.py score-vlm \
  --manifest data/visual_qc/synthetic_validation/synthetic_manifest.jsonl \
  --predictions data/visual_qc/vlm_reviews/synthetic_qwen25vl3b_v2.jsonl \
  --output data/visual_qc/vlm_reviews/synthetic_qwen25vl3b_v2_metrics.json
```

## 7. ABIDE/PCP 的用途

本地 `data/abide_pcp` 包含 1035 名被试的 CPAC `func_mean`、`func_mask` 和 `func_preproc`，CSV 还包含 QAP 指标与人工评分。它适合做辅助训练、弱监督和跨站点评估，但不是 fMRIPrep 输出，当前也缺少对应结构像，因此不能直接替代本项目的 fMRIPrep 人工标注集。
