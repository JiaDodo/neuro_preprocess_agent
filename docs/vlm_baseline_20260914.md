# Qwen2.5-VL-3B 零样本 QC 基线（2026-09-14）

## 目的

验证本地 VLM 是否能够稳定读取项目生成的 fMRIPrep QC panels、返回可解析的结构化结果，
并建立后续 prompt 调整、模型对比和微调的可复现基线。该实验不用于证明临床或科研有效性。

## 环境与模型

- 模型：`Qwen/Qwen2.5-VL-3B-Instruct`
- 设备：单张 NVIDIA RTX 3090，推理显存约 7.8 GiB
- 本地目录：`data/models/Qwen2.5-VL-3B-Instruct`
- 输入：每名真实被试最多 8 张独立 QC panel
- 输出：Pydantic 校验的 observations、代码派生 findings 和 shadow decision
- 门禁：`affects_gate=false`

两个权重分片 SHA-256：

```text
41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8
365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4
```

## 真实数据观察

在 OpenNeuro `ds000001` 的 10 名被试上，模型能够完成 GPU 推理和 JSON 输出，但判断高度
模板化，反复报告 `brain_mask_error` 和 `severe_motion`。加入真实 framewise displacement 数值后，
模型仍会把正常范围运动解释为异常。真实面板没有逐项人工金标准，因此这里只能确认工程可运行，
不能计算准确率。

此前定量规则触发的 `sub-04` 中心切片 mask 和 coregistration reportlet 肉眼未见明确失败，
所以它应保留为“规则告警、需人工复核”，不能直接当作 VLM 阳性标签。

## 合成异常结果

数据为一个真实被试派生的五类 component proxy：brain mask 偏移、覆盖裁剪、
ghosting/dropout、coregistration 偏移和严重运动曲线。

| 指标 | 数值 |
|---|---:|
| 推理覆盖率 | 1.000 |
| Micro precision | 0.222 |
| Micro recall | 0.400 |
| Micro F1 | 0.286 |
| Exact match | 0.000 |
| TP / FP / FN | 2 / 7 / 3 |

结果文件：

```text
data/visual_qc/vlm_reviews/synthetic_qwen25vl3b_v2.jsonl
data/visual_qc/vlm_reviews/synthetic_qwen25vl3b_v2_metrics.json
```

## 结论

当前实现已经完成“模型下载、单卡加载、多图推理、结构化校验、异常隔离、离线评分和
LangGraph QC worker 集成”的工程闭环，但 3B 通用模型的零样本分类质量不足。它只能作为
shadow evidence，不能替代规则 QC 或人工复核，更不能决定 MySQL 写入。

完整图验证使用 thread `vlm-shadow-integration-20260914`。10 名被试均完成 VLM 推理，规则 QC
随后进入 LangGraph 人工复核；人工以“零样本误报较多”为由拒绝，最终状态为 `qc_rejected`，
未执行数据库写入。审计报告为：

```text
runs/reports/run_20260914_160650_e449685f_summary.json
```

下一阶段应先积累真实人工标签并固定 train/validation/test 的被试级、站点级划分，再比较：

1. 更明确的单任务 panel prompt；
2. Qwen2.5-VL-7B 零样本基线；
3. DINOv2/医学影像 encoder 加轻量分类头；
4. Qwen2.5-VL LoRA 微调。

模型升级的最低门槛应同时约束覆盖率、每类 recall、false-positive rate 和跨站点表现，
不能只看总体 accuracy。
