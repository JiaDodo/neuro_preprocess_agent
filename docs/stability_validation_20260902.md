# 三被试端到端稳定性验证

验证日期：2026-09-02  
数据集：OpenNeuro `ds000001`  
被试：`sub-01`、`sub-02`、`sub-03`  
运行 ID：`run_20260902_154016_100c72ba`  
LangGraph thread ID：`stability-ds000001-3sub-v1`

## 验证范围

本次使用真实公开 BIDS 数据验证以下链路：

```text
OpenNeuro 增量下载
-> BIDS 预检
-> LangGraph Send 分批调度
-> Docker fMRIPrep
-> 规则 QC
-> 视觉 QC packet 与 DINOv2 embedding
-> MySQL 入库
-> JSON 运行报告
```

配置文件为 `configs/openneuro_ds000001_stability_3subjects.json`。fMRIPrep 最多并行
2 名被试，每名被试配置 16 个进程、8 个 OMP 线程和 32 GB 内存，输出空间为
`MNI152NLin2009cAsym`，不运行 FreeSurfer recon-all。

## 运行结果

OpenNeuro 共检查 29 个目标文件，总下载量约 451.5 MB。第一次下载因临时 DNS/网络错误
缺少 `sub-02_T1w.nii.gz`；LangGraph fetch 节点自动重试后只补下载缺失文件，没有重复下载
完整数据集。

BIDS 预检通过。三名被试均包含：

- 1 个 T1w；
- 3 个 BOLD run；
- 无 BIDS error 或 warning。

| 被试 | fMRIPrep 状态 | 耗时 | 正式输出文件 | HTML 报告 |
| --- | --- | ---: | ---: | --- |
| sub-01 | completed | 1920.372 秒 | 99 | 存在 |
| sub-02 | completed | 1921.730 秒 | 99 | 存在 |
| sub-03 | completed | 1902.442 秒 | 99 | 存在 |

`sub-01` 和 `sub-02` 同批并行，完成后 `sub-03` 自动进入下一批。三份日志都包含
`fMRIPrep finished successfully!`，没有记录 pipeline error。

正式 derivatives 约 1.9 GB，work 目录约 5.5 GB。每名被试均生成 8 个视觉模型输入面板、
一张 QC 总览和一个有限值、L2 归一化的 384 维 DINOv2 embedding。

## QC 结果

| 被试 | mean FD | max FD | FD > 0.5 比例 | mean std DVARS | 规则结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| sub-01 | 0.0768 mm | 0.2165 mm | 0 | 1.1358 | pass |
| sub-02 | 0.0973 mm | 0.4074 mm | 0 | 1.1155 | pass |
| sub-03 | 0.0965 mm | 0.4648 mm | 0 | 0.9685 | pass |

每名被试均通过文件完整性、NIfTI header、结构/功能 mask shape、视觉产物数量、
confounds、运动阈值和成功日志检查。人工抽查 QC packet 未发现明显脑掩膜、标准化、
BOLD-T1 配准或覆盖错误。DINOv2 当前状态仍为 `needs_reference`，不影响 QC 门禁。

## 数据库与状态

LangGraph 最终状态为 `end`，无错误事件。MySQL `preprocessing_runs` 中存在一条
`qc_passed` 运行记录，`subject_qc_results` 中存在 3 条被试记录，三者的
`qc_passed`、`hard_checks_passed` 和 `approved_for_database` 均为 true。

最终报告：`runs/reports/run_20260902_154016_100c72ba_summary.json`。

## 暴露的问题

本次使用的旧 Docker 命令以 root 身份运行，因此本次 derivatives 和 work 文件为
`root:root`。测试后已修改命令生成器，为后续容器增加宿主 UID/GID 映射。镜像内
TemplateFlow 路径可由该 UID 读写。修复后使用真实 `ds000001/sub-01` 和 FreeSurfer
license 运行 fMRIPrep `--boilerplate-only`：BIDS 校验通过、632 节点工作流构建成功，
生成的 citation、配置和 work 文件均为 `dodo:dodo`。新增命令回归测试，但尚未再执行
一次完整的长时间影像计算。本次历史 derivatives 和 work 目录也已在不修改内容的前提下
统一调整为 `dodo:dodo`。

本次验证证明三被试有界并行链路可以完成，但尚不能代表大规模生产稳定性。下一轮应覆盖：

- 至少 10 名被试；
- 多 session BIDS 数据；
- 人为制造单被试失败，验证部分失败隔离；
- 中途终止后从 checkpoint 恢复；
- 修复后 UID/GID 下的完整 fMRIPrep 运行。
