# sub-04 预去颅骨输入回归验证

验证日期：2026-09-15  
数据集：OpenNeuro `ds000001/sub-04`  
修复 run ID：`run_20260914_222541_33ced198`

## 问题

十被试稳定性验证中，`sub-04` 的原始 T1w 已经去颅骨，但旧任务仍按常规输入执行脑提取。
旧输出的原生空间结构脑掩膜明显偏小，标准空间结构/功能 mask Dice 仅为 `0.693498`，
因此被规则 QC 标记为需要人工复核并被人工拒绝入库。

## 修复路径

新版主图在 BIDS 准备和 fMRIPrep subject fan-out 之间增加 Input QC：

```text
BIDS prepare -> input_qc -> input_qc_review -> fMRIPrep subject worker
```

Input QC 检测到原始 T1w 非零体素占比为 `0.137212`，低于候选阈值 `0.3`，将其标记为
`suspected_pre_skull_stripped`。人工回复 `sub-04使用skip` 后，最终 Docker 命令使用
`--skull-strip-t1w skip`。该决定、原始指标和最终参数均写入报告，最终参数同时进入任务配置指纹。

## 结果

fMRIPrep 成功完成，共生成 107 个文件。输出 QC 硬检查、运动检查和当前定量阈值均通过，
MySQL 在首次尝试完成写入。

| 指标 | 旧输出 | `skip` 修复输出 | 变化 |
| --- | ---: | ---: | ---: |
| 原生 T1 mask 体素 | 343,400 | 811,601 | +136.34% |
| 原生 T1 mask 体积 | 610,488.60 mm3 | 1,442,845.53 mm3 | +136.34% |
| 标准空间 T1 mask 体素 | 1,041,260 | 1,117,519 | +7.32% |
| 标准空间 BOLD mask 体素 | 90,862 | 47,471 | -47.75% |
| BOLD median tSNR | 76.625 | 75.784 | -1.10% |
| 结构/功能 mask Dice | 0.693498 | 0.921624 | +0.228126 |

Dice 提升至与该批次其他被试接近的范围，异常扩张的 BOLD mask 缩小，而 tSNR 基本稳定。
新版 QC packet 中，原生结构脑边界、标准化和结构/功能配准也比旧输出合理。

## 证据位置

- 运行报告：`runs/reports/run_20260914_222541_33ced198_summary.json`
- fMRIPrep 日志：`runs/logs/fmriprep_ds000001_sub04_skip_regression/run_20260914_222541_33ced198/sub-04.log`
- 修复输出：`data/derivatives/fmriprep_ds000001_sub04_skip_regression/`
- 新 QC packet：`data/visual_qc/packets/run_20260914_222541_33ced198/sub-04_qc_packet.png`
- 新 packet manifest：`data/visual_qc/packets/run_20260914_222541_33ced198/manifest.jsonl`

## 结论与限制

本次回归证明 Input QC 的逐被试参数决策可以修复一个真实的预处理失败模式，并阻止错误参数结果
被配置指纹复用。新版 packet 已由标注者 `dodo` 独立判定为 `pass`，输入状态确认为
`pre_skull_stripped`，所有输出异常标签为 false。旧失败输出与新修复输出现已登记到
`evals/manifests/visual_qc_golden_pairs.json`，可用 `scripts/validate_visual_regressions.py`
自动检查文件、人工结论、指标阈值和修复幅度。
