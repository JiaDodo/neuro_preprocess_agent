# 十被试端到端稳定性与 QC 门禁验证

验证日期：2026-09-14  
数据集：OpenNeuro `ds000001`  
被试：`sub-01` 至 `sub-10`  
首次执行 run ID：`run_20260914_102449_fde23fda`  
QC 复跑 run ID：`run_20260914_120728_a6b0f3cd`

## 验证范围

```text
自然语言任务与计划确认
-> preflight
-> OpenNeuro 增量下载
-> 精确被试选择与 BIDS 预检
-> 4 worker fMRIPrep fan-out
-> 跨 run 完成标记复用
-> 文件/日志/运动/定量影像 QC
-> 人工 QC interrupt
-> MySQL 门禁
-> 结构化报告
```

运行配置为 `configs/openneuro_ds000001_stability_10subjects.json`。OpenNeuro 缓存包含 85 个文件约 1.5 GB，十名被试均包含 1 个 T1w 和 3 个 BOLD run，BIDS 预检无 error/warning。预检确认 96 个 CPU、约 455 GB `MemAvailable`、约 3.1 TB 空闲磁盘、MySQL 连接和 fMRIPrep 镜像均可用；实际镜像 ID 为 `sha256:b205d9b33c21ebeb77b44199c2845a563618d10bb6a64fb5d67b9d73808dff5a`。

## 预处理结果

`sub-01` 至 `sub-03` 通过历史成功日志、HTML、NIfTI 产物和配置指纹完成接管，补写完成标记并返回 `reused`。`sub-04` 至 `sub-10` 本次真实执行。

| 指标 | 结果 |
|---|---:|
| 成功被试 | 10/10 |
| 复用被试 | 3 |
| 新计算被试 | 7 |
| 每被试正式输出 | 99 个文件 |
| 新计算耗时 P50 | 2022.775 秒 |
| 新计算耗时 P95 | 2043.106 秒 |
| root 所有权输出 | 0 |

新计算被试耗时为 1987.103 至 2043.106 秒。首次批次中的复用任务占用了 fan-out 槽位，导致 `sub-04` 单独形成首批，之后为 `sub-05` 至 `sub-08` 四并发、`sub-09` 至 `sub-10` 两并发。这不影响正确性，但后续可在 fan-out 前预判可复用任务以提高调度效率。

## 定量 QC

所有被试均通过文件存在、NIfTI header、shape、fMRIPrep 成功证据和非有限值硬检查。运动阈值全部通过。

| 被试 | mean FD (mm) | FD > 0.5 比例 | median tSNR | 标准空间 mask Dice |
|---|---:|---:|---:|---:|
| 01 | 0.0768 | 0 | 70.24 | 0.941 |
| 02 | 0.0973 | 0 | 61.65 | 0.946 |
| 03 | 0.0965 | 0 | 52.48 | 0.926 |
| 04 | 0.0980 | 0 | 76.63 | **0.693** |
| 05 | 0.1333 | 0.0167 | 59.52 | 0.935 |
| 06 | 0.0648 | 0 | 79.14 | 0.945 |
| 07 | 0.0711 | 0 | 63.01 | 0.939 |
| 08 | 0.2001 | 0.0033 | 69.79 | 0.938 |
| 09 | 0.1627 | 0.0201 | 53.58 | 0.916 |
| 10 | 0.0779 | 0 | 76.18 | 0.952 |

`sub-04` 的标准空间功能 mask 约 3.55 L、占图像 68.2%，而其余九人约 1.70 至 1.89 L、占 32.6% 至 36.3%；其结构/功能 mask Dice 为 0.693，其余为 0.916 至 0.952。该被试运动正常，因此异常更可能来自 mask、覆盖或配准，而不是头动。

系统将其判为 `review_required` 并触发 LangGraph QC interrupt。人工复核记录选择 `reject`，最终状态为 `qc_rejected`，`db` 加入 `skipped_steps`，MySQL 没有收到该批次记录。该结果说明“10/10计算完成”不等于“10/10质量合格”，QC 门禁能够阻止明显异常混入业务数据库。

## 本轮发现并修复的问题

1. BIDS 输入最初忽略 `preprocess.subjects`，可能把目录内全部被试加入任务；现已严格选择并对不存在的被试报错。
2. 跨 run 复用最初只检查新日志路径，导致已完成被试被重复计算；现改为配置指纹完成标记，并支持一次性接管旧成功日志。
3. QC 最初把 `reused` 被试缺少本次日志误判为硬失败；现接受有效完成标记作为成功证据。
4. Linux 内存预检最初读取空闲页而非 `MemAvailable`，会忽略可回收缓存；现已修正。
5. DICOM 预检最初只从 `PATH` 查找 dcm2niix；现也检查 dcm2bids 同目录工具。

## 其他验证

- 真实 MySQL：首次写入 `written`，同一 run 再写 `updated`，运行表保持 1 行、被试表保持预期行数；QC 失败新增 0 行。
- MySQL 不可连接：受控重试耗尽后写入结构化错误并生成 `failed` 报告。
- Golden 输入：OpenNeuro 85 文件和 DICOM 14,988 文件的聚合 SHA-256 指纹已经创建并复核。
- 真实 Golden suite：4 个 BIDS 案例共 14/14 断言通过；真实 DICOM 转 BIDS 案例单独通过，处理 14,986 份 DICOM。
- 离线系统评测：8/8 场景、32/32 断言通过，`false_pass_count=0`。

稳定性机器可读汇总为 `runs/reports/run_20260914_120728_a6b0f3cd_stability.json`，人工可读表为同名 Markdown。QC 审计报告为 `runs/reports/run_20260914_120728_a6b0f3cd_summary.json`。

## 尚未覆盖

- 独立站点和多 session 的真实 fMRIPrep 数据；
- 单被试容器崩溃后的同批失败隔离；
- 进程被强制终止后的跨进程 checkpoint 恢复演练；
- 定量 QC 阈值的跨数据集校准；
- VLM shadow 评测、API/任务队列和多人权限。

## 后续修复验证

`sub-04` 后续被确认属于预去颅骨输入。新增 Input QC 后，人工确认
`--skull-strip-t1w skip` 的独立回归成功完成，标准空间结构/功能 mask Dice 从
`0.693498` 提升至 `0.921624`，异常 BOLD mask 体积也恢复到同批次合理范围。
完整证据见 [sub-04 预去颅骨输入回归验证](sub04_skullstrip_regression_20260915.md)。
