# Neuro Preprocess Agent 项目总结

更新时间：2026-09-14  
版本：0.7.0

## 项目定位

这是一个基于 LangGraph 的神经影像数据处理 Agent 系统。它把自然语言需求转换为一条受控的数据流水线，并通过 Supervisor 制定计划、Human-in-the-loop 审批、Worker 执行和 QC 门禁完成闭环。

项目不是让多个 LLM 自由聊天。这里的“Agent”主要体现在：根据用户意图规划任务、按环境状态调度工具、遇到异常调整路径、保留会话状态和人工决策。下载、转换、fMRIPrep 和数据库写入仍由确定性代码执行。

## 当前闭环

```text
自然语言请求
-> 配置加载与意图提取
-> Supervisor 计划
-> 人工确认或自然语言修改
-> 数据权限、磁盘、Docker、工具、资源和数据库运行前检查
-> 本地/URL/OpenNeuro 数据获取
-> BIDS/DICOM/NIfTI 判断与准备
-> 输入 T1w 完整性检查与逐被试去颅骨策略复核
-> Docker fMRIPrep 并行预处理
-> 文件、日志、NIfTI header、confounds 和视觉产物 QC
-> 自动 QC 通过后写 JSONL/MySQL；软告警经人工审计后决定入库
-> 结构化报告
```

真实本地 BIDS 数据 `sub-local-demo` 已经成功跑通过 fMRIPrep。另用 14,986 个真实 DICOM 完成 T1w/BOLD/DWI 转换和 PyBIDS 预检，并选择性下载 OpenNeuro `ds000001/sub-01` 至 `sub-10` 完成公开数据源与十被试稳定性验证。

## 主要技术点

| 能力 | 实现 |
| --- | --- |
| 编排 | LangGraph `StateGraph` |
| 计划 | DeepSeek 结构化输出，失败时规则回退 |
| 人工审批 | `interrupt()` + `Command(resume=...)` |
| 状态恢复 | CLI SQLite / 平台托管 checkpointer + thread id |
| 工具治理 | Worker registry + risk + allow/ask/deny |
| 配置 | Pydantic v2 严格校验，未知字段直接拒绝 |
| 数据获取 | local path、URL、OpenNeuro |
| 格式统一 | PyBIDS 预检、DICOM header 校验、dcm2bids participant/session jobs、NIfTI manifest |
| 预处理 | Docker fMRIPrep，LangGraph `Send` 分批并行、锁、隔离和聚合 |
| QC | fMRIPrep 前 T1w 输入筛查；输出端模态感知完整性、header、shape、运动阈值、mask/tSNR/Dice 定量指标、三态门禁、视觉 QC shadow 基础设施 |
| 业务存储 | 原子/并发安全 JSONL、MySQL 8.4 运行级与被试级事务 upsert |
| 可观察性 | 结构化 events、subject 日志、summary JSON |
| 交互 | 自然语言 CLI、doctor、tools、mysql-init |

## 本次架构升级

原型阶段的 `graph.py` 有约 980 行，状态、LLM prompt、文件复制、Docker、QC 和报告混在一起。当前调整为：

- `graph.py`：只保留图、节点适配、路由和 HITL。
- `agents/supervisor.py`：计划生成与安全校验。
- `tools/`：领域 worker 和注册表。
- `runtime.py`：供 CLI 和未来 API 共用的运行接口。
- `db.py`：JSONL 与真实 MySQL 实现。
- `config.py`：配置模型、路径和可执行文件发现。

架构参考 OpenCode、Hermes Agent 和 OpenClaw 的共同模式：入口与核心分离、中央工具注册、按 Agent/工具限制权限、SQLite 会话持久化、统一错误结果以及执行过程可观察。没有复制这些项目与神经影像无关的通用编码工具和消息平台功能。

## 安全约束

- LLM 不能直接执行 shell，也不能自由生成 Docker 参数。
- `permissions.<worker>=deny` 会在 dispatch 层拒绝执行。
- `ask` worker 必须在已批准计划中运行。
- QC 失败强制跳过数据库。
- 预检硬失败直接报告；资源 warning 必须再次人工确认。
- 运动阈值超限进入 `review_required` 和第二个 HITL；人工批准/拒绝原因写入状态、报告和数据库 payload。
- 输入 T1w 在 fMRIPrep fan-out 前检查；疑似预去颅骨和多 session 混杂状态必须人工确认逐被试 `skip/force/auto`，最终命令进入配置指纹。
- 本地 staging 按 `run_id` 隔离，不再先删除已有目录。
- fMRIPrep work/log 按 `run_id` 隔离，同一被试由文件锁防止并发覆盖。
- fMRIPrep 容器映射宿主用户 UID/GID，避免 derivatives 和 work 目录产生 root 所有文件。
- 配置指纹匹配的已完成 subject 在跨运行恢复/重试时标记为 `reused`，不重复计算。
- DICOM/NIfTI 转换目录按 `run_id` 隔离。
- 报告和 JSONL 原子写入，JSONL 使用跨进程锁保护并行更新。
- thread id 只能对应一个任务，拒绝覆盖已有 checkpoint。
- CLI JSON 输出会隐藏配置中的 API key 和数据库密码。
- worker 失败不会丢失上下文，而是生成 `status=failed` 的报告。

## 当前不包含

- 经多中心外部数据验证或微调的 VLM 分类器；当前 Qwen2.5-VL 只以 shadow 模式生成结构化辅助审查证据。
- 自动推断所有 BIDS sidecar 扫描参数；功能 NIfTI 仍强制要求可信 `RepetitionTime`，多被试/多 session 使用显式 manifest。
- 面向最终用户的任务管理 Web UI、定时任务和消息平台入口；当前仅提供独立的视觉 QC 人工标注界面。
- 多个 LLM Agent 自由协作；当前是一个 LLM Supervisor 加多个受控 worker。

## 验证状态

自动测试覆盖：

- mock 完整闭环。
- 初次审批和修改后的二次审批。
- 自然语言审批解析。
- 非法 HITL 输入重新询问。
- 三 subject `Send` fan-out 与 reducer 聚合。
- QC 失败阻断数据库。
- JSONL 按 `run_id` 幂等更新。
- worker 权限拒绝。
- worker 异常生成失败报告。
- 平台导出图不嵌入本地 checkpointer。
- 本地 dry-run 使用真实源目录并执行 PyBIDS 预检。
- BOLD 缺少 `RepetitionTime` 时预检失败。
- T1-only 数据不错误要求功能像产物。
- 已完成 subject 可在不启动命令的情况下安全复用。
- 高运动结果进入人工复核，并通过第二个 LangGraph interrupt 记录带原因的决定。
- 影像与 mask shape 不一致触发硬性 QC 失败。
- 20 个并发 JSONL 写入不丢记录。
- thread id 误复用和错误恢复被拒绝。
- NIfTI 转换输出按 run id 隔离。
- 配置拼写错误不会被静默忽略。

共 56 项自动测试通过；`langgraph dev` 已验证可以加载 `pipeline` 图。

此外已增加产业化评测骨架：以严格 JSON 用例驱动完整 LangGraph `interrupt/resume`，在 Worker Registry 边界进行可恢复的故障注入，并生成 JSON/Markdown 评测报告。离线 smoke suite 覆盖正常闭环、QC/输入安全拦截、人工修改与拒绝、瞬时故障重试；质量门禁单独统计异常任务误放行到数据库的 `false_pass_count`。详见 `docs/evaluation.md`。

现有真实数据 `sub-local-demo` 已用新版规则重新验证：关键输出和 4 个 NIfTI header 通过，confounds 共 240 行，12 个视觉报告产物存在，fMRIPrep 成功日志通过。该结果只代表结构与产物完整，不代表已经完成视觉异常诊断。

新增基础设施与数据验证：

- MySQL 8.4 容器已启动，`preprocessing_runs` 和 `subject_qc_results` 完成真实建表、连接、事务写入和 SQL 回查。
- MySQL 已验证首次写入、同一 run 幂等更新、被试级事务记录、QC 拦截和不可连接检测；重试耗尽后仍生成失败报告。
- 真实 DICOM 输出通过 fMRIPrep 容器内置 `bids-validator@1.14.10`，仅保留 SliceTiming、Authors 和 README 长度警告，无 error。
- OpenNeuro worker 保留 `include` 策略，目前缓存 `ds000001/sub-01` 至 `sub-10` 共 85 个文件约 1.5 GB，十名被试均通过 BIDS 预检。

运行命令：

```bash
cd /path/to/neuro_preprocess_agent
.venv/bin/python -m unittest discover -s tests -v
```

视觉 QC 第一阶段已经能够从真实 `sub-local-demo` 输出生成结构像、功能像、运动曲线及五类 fMRIPrep reportlet 总览，并生成五类带来源标记的合成异常。下一阶段需要积累人工确认的正常/异常标签并建立跨站点评估，再训练分类头；模型仍作为现有完整性、数值规则和人工审计之外的辅助证据。

三被试稳定性验证已使用 OpenNeuro `ds000001` 的 `sub-01`、`sub-02`、`sub-03`
跑通有界双 worker fMRIPrep：3/3 被试成功，每人 99 个正式输出；规则 QC、三份
视觉 packet、DINOv2 GPU embedding、MySQL 运行级和被试级写入全部完成。详细记录见
`docs/stability_validation_20260902.md`。

十被试验证进一步覆盖 `sub-01` 至 `sub-10`：10/10 完成，每人 99 个输出，3 人跨 run
复用、7 人真实新计算，新计算耗时 P50/P95 为 2022.775/2043.106 秒。定量 QC 检出
`sub-04` 功能 mask 体积和覆盖异常、标准空间结构/功能 mask Dice 仅 0.693，经人工复核
拒绝后整批未入库。该验证同时暴露并修复了 BIDS 被试过滤、历史结果复用、复用日志 QC
和 Linux 可用内存估算问题。详见 `docs/stability_validation_10subjects_20260914.md`。

跨数据集 pilot 新增 4 个 OpenNeuro 数据集，覆盖双 session、双 run、儿童数据和旧版
BIDS 元数据。2 个符合现行 schema 的案例均完成 fMRIPrep、QC 和 MySQL 入库；2 个旧版
元数据案例在 fMRIPrep 前被 BIDS Validator 拦截，未发生失败任务误入库。验证期间修复了
OpenNeuro 缓存幂等、preprocess 失败路由和多 session 输出发现问题。详见
`docs/cross_dataset_validation_20260916.md`。

有界并发评测复用生产 `AgentRuntime`、LangGraph `Send` 路由和 reducer，仅将耗时的单被试
预处理替换为固定时长任务。12 个任务、5 次重复条件下，4 worker 相对 1 worker 的中位
加速比为 `3.80x`，并行效率为 `94.9%`，实测峰值严格等于配置上限。已有七被试真实
fMRIPrep 计算中峰值并发为 4、整体吞吐为 `4.16 subjects/hour`，其中饱和四任务批次为
`7.05 subjects/hour`。模拟调度指标与真实历史观测分别报告，未将前者外推为 fMRIPrep
性能。详见 `docs/concurrency_benchmark_20260916.md`。

视觉模型层已增加 Qwen2.5-VL-3B-Instruct 本地推理后端。它复用结构像、功能像、运动、
分割、标准化、配准、覆盖和 carpet plot 面板，以严格 JSON 输出被试级
`pass / review / fail`、异常类别与证据；异常输出按被试隔离，且 `affects_gate` 固定为 false。

人工数据闭环已增加 Streamlit 盲标界面。标注者可以浏览 packet、定量指标和历史记录，
提交 `pass / uncertain / fail`、多标签、严重程度、排除建议和证据备注。VLM 建议默认隐藏，
合成样本的标识和文件名不会暴露异常类别；每次修改追加到 JSONL 账本，不覆盖审计历史。

新增 fMRIPrep 前置 Input QC：在十被试真实输入上，规则只标记人工确认的
`sub-04` 为预去颅骨候选（非零体素占比 `0.137212`），其余九个面部脱敏样本为
`0.793230-0.925644`。候选会在 subject fan-out 前中断，人工决定按被试改写
`--skull-strip-t1w`，避免将面部脱敏误当成预去颅骨，也避免错误复用旧配置产物。
`sub-04` 使用人工确认的 `skip` 完成独立回归后，原生 T1 mask 从 343,400 增至
811,601 体素，结构/功能 mask Dice 从 `0.693498` 提升到 `0.921624`，输出 QC
通过并成功写入 MySQL。详见 `docs/sub04_skullstrip_regression_20260915.md`。
