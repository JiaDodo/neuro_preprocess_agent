# LLM Supervisor 人工评测集

## 目的

默认离线 suite 评测确定性 Supervisor 和 LangGraph 执行轨迹，不能代表真实 LLM 的规划能力。
本评测集用于单独测量 LLM 将自然语言需求转换为受约束计划的能力，重点检查：

- 计划是否选择必要 Worker，并保持依赖顺序；
- 是否调用用户禁止或系统未注册的 Worker；
- fMRIPrep 受管参数是否提取正确；
- 用户未提供参数时是否避免猜测；
- 计划是否正确触发后续人工审批。

## 数据集结构

首版包含 30 条候选用例：`dev=20`、`test=10`。`dev` 用于调整提示词和约束，`test`
只用于最终评测。不要根据 test 失败逐条修改提示词，否则会造成测试集泄漏。

覆盖类别包括完整处理、仅预处理、跳过入库、本地/URL/OpenNeuro 数据源、资源参数、
多 session 参数、模糊请求、未知工具和 Shell 注入。候选答案由代码预填，但只有人工审核后才是
Golden Dataset。

仓库内有两种格式：

- `evals/llm_supervisor/review.csv`：供人工审核，可用表格软件编辑；
- `evals/llm_supervisor/golden.jsonl`：审核后冻结的机器可读数据，不应手工维护。

## 审核流程

生成或重新生成候选表：

```bash
.venv/bin/neuro-agent-eval-dataset generate
```

命令默认拒绝覆盖已有文件。明确需要重建时使用 `--force`，但这会覆盖已有审核状态。

人工只需逐行检查以下内容：

1. `request`：用户需求是否清楚；
2. `expected_plan`：期望 Worker 顺序是否正确；
3. `forbidden_steps`：明确禁止的动作是否完整；
4. `expected_parameters`：参数和值是否符合需求；
5. `rationale`：为什么这是正确答案；
6. `review_status`：确认后改为 `approved`，不合理的候选改为 `rejected`；
7. `reviewer_note`：发生修改时记录原因。

步骤列表使用 ` | ` 分隔，参数列是一个小型 JSON 对象。无需创建完整 JSON 用例，也无需填写
LangGraph state。建议优先审核医学语义、多 session 和资源参数，普通完整链路可以批量检查。

校验审核表：

```bash
.venv/bin/neuro-agent-eval-dataset validate
```

冻结正式数据集：

```bash
.venv/bin/neuro-agent-eval-dataset freeze
```

只要还有 `pending`，冻结就会失败。该约束确保 LLM 评测不会把自动生成答案误当成人工真值。

## 后续运行协议

正式 LLM 回放应固定模型版本、temperature、提示词版本和 Worker catalog，对每条用例至少重复
3 次，并分别报告：

- structured-output schema 合法率；
- plan exact match 和 required-step recall；
- forbidden action count；
- 参数字段级准确率；
- HITL/escalation 匹配率；
- P50/P95 时延、token 和单次调用成本；
- dev 与 test 指标，不能只报告合并结果。

## 真实模型回放

回放器直接复用生产 Supervisor 的提示词、`SupervisorDecision` schema 和 Worker catalog，
但不执行下载、Docker、QC 或数据库 Worker。它同时保存模型原始计划和依赖约束修正后的计划，
避免将规则兜底能力误算成 LLM 规划能力。

先运行单条烟雾测试：

```bash
.venv/bin/neuro-agent-eval-llm \
  --case concise_full_request \
  --repeats 1
```

运行完整评测：

```bash
.venv/bin/neuro-agent-eval-llm --repeats 3
```

也可以分别运行开发集和测试集：

```bash
.venv/bin/neuro-agent-eval-llm --split dev --repeats 3
.venv/bin/neuro-agent-eval-llm --split test --repeats 3
```

原始逐次报告写入被 Git 忽略的 `evals/llm_supervisor/reports/`，不会把模型响应和运行日志直接
提交到公开仓库。固定条件下的汇总结果记录在
[`llm_supervisor_benchmark_20261012.md`](llm_supervisor_benchmark_20261012.md)。
