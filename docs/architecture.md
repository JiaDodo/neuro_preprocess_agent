# 系统架构

## 1. 设计边界

本项目是“受控数据流水线 + Agent 调度”，不是多个 LLM 自由聊天。神经影像预处理具有高计算成本、长运行时间和严格数据依赖，因此采用以下分工：

- Supervisor：理解需求、生成执行计划、安排下一步、处理异常分支。
- LangGraph：保存状态、执行节点顺序、暂停和恢复。
- Worker：执行运行前检查，并调用 OpenNeuro、dcm2bids、Docker fMRIPrep、QC 和数据库。
- Policy：决定 worker 是 `allow`、`ask` 还是 `deny`。
- Storage：SQLite 保存 graph checkpoint，JSONL/MySQL 保存业务运行记录。

LLM 输出只是候选计划。程序会重新验证顺序、去重并执行硬约束，因此模型无法让 `db` 绕过 `qc`。

## 2. 分层

```text
CLI / future API
       |
       v
AgentRuntime                 会话、thread_id、invoke、resume
       |
       v
LangGraph                    state、interrupt、routing、checkpoint
       |
       +--> Supervisor       LLM 首次计划，规则兜底，安全校验
       |
       +--> WorkerRegistry   catalog、risk、permission、dispatch
                 |
                 +--> preflight --> source/fetch
                 +--> preprocess prepare
                        +--> input QC / HITL
                         +--> Send(subject workers)
                         +--> finalize
                 +--> qc
                 +--> db/report
```

`graph.py` 仍然可以一眼看到完整关系，但具体命令和文件处理不再堆在图定义里。

## 3. Supervisor

LLM 只在首次制定计划时调用。正常节点之间按已批准计划确定性推进；worker 报错或 QC 失败时走代码级安全分支。这样减少重复 token 消耗，也避免模型在长任务中不断改变计划。

结构化输出为：

```json
{
  "execution_plan": ["preflight", "source", "fetch", "preprocess", "qc", "db", "report"],
  "next_node": "preflight",
  "skip_steps": [],
  "reason": "按用户需求执行完整预处理"
}
```

硬约束包括：

- `preflight/source/fetch` 在 `preprocess` 前。
- 存在 `db` 时必须先有 `qc`。
- `report` 永远在最后。
- QC 硬失败时把计划中的 `db` 加入 `skipped_steps`；软阈值告警进入 `qc_review`，只有带审计说明的人工批准才能继续入库。
- worker 报错时直接生成失败报告。
- LLM 不可用时按 `fallback_to_rule` 回退到确定性计划。

## 4. Human-in-the-loop

具有网络、计算或数据库副作用的 worker 默认权限为 `ask`。首次计划集中展示这些操作，一次授权覆盖该计划；修改来源、并行数或计划后，`plan_approved` 会重新变为 `false`，必须再次确认。预检硬错误直接终止，资源预算 warning 触发额外 `interrupt()`；fMRIPrep 前发现疑似预去颅骨或同一被试 T1w 状态混杂时，暂停并允许按被试调整 `skip/force/auto`；自动输出 QC 出现软阈值告警时，再次要求查看 HTML/指标并留下批准或拒绝原因。硬错误没有人工绕过路径。

这比让用户输入 JSON 更适合 CLI，也比每个 subject 启动 Docker 前重复询问更实用。

## 5. 错误和恢复

确定性的数据/配置错误会被包装为结构化错误：

```json
{
  "node": "preprocess",
  "type": "RuntimeError",
  "message": "docker not found in PATH",
  "retryable": false
}
```

错误进入 LangGraph state，Supervisor 跳到 `report`，因此失败任务也会留下可审计报告。网络和 LLM 瞬时错误使用节点级 `RetryPolicy`；数据库节点按受控次数指数退避，重试耗尽后把错误写入状态并进入失败报告，而不是让整张图停在未解释的异常上。

fMRIPrep 使用 `Send` 为每个 subject 创建独立 worker，结果通过 reducer 汇总。调度器每次最多发送 `max_workers` 个任务，当前批次形成 barrier 后再发送下一批，避免配置只影响显示而不限制真实并发。已完成 subject 会形成 LangGraph checkpoint，恢复粒度不再是整个预处理批次。

运行期文件也按职责隔离：staging、fMRIPrep work 和日志都包含 `run_id`；subject lock 则跨运行共享，用于阻止两个任务同时改写同一被试的 derivatives。配置指纹匹配的完成标记、HTML report 和 NIfTI 产物共同证明被试已经成功，worker 返回 `reused` 而不重复启动 Docker。升级前的历史成功日志可被一次性接管并补写完成标记；后续改变镜像或受管参数时不会错误复用旧配置结果。

CLI 的 HITL 状态写入 SQLite checkpoint，可用原 `thread_id` 跨进程恢复。新任务必须使用新 `thread_id`；运行时会拒绝覆盖已有线程或恢复已结束线程。`langgraph.json` 导出的平台图不携带自定义 checkpointer，由 LangGraph API 管理持久化。

## 6. QC 门禁

QC 门禁仍由确定性规则控制：

- PyBIDS 索引、被试/模态识别、空文件和 BOLD `RepetitionTime` 预检。
- fMRIPrep 前检查 T1w 可读性、三维 shape、voxel size、非有限值和非零体素占比；候选异常通过独立 HITL 决定逐被试去颅骨参数。
- fMRIPrep 返回码和成功日志。
- 根据输入模态检查 T1w/BOLD、mask、confounds 和 HTML report 是否存在且非空。
- NIfTI header、维度、shape、datatype 和 bitpix 是否可解析。
- 预处理影像与 brain mask 的前三维 shape 是否一致。
- confounds 行数、平均 FD、FD 超阈值比例和 standardized DVARS。
- fMRIPrep figures 和 HTML 视觉产物是否齐全。

QC 有三个自动结果：硬性检查和运动规则均通过为 `completed`；结构完整但运动阈值超限为 `review_required`；文件、日志、header 或 shape 失败为 `failed`。`review_required` 暂停图并等待审计式人工决定，批准后设置 `approved_for_database=true`，拒绝则直接报告；硬失败始终阻断数据库。阈值是可配置工程默认值，不应被表述为统一科研标准。

定量 QC 使用 NIfTI 数据计算非有限值比例、零值比例、强度分位数、mask 体积和边界接触比例、BOLD tSNR；当结构和功能 mask 位于同一标准空间时，将功能 mask 重采样到结构网格并计算 Dice。明确的数据损坏作为硬错误，尚需跨数据集校准的 tSNR、边界和 Dice 阈值作为软告警。

视觉 QC shadow 层会生成标准化 packet，并可调用 DINOv2 提取面板 embedding。它的结果写入 `qc_result.visual_model_qc`，`affects_gate=false`，因此不会替换文件完整性、数值规则或人工复核。

视觉报告“存在”不等于影像“视觉合格”。当前版本尚无经过真实异常标签验证的分类头；配准偏移、鬼影、覆盖不足等结论仍由人工复核。DINOv2 在正常参考库建立前只生成 embedding，不输出伪造的异常概率。

## 7. 数据持久化

- `data/checkpoints/checkpoints.sqlite`：LangGraph 执行状态。
- `runs/reports/<run_id>_summary.json`：完整运行报告。
- `runs/db_records.jsonl`：无 MySQL 环境下的轻量业务记录，按 `run_id` 幂等更新。
- MySQL `preprocessing_runs`：运行级幂等记录。
- MySQL `subject_qc_results`：被试级 QC 指标、问题与人工放行状态。

checkpoint 是控制平面状态，MySQL/JSONL 是业务结果，二者职责不同。

报告通过同目录临时文件和原子替换提交。JSONL 在此基础上使用 advisory file lock，防止并行任务执行 read-modify-write 时相互覆盖；MySQL 使用事务、主外键和 `ON DUPLICATE KEY UPDATE` 保证运行级与被试级记录一致且幂等。数据库节点在暂时故障时指数退避重试，耗尽后把错误写回 Graph State 并继续生成失败报告。本地 MySQL 8.4 服务由 `scripts/mysql_local.sh` 管理，只绑定回环地址并使用独立持久化 volume。
