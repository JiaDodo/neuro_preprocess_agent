# 项目评测与质量门禁

## 1. 目标

本项目的评测对象不是单一模型，而是“数据输入、LangGraph 调度、预处理、QC、人工复核、数据库门禁、报告”组成的完整系统。评测必须同时回答：

1. 正常任务能否稳定完成；
2. 错误数据和异常结果能否被拦截；
3. 暂时性故障能否自动恢复；
4. 人工修改与复核是否留下可审计结果；
5. 未通过 QC 的数据是否可能误入数据库。

核心安全指标是 `false_pass_count`：已标记为必须阻断入库的用例，却被 QC 放行或触达数据库 Worker 的次数。当前质量门禁要求所有用例通过且该指标为 0。

## 2. 当前离线评测

默认 suite 为 `evals/suites/offline_smoke.json`，不访问网络、不启动 Docker、不占用 GPU、不连接 MySQL。每个用例使用独立的 `InMemorySaver`、thread ID、配置文件和工作目录，避免不同运行互相污染。

现有场景：

| 场景 | 验证目标 |
|---|---|
| 正常双被试任务 | 审批、fan-out、QC、模拟入库和报告闭环 |
| QC 硬失败 | 失败结果不能进入数据库 |
| 本地源不存在 | 上游错误生成失败报告且阻止入库 |
| 人工修改计划 | 修改后再次确认，并可删除入库步骤 |
| 人工拒绝软 QC | 复核意见被记录，数据库被阻断 |
| 下载瞬时故障 | LangGraph RetryPolicy 重试后恢复 |
| 数据库瞬时故障 | 数据库节点重试后恢复并保持幂等 |
| 数据库持续故障 | 重试耗尽后生成结构化失败报告 |

运行全部离线评测：

```bash
.venv/bin/neuro-agent-eval
```

也可以直接运行源码入口：

```bash
.venv/bin/python evals/run_evals.py
```

按标签或用例筛选：

```bash
.venv/bin/neuro-agent-eval --tag safety
.venv/bin/neuro-agent-eval --case transient_fetch_retry_recovers
```

命令退出码可直接作为 CI gate：通过为 0，失败为 1。结果写入 `evals/reports/`：

- `latest.json`：供 CI、监控或后续 dashboard 读取；
- `latest.md`：供人工审阅和项目展示；
- `workspaces/<timestamp>/<case_id>/`：每个用例的隔离配置、流水线报告和中间记录。

## 3. 用例协议

Suite 使用严格 Pydantic schema 校验。一个用例由以下信息组成：

- `request`：传给 Agent 的自然语言任务；
- `config_overrides`：只覆盖该场景需要变化的配置；
- `responses`：按顺序恢复 LangGraph interrupt，测试 HITL；
- `fault`：在 Worker Registry 边界注入固定结果或瞬时错误；
- `assertions`：对最终 Graph State 做路径断言；
- `must_block_database`：声明该场景绝不能触达数据库。

故障注入只在单个用例的上下文中生效，结束后恢复原 Worker。生产代码不包含 `if testing` 分支。

支持的断言操作符包括 `eq`、`ne`、`contains`、`not_contains`、`exists`、`not_exists`、`length_eq`、`ge` 和 `le`。

首版真实数据 suite 保存于被忽略的 `local/evals/suites/real_golden_smoke.json`，不随公开版本分发。它包含3个 OpenNeuro 单被试案例、1个三被试 fan-out 案例和1个 DICOM 转 BIDS 案例。前三类只执行真实 BIDS 检查和命令规划，DICOM 案例会实际转换但不运行 fMRIPrep。它们来自两个底层数据源，因此只能作为集成回归，不能证明跨站点泛化。

本地 Golden Dataset 可用聚合内容指纹防止输入悄然变化：

```bash
.venv/bin/python scripts/golden_manifest.py create \
  --case ds000001=data/openneuro_sample/ds000001 \
  --case local_dicom=/path/to/your/dicom
.venv/bin/python scripts/golden_manifest.py verify
```

本地 manifest 包含绝对路径和内容哈希，不进入版本控制；结构示例为 `evals/manifests/golden.example.json`。

人工确认的“错误输出/修复输出”使用独立的配对回归 manifest。它同时验证 packet、人工标签、
定量指标阈值和修复前后的改善幅度：

```bash
.venv/bin/python scripts/validate_visual_regressions.py
```

首个案例为预去颅骨的 `ds000001/sub-04`：旧参数输出经人工判为 `fail`，Input QC 建议并经
人工确认使用 `--skull-strip-t1w skip` 后，新输出判为 `pass`。机器可读定义位于
`evals/manifests/visual_qc_golden_pairs.json`。

## 4. 分层评测路线

### L1 单元与契约测试

验证配置约束、BIDS 预检、模态识别、fMRIPrep 命令生成、QC 指标、数据库幂等和报告 schema。由 `unittest` 在每次 CI 中执行。

### L2 离线系统评测

验证 LangGraph 路由、checkpoint/resume、HITL、故障重试、失败隔离和数据库安全门禁。由默认 offline suite 执行。

### L3 真实工具集成评测

使用少量去标识 DICOM、NIfTI 和 BIDS fixture，执行 dcm2bids、fMRIPrep 和 MySQL。建议固定容器 digest、工具版本和期望产物清单，在有 Docker 的自托管 runner 上运行，不放入普通 GitHub Actions。

### L4 Golden Dataset 验收

建立 30 至 50 个有人工结论的用例，至少覆盖正常数据、缺文件、错误元数据、配准失败、头动过大、伪影和多被试/多 session。按数据集和站点拆分 train/reference 与 test，避免同源泄漏。

### L5 稳定性与运营指标

连续运行 10、50、100 个被试，统计被试成功率、整批成功率、失败隔离率、重试恢复率、吞吐、P50/P95 时长、CPU/GPU/内存峰值、磁盘增量和单被试成本。

## 5. VLM QC 的后续接入

VLM 尚不进入阻断链路。接入顺序应为 shadow mode、与人工标签对照、阈值校准，再决定是否参与复核排序。需要报告灵敏度、特异度、AUROC/AUPRC，并重点控制“明显异常被判为正常”的假阴性率。没有足够真实异常前，合成异常只能验证工程链路，不能证明临床或科研泛化能力。

## 6. 当前边界

离线 suite 证明的是调度和安全门禁逻辑，不证明 fMRIPrep 算法精度或视觉 QC 模型效果。真实 MySQL 已验证事务写入、同一 run 更新、被试级记录和 QC 阻断；真实 suite 已覆盖 BIDS 检查与 DICOM 转换，但仍需增加独立站点、多 session 和真实异常数据。
