# 开源 Agent 架构借鉴

调研对象：

- [OpenCode](https://github.com/anomalyco/opencode)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent)
- [OpenClaw](https://github.com/openclaw/openclaw)

## 采用的模式

| 成熟项目中的模式 | 本项目实现 |
| --- | --- |
| 入口与核心运行时分离 | `cli.py` 只负责交互，`runtime.py` 提供统一 start/resume/state |
| Agent profile 与专用 subagent | Supervisor 专职计划，领域 worker 专职执行，不让所有组件拥有相同权限 |
| 中央工具注册表 | `WorkerRegistry` 提供 catalog、dispatch、risk 和输出字段 |
| allow/ask/deny 权限 | `permissions` 配置 + plan approval，deny 在执行层再次检查 |
| SQLite session persistence | LangGraph SQLite checkpointer + 稳定 thread id |
| 可观察工具调用 | 每个节点产生结构化 event，CLI 展示关键状态，报告保留完整轨迹 |
| 统一错误包装 | worker 异常进入 state，失败也生成报告 |
| 平台无关核心 | CLI 通过 `AgentRuntime` 调图，后续 API/网页可以复用同一入口 |
| 配置和插件边界 | Pydantic 配置；领域 worker 与 graph 解耦，可继续增加新数据源/QC 后端 |

## 没有照搬的内容

- 没有实现通用 shell/file-edit 编码 Agent，因为本项目命令集合是受控的神经影像工具。
- 没有引入自由多 Agent 对话。数据依赖由 LangGraph 明确表达。
- 没有实现长期人格记忆、消息平台网关、RAG、cron 和 VLM；这些都不是第一版闭环的必要条件。
- 没有让 LLM 每一步重新规划。首次计划后，正常路径确定性执行，只在异常和 QC 分支应用安全策略。

借鉴的重点是运行时边界、工具治理、会话持久化和可观察性，而不是复制项目体量。
