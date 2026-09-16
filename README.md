# Neuro Preprocess Agent

面向神经影像数据的 LangGraph Supervisor + Worker 自动化流水线。用户用自然语言指定本地数据、URL 或 OpenNeuro 数据集，系统生成计划并等待确认，然后完成数据获取、BIDS 准备、fMRIPrep、规则 QC、QC 门禁入库和结构化报告。

当前版本加入视觉 QC shadow 基础设施：生成标准化 QC packet、合成异常、DINOv2 embedding，
并可用本地 Qwen2.5-VL 输出结构化视觉审查。项目提供盲标优先的本地人工标注界面，
人工结论写入 append-only 审计账本。VLM 尚未经过本项目数据微调和多中心验证，其证据不影响数据库门禁。

最新十被试验证中 fMRIPrep 10/10 完成，规则与定量 QC 识别出一名功能 mask/配准异常被试并触发人工拒绝，整批数据库写入被阻断。详见 [十被试稳定性验证](docs/stability_validation_10subjects_20260914.md)。

跨数据集 pilot 进一步覆盖多 session、双 run、儿童和旧版 BIDS 元数据；验证过程中发现并修复失败路由、OpenNeuro 缓存幂等和完整 BIDS schema 预检问题。详见 [跨数据集验证](docs/cross_dataset_validation_20260916.md)。

## 主流程

```text
load_config -> supervisor -> plan_approval -> preflight
                              |
                              v
preflight -> source -> fetch -> BIDS prepare -> input_qc -> fMRIPrep -> output_qc -> db -> report
                                              |                         |
                                              +-> input review          +-> output review/report
```

Supervisor 负责计划，Worker 负责确定性执行。LLM 不能直接拼接 Docker 命令、写数据库或绕过 QC；这些约束由代码和图路由执行。

## 已实现能力

- 自然语言识别本地绝对路径、OpenNeuro `dsXXXXXX`、URL、并行数和运行模式。
- LangGraph `interrupt()` 人工确认；修改计划后再次展示，而不是直接执行。
- `mock`、`dry_run`、`run` 三级运行模式。
- 本地数据隔离复制，避免删除原始数据或覆盖其他运行的 staging 目录。
- 自动识别 BIDS、DICOM 和 NIfTI；DICOM 支持显式 participant/session job，NIfTI 支持多被试、多 session manifest。
- DICOM header 检查不读取像素并对患者标识做哈希；一个转换 job 混入多个患者或检查时直接拒绝执行。
- 使用 PyBIDS 做预检，识别被试和 T1w/BOLD 模态，并检查空文件与 BOLD `RepetitionTime`。
- fMRIPrep 分发前检查原始 T1w 的可读性、shape、voxel size、非有限值和非零体素占比；疑似预去颅骨或多 session 状态混杂时暂停并按被试确认 `skip/force/auto`。
- 使用 LangGraph `Send` 按 subject 并行运行 fMRIPrep，分别保存状态和日志后再聚合。
- 每次运行使用独立 log/work 目录；同一被试使用文件锁，图重试时可复用同次已完成结果。
- DICOM/NIfTI 转换后的 BIDS 目录按 `run_id` 隔离，防止不同任务混合数据。
- 按输入模态检查关键输出、NIfTI header、影像/mask shape、fMRIPrep 日志、运动指标和视觉报告产物。
- 从 NIfTI 和 fMRIPrep reportlets 生成被试级 QC packet；支持五类可追踪合成异常、DINOv2 embedding 和 Qwen2.5-VL shadow 审查。
- QC 使用 `completed`、`review_required`、`failed` 三态；软阈值告警进入第二个 LangGraph interrupt，人工决定和原因会留痕，硬错误不能人工绕过。
- 正式执行前检查数据权限、输出目录、磁盘、Docker、fMRIPrep 镜像、FreeSurfer license、DICOM 工具、资源预算和数据库连接；硬失败直接生成报告。
- 定量 QC 保存非有限值、强度分布、mask 体积/边界、BOLD tSNR 和标准空间结构/功能 mask Dice，阈值按 hard fail 与人工复核两级配置。
- JSONL 和 MySQL 两种运行记录后端；MySQL 同时保存运行级记录与被试级 QC 指标。
- fMRIPrep 参数使用 Pydantic 结构化白名单生成，LLM 只能建议受管字段，不能覆盖路径、participant、work dir 或 Docker 命令。
- 报告使用原子替换，JSONL 使用跨进程锁和原子写，支持并发任务。
- CLI 使用 SQLite checkpoint；LangGraph API 使用平台持久化，并支持稳定 thread id、HITL 恢复和结构化事件轨迹。
- 新任务不得复用已有 `thread_id`，已完成任务不得再次 resume，避免 checkpoint 状态污染。
- 下载、LLM 和数据库节点使用 LangGraph `RetryPolicy` 处理瞬时错误。
- Pydantic 严格拒绝未知配置字段，避免参数拼写错误被静默忽略。
- `doctor`、`tools`、`status`、`mysql-init` 和对话式 CLI。

## 安装

最小安装不需要影像数据、GPU 或数据库，可运行 mock 流程和自动测试：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,llm,mysql,openneuro,visual-qc]"
bash scripts/run_demo.sh
```

真实预处理需自行准备 Docker 镜像、可信 BIDS/DICOM 数据及 FreeSurfer license。
`dcm2bids` 默认从 PATH 查找；license 放在被忽略的 `data/private/license.txt`，或通过
`FS_LICENSE` 指定绝对路径。DICOM 转换配置必须按实际扫描协议验证，
`configs/dcm2bids.example.json` 中的占位匹配条件不能直接用于正式转换。
视觉模型是可选模块，先安装与本机驱动兼容的 GPU PyTorch，再安装 `.[visual-model]`。
个人服务器环境重建脚本不随公开版本分发。

不需要 GPU 时，可以使用普通 Python 3.11 环境安装所有组件：

```bash
python3.11 -m venv .venv-server
.venv-server/bin/python -m pip install -e ".[server,llm,mysql,openneuro,visual-qc]"
```

先检查环境：

```bash
.venv/bin/python -m neuro_preprocess_agent.cli doctor --config configs/example.json
```

## 使用

直接进入对话：

```bash
.venv/bin/python -m neuro_preprocess_agent.cli
```

示例输入：

```text
帮我处理 OpenNeuro ds000001，先试运行
处理本地 /path/to/bids，正式运行，并行数为2
```

系统会用自然语言展示来源、模式、步骤、需要授权的操作、输出目录和并行数。可以回复：

```text
继续
停止
并行数改成1
数据来源改成本地/path/to/data
跳过入库
只做预处理
```

修改计划后会再次确认。

脚本化运行：

```bash
.venv/bin/python -m neuro_preprocess_agent.cli run \
  --request "检查本地 /path/to/bids，试运行，跳过入库" \
  --config configs/example.json \
  --yes
```

`--yes` 只自动确认初始执行计划；运行前资源警告、输入 T1w 候选异常和输出 QC 告警仍会暂停，不能被批量授权绕过。
交互模式在 fMRIPrep 等长任务执行期间每 30 秒打印心跳和最新日志路径，避免把正常计算误认为 CLI 卡死。

对普通 NIfTI 功能像，必须提供可信的重复时间。可在相邻 JSON sidecar 中提供，或配置：

```json
{
  "preprocess": {
    "nifti_subject_label": "001",
    "nifti_task_name": "rest",
    "nifti_metadata": {"RepetitionTime": 2.0}
  }
}
```

这只能完成简单单被试数据的最小 BIDS 整理。多被试、多 session NIfTI 必须提供 `preprocess.nifti_manifest`；正式 DICOM 队列应使用 `preprocess.dicom_jobs` 明确每个 participant/session 的目录和已验证 dcm2bids 配置。包含本地病例信息的配置不随公开版本分发。

fMRIPrep 参数不要直接拼字符串，应配置 `preprocess.fmriprep_options`：

```json
{
  "fmriprep_options": {
    "output_spaces": ["MNI152NLin2009cAsym:res-2"],
    "nprocs": 8,
    "omp_nthreads": 2,
    "memory_mb": 24000,
    "subject_anatomical_reference": "first-lex"
  }
}
```

参数选择原则见 [fMRIPrep 参数策略](docs/fmriprep_parameter_policy.md)。

输入 T1w 检查默认启用。它使用非零体素占比做预去颅骨候选筛查，默认阈值为 `0.3`；
候选结果必须人工确认，不能视为医学判断。确认时可直接回复：

```text
继续
sub-04 使用 skip
所有被试使用 auto
停止
```

显式配置 `fmriprep_options.skull_strip_t1w=skip/force` 会覆盖自动建议，但输入不可读、维度异常或含非有限值仍会暂停复核。

已标注 `sub-04` 的独立回归配置为 `configs/openneuro_ds000001_sub04_skullstrip_regression.json`。
它读取本地 OpenNeuro 缓存，并写入新的 derivatives 目录以便和旧结果比较。

QC 运动复核阈值可在配置中调整：

```json
{
  "qc": {
    "fd_threshold_mm": 0.5,
    "max_mean_fd_mm": 0.5,
    "max_fd_outlier_fraction": 0.2,
    "max_mean_std_dvars": 1.5,
    "block_database_on_review": true
  }
}
```

这些是工程默认值，不是适用于所有研究的统一标准。正式分析应按研究方案、年龄群体和任务类型调整，并记录阈值版本。

暂停后跨进程恢复：

```bash
.venv/bin/python -m neuro_preprocess_agent.cli run \
  --thread-id task-001 \
  --resume "并行数改成1"
```

查看持久化线程状态：

```bash
.venv/bin/python -m neuro_preprocess_agent.cli status --thread-id task-001
```

## MySQL

项目使用绑定到 `127.0.0.1:3306` 的 MySQL 8.4 Docker 服务。首次启动会生成 `.env.mysql.local`、持久化 volume、应用用户和两张表：

```bash
scripts/mysql_local.sh start
```

CLI 会自动读取项目根目录下被忽略版本控制的 `.env.mysql.local`，也可在当前 shell 手动加载：

```bash
set -a
source .env.mysql.local
set +a
export NEURO_AGENT_MYSQL_PASSWORD="$MYSQL_PASSWORD"
```

使用 `configs/mysql.docker.json` 的连接信息。`scripts/mysql_local.sh status|stop|logs|shell` 分别用于查看、停止、看日志和进入 SQL；只有 `runtime.mode=run` 且 QC 自动通过或人工审查通过时才会写入。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/neuro-agent-eval
```

真实数据回归需要自行准备数据与配置，不随公开版本分发。已有服务器的套件位于
被忽略的 `local/evals/suites/real_golden_smoke.json`；公开 CI 仅执行合成数据与故障注入测试。

自动测试覆盖完整 mock 闭环、多阶段 HITL、输入 T1w 筛查与逐被试参数改写、OpenNeuro 选择性下载配置、严格配置、fMRIPrep 参数门禁、NIfTI manifest、subject fan-out、BIDS 预检、运行前检查、定量 QC、数据库重试收敛、线程生命周期、并发/幂等 JSONL、权限拒绝、失败报告和视觉 QC 数据构建。

离线系统评测进一步覆盖正常双被试闭环、QC 硬失败拦截、无效输入、人工修改计划、人工拒绝 QC 和瞬时下载故障恢复。质量门禁要求所有用例通过且未通过 QC 的任务误入库次数为 0；JSON、Markdown 报告和隔离工作目录写入 `evals/reports/`。详细定义和真实数据验收路线见 [评测文档](docs/evaluation.md)。

并发评测通过生产 LangGraph 图执行 12 个固定时长 subject 任务：`max_workers=4` 时相对
单 worker 达到 `3.80x` 加速和 `94.9%` 并行效率，且实测峰值并发未超过配置上限。既有
七被试真实 fMRIPrep 计算的峰值并发为 4、整体吞吐为 `4.16 subjects/hour`；这些真实数据
属于历史运行观测，不等同于受控加速实验。复现命令和口径见
[并发评测](docs/concurrency_benchmark_20260916.md)。

视觉 QC 的数据构建、标签和模型命令见 [视觉 QC 文档](docs/visual_qc.md)。
人工标注界面及盲标规范见 [人工标注说明](docs/human_annotation.md)。
完整 VLM 流水线配置见 `configs/visual_qc_vlm_shadow.json`；建议先运行离线
`vlm-review`，确认模型对已知待复核被试的排序和理由，再通过主图复用 fMRIPrep 产物。
当前 3B 模型的零样本质量和限制见 [VLM 基线报告](docs/vlm_baseline_20260914.md)。

LangGraph 开发服务器使用单独的 Python 3.11 环境启动：

```bash
.venv-server/bin/langgraph dev --no-browser
```

`langgraph.json` 导出的是不带自定义 checkpointer 的平台图；CLI 通过 `build_local_graph()` 使用本地 SQLite。fMRIPrep 依赖宿主机 Docker，因此当前部署目标是本地工作站或具备 Docker 执行能力的服务器。

## 目录

```text
src/neuro_preprocess_agent/
  graph.py                 # StateGraph、Send fan-out、重试、HITL 和边
  runtime.py               # CLI/API 共用的运行与恢复入口
  config.py                # Pydantic 配置模型和路径解析
  state.py                 # LangGraph 共享状态
  events.py                # 结构化事件
  io_utils.py              # 文件锁和原子写
  agents/supervisor.py     # LLM 计划 + 硬约束调度
  tools/registry.py        # Worker 注册和权限门禁
  tools/source.py          # 数据源解析与获取
  tools/preflight.py       # 工具、服务、存储和资源预算检查
  tools/preprocess.py      # BIDS 准备和 fMRIPrep
  tools/qc.py              # 规则 QC
  tools/qc_metrics.py      # mask、tSNR、重叠度等定量指标
  tools/workers.py         # Worker 组装、入库和报告
  evaluation.py            # 用例执行、故障注入、指标与评测报告
  db.py                    # JSONL/MySQL 持久化
evals/
  fixtures/                # 离线评测基础配置
  suites/                  # 可版本化的 Golden Case 定义
  reports/                 # 本地评测产物（不入版本控制）
```

详细设计见 [架构文档](docs/architecture.md)、[变更日志](CHANGELOG.md) 和 [开源架构借鉴](docs/open_source_agent_architecture_notes.md)。
