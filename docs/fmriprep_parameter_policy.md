# fMRIPrep 参数策略

## 为什么当前不做成 Skill

Codex/Deep Agents 的 `SKILL.md` 适合给语言 Agent 按需加载操作知识，但本项目当前是显式 `StateGraph`，没有 Skill loader。更重要的是，fMRIPrep 参数会直接影响科研产物，不能只靠提示词约束。

当前采用四层设计：

1. 用户用自然语言表达资源和处理需求。
2. Supervisor 只能返回 `FMRIPrepPlanUpdate` 中定义的结构化字段。
3. `AppConfig` 用 Pydantic 检查类型、范围和枚举值。
4. `tools/preprocess.py` 生成确定性命令，并在第一次 HITL 中把关键参数展示给用户。

在 BIDS 准备完成后，系统还会运行一次输入 T1w 筛查。默认 `skull_strip_t1w=auto` 时，
明显保留头部或仅做面部脱敏的输入使用 `force`；疑似已去颅骨的输入建议 `skip` 并触发
`interrupt()`，由人工按被试确认。检测依据、建议、人工覆盖和最终参数都会进入运行报告。

因此，现阶段“版本化参数 schema + 硬校验 + 人工确认”比单独写一个 Skill 更可靠。以后若 Supervisor 改为支持动态 Skill 加载，可以增加一个 fMRIPrep Skill 用于解释参数和提出建议，但最终仍必须经过现有 schema 和命令构造器。

## 版本基线

- 容器：`nipreps/fmriprep:latest`
- 本机验证版本：fMRIPrep `25.2.5`
- 官方用法：<https://fmriprep.org/en/stable/usage.html>

官方要求输入为有效 BIDS，通常至少包含 T1w 和 BOLD；`--anat-only` 时可只处理解剖像。项目不会因为用户说“跳过验证”就忽略自己的 PyBIDS 预检。

## 受管参数

`preprocess.fmriprep_options` 当前管理：

- 输出：`output_layout`、`output_spaces`、`level`
- 资源：`nprocs`、`omp_nthreads`、`memory_mb`、`low_mem`
- 工作流：`anat_only`、`ignore`、`force`
- 纵向/多 session：`subject_anatomical_reference`
- 解剖：`skull_strip_t1w`、`fs_no_reconall`
- 运行保护：`skip_bids_validation`、`stop_on_first_crash`

输入、输出、participant、work dir 和 FreeSurfer license 由程序管理，禁止通过 `fmriprep_args` 覆盖。`fmriprep_args` 只保留为有限的专家扩展入口；新增常用参数时应优先加入 `FMRIPrepOptions`，而不是长期堆积自由字符串。

## 选择原则

- 不从 DICOM/NIfTI 文件名猜 `RepetitionTime`、相位编码或总读出时间。
- 不根据模型常识自动选择 `--ignore fieldmaps` 或 `--force syn-sdc`。
- 多 session 数据应由研究设计决定使用 `first-lex`、`unbiased` 或 `sessionwise`。
- `max_workers` 是同时运行的被试数，`nprocs` 是每个 fMRIPrep 容器的进程上限；总资源大致受两者乘积影响。
- `--low-mem` 会增加工作目录磁盘占用，磁盘紧张时不能仅为了省内存开启。
- `--skip-bids-validation` 默认关闭；只有外部 validator 已独立通过且原因已记录时才考虑开启。
- 非零体素占比只是预去颅骨候选筛查信号，不足以替代影像审查；低于
  `pre_skull_stripped_nonzero_fraction` 的样本必须人工确认，不能静默改参数。

## 接口检查

升级镜像后执行：

```bash
scripts/check_fmriprep_interface.sh
```

该检查会读取容器实际版本和 `--help`，确认项目使用的受管参数仍存在。升级后还应对一份小 BIDS fixture 做 dry-run，并对一名真实被试做回归验证。
