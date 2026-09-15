# ABIDE/PCP 视觉 QC 数据集构建方案

更新时间：2026-06-16

## 1. 背景

用户希望用视觉模型替代一部分人工 QC 工作。当前项目已经完成：

```text
本地 BIDS 数据
-> fMRIPrep
-> 规则式 QC
-> 入库
-> 报告生成
```

下一步希望构建视觉 QC 数据集，用于训练或评估模型判断预处理后的影像是否存在明显问题，例如：

- skull stripping 失败
- 脑组织被裁剪
- 配准错误
- 功能像信号异常
- carpet plot 明显异常
- motion / artifact 过重

ABIDE/PCP 提供了一个有价值的数据来源：

```text
Phenotypic_V1_0b_preprocessed1.csv
```

该 CSV 中包含：

1. QAP/PCP 自动计算的数值 QC 指标
2. 多个 rater 的人工 QC 标签
3. 人工 notes
4. 最终样本纳入标志

对应网站：

- PCP datasets 页面：<http://preprocessed-connectomes-project.org/datasets.html>
- ABIDE preprocessed 页面：<https://fcon_1000.projects.nitrc.org/indi/abide/preprocessed.html>
- ABIDE PCP GitHub：<https://github.com/preprocessed-connectomes-project/abide>

说明：PCP 主站部分页面当前可能返回 502，但 NITRC 镜像页面和 GitHub 仓库可访问。

## 2. PCP/ABIDE 数据来源

ABIDE I 包含 1112 个结构像和静息态功能像数据集，来自多个站点。PCP 对 ABIDE 数据做了预处理，并提供下载脚本和表型/QC 文件。

NITRC 页面说明：ABIDE 数据由不同团队使用各自工具进行预处理，functional preprocessing 涉及：

- CCS
- CPAC
- DPARSF
- NIAK
- 以及 LLE 相关派生数据

PCP GitHub 仓库中包含：

```text
Phenotypic_V1_0b_preprocessed1.csv
download_abide_preproc.py
download_abide_preproc_guide.txt
```

`download_abide_preproc.py` 的作用是：根据 derivative、pipeline、strategy 和部分表型筛选条件，从 FCP-INDI 的 S3 bucket 下载 ABIDE preprocessed outputs。

下载脚本支持的 derivative 包括：

```text
alff
degree_binarize
degree_weighted
eigenvector_binarize
eigenvector_weighted
falff
func_mask
func_mean
func_preproc
lfcd
reho
rois_aal
rois_cc200
rois_cc400
rois_dosenbach160
rois_ez
rois_ho
rois_tt
vmhc
```

支持的 pipeline：

```text
ccs
cpac
dparsf
niak
```

支持的 strategy：

```text
filt_global
filt_noglobal
nofilt_global
nofilt_noglobal
```

注意：PCP 下载脚本中默认包含 `mean_fd_thresh = 0.2` 的筛选逻辑，这会影响下载到的样本分布。做训练集时不要无意识使用这个筛选，否则会人为减少高运动/差质量样本。

## 3. 本地 CSV 字段结构

本地文件：

`data/qc_labels/Phenotypic_V1_0b_preprocessed1.csv`（本地文件，不随源码分发）

总行数：

```text
1112 subjects
```

总列数：

```text
106 columns
```

QC 相关字段位于 CSV 后半部分。

## 4. 工具自动计算的 QC 指标

以下字段是工具计算的数值指标，不是人工标签：

```text
anat_cnr
anat_efc
anat_fber
anat_fwhm
anat_qi1
anat_snr

func_efc
func_fber
func_fwhm
func_dvars
func_outlier
func_quality
func_mean_fd
func_num_fd
func_perc_fd
func_gsr
```

含义：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `anat_cnr` | anatomical | contrast-to-noise ratio |
| `anat_efc` | anatomical | entropy focus criterion |
| `anat_fber` | anatomical | foreground-background energy ratio |
| `anat_fwhm` | anatomical | spatial smoothness |
| `anat_qi1` | anatomical | artifact voxel proportion |
| `anat_snr` | anatomical | signal-to-noise ratio |
| `func_efc` | functional | functional EFC |
| `func_fber` | functional | functional FBER |
| `func_fwhm` | functional | functional spatial smoothness |
| `func_dvars` | functional | DVARS |
| `func_outlier` | functional | outlier fraction |
| `func_quality` | functional | time-series quality metric |
| `func_mean_fd` | motion | mean framewise displacement |
| `func_num_fd` | motion | number of FD-outlier volumes |
| `func_perc_fd` | motion | percent FD-outlier volumes |
| `func_gsr` | functional | ghost-to-signal ratio |

这些指标适合进入当前项目的规则 QC：

```text
rule_file_image_qc
-> motion thresholds
-> DVARS thresholds
-> outlier thresholds
-> anatomical/functional quality metric outlier detection
```

## 5. 人工审计标签

以下字段是人工 rater 结果：

```text
qc_rater_1
qc_notes_rater_1

qc_anat_rater_2
qc_anat_notes_rater_2

qc_func_rater_2
qc_func_notes_rater_2

qc_anat_rater_3
qc_anat_notes_rater_3

qc_func_rater_3
qc_func_notes_rater_3
```

这些字段是视觉模型训练/评估最重要的标签来源。

### 5.1 rater 字段分布

从本地 CSV 统计得到：

```text
qc_rater_1:
  OK     997
  fail   111
  maybe    4

qc_anat_rater_2:
  OK       866
  maybe    194
  fail      40
  blank     12

qc_func_rater_2:
  OK       842
  maybe    207
  fail      51
  blank     12

qc_anat_rater_3:
  OK      1003
  fail     108
  blank      1

qc_func_rater_3:
  OK      1035
  fail      76
  blank      1
```

### 5.2 notes 字段示例

人工 notes 中出现的典型问题：

```text
skull-striping fail
Motion
headmotion
noise
half head
no func images
no anat images
ic-cerebellum
ic-parietal
signal drop off too much
registration error
dorsal cropped
ventral edge is cropped
wraparound
```

这些 notes 可以转成更细的错误类型标签。

## 6. 最终样本纳入字段

字段：

```text
SUB_IN_SMP
```

本地统计：

```text
SUB_IN_SMP = 1: 763
SUB_IN_SMP = 0: 349
```

解释：

- `1`：最终纳入样本
- `0`：最终未纳入样本

注意：`SUB_IN_SMP` 不应简单等同于视觉 QC 标签。它可能混合了：

- 图像质量
- 数据完整性
- motion
- 站点处理问题
- 表型/样本筛选逻辑

因此它适合作为最终可用性标签，但不适合作为唯一视觉训练标签。

## 7. 推荐标签构造策略

不要直接用单个 rater 字段训练。

推荐构造 consensus labels。

### 7.1 Anatomical QC 标签

使用：

```text
qc_anat_rater_2
qc_anat_rater_3
qc_anat_notes_rater_2
qc_anat_notes_rater_3
```

规则：

```text
anat_pass:
  qc_anat_rater_2 == OK
  and qc_anat_rater_3 == OK

anat_fail:
  qc_anat_rater_2 == fail
  and qc_anat_rater_3 == fail

anat_needs_review:
  rater 不一致
  or 出现 maybe
  or 任一 rater 为空
```

### 7.2 Functional QC 标签

使用：

```text
qc_func_rater_2
qc_func_rater_3
qc_func_notes_rater_2
qc_func_notes_rater_3
```

规则：

```text
func_pass:
  qc_func_rater_2 == OK
  and qc_func_rater_3 == OK

func_fail:
  qc_func_rater_2 == fail
  and qc_func_rater_3 == fail

func_needs_review:
  rater 不一致
  or 出现 maybe
  or 任一 rater 为空
```

### 7.3 不建议的做法

不建议这样做：

```text
把 OK / maybe / fail 全部直接当三分类标签训练
```

原因：

- `maybe` 本质是人工不确定，不是稳定类别
- rater 间可能不一致
- notes 表示的问题类型不统一
- 不同 pipeline 的图像风格可能不同

## 8. 图像输入如何构建

CSV 只提供标签，不提供图像。

视觉模型训练需要把每个 subject 映射到图像文件。

有两条路线。

### 路线 A：使用 PCP preprocessed derivatives 生成视觉 QC 图

从 PCP 下载以下 derivative：

```text
func_mean
func_mask
func_preproc
```

可选：

```text
alff
falff
reho
vmhc
```

然后自己生成标准化 QC panel：

```text
anat panel:
  T1w / brain mask / skull-strip overlay

func panel:
  mean functional image
  functional mask overlay
  selected slices of func_preproc
  motion/confounds plot
```

优点：

- 可以直接利用 PCP 下载脚本
- 可以批量获取很多 subjects
- 与 CSV 中人工 QC 标签更接近

缺点：

- PCP derivatives 不是 fMRIPrep 输出
- 没有 fMRIPrep HTML report
- 需要自己生成 QC 图

### 路线 B：重新跑 fMRIPrep，生成统一风格的 QC 图

流程：

```text
ABIDE raw/BIDS
-> 本项目 fMRIPrep
-> fMRIPrep HTML + figures
-> 使用 CSV 人工标签作为弱标签
```

优点：

- 图像风格与当前项目一致
- 直接训练/评估 fMRIPrep 视觉 QC
- 能和当前 pipeline 无缝结合

缺点：

- 计算成本高
- 需要 raw/BIDS ABIDE 数据
- CSV 标签来自 PCP 处理流程，与 fMRIPrep 结果存在 domain shift

### 推荐

第一阶段采用路线 A：

```text
PCP derivatives
-> 自动生成 QC panel
-> 用 CSV rater 标签做训练/评估
```

第二阶段对少量样本采用路线 B：

```text
重跑 fMRIPrep
-> 检查 label transfer 是否可靠
-> 逐步积累本项目自己的人工复核标签
```

## 9. 训练样本表设计

建议生成一个新的 manifest：

```text
docs/abide_pcp_visual_qc_manifest.csv
```

字段：

```csv
subject_id,file_id,site_id,dx_group,modality,pipeline,strategy,derivative,image_path,label,rater_2,rater_3,notes,sub_in_smp,metric_json
```

示例：

```csv
50002,Pitt_0050002,PITT,1,anat,pcp,unknown,anat_panel,panels/Pitt_0050002_anat.png,pass,OK,OK,,1,"{...}"
50002,Pitt_0050002,PITT,1,func,cpac,nofilt_noglobal,func_panel,panels/Pitt_0050002_func.png,fail,fail,fail,"ic-cerebellum",0,"{...}"
```

## 10. 视觉模型任务定义

建议不要一开始定义为“代替人工最终裁决”。

更合适的任务定义：

```text
自动视觉 QC 初筛 / 排序 / 辅助复核
```

模型输出：

```json
{
  "decision": "pass | needs_review | fail",
  "confidence": "low | medium | high",
  "visible_issues": [
    "skull stripping error",
    "registration error",
    "cropped brain",
    "signal dropout"
  ],
  "recommendation": "accept | human_review | reject"
}
```

训练/评估时：

- `pass` 和 `fail` 使用 rater 共识样本
- `needs_review` 使用 rater 分歧和 maybe 样本
- 人工 notes 用作解释性标签，不一定第一版就训练

## 11. 与当前项目的结合方式

当前 `qc_node` 已经输出：

```text
visual_qc_artifacts
```

后续可以新增或扩展：

```text
visual_qc_dataset_builder
visual_qc_model
visual_qc_review
```

推荐流程：

```text
rule_file_image_qc
-> collect_visual_artifacts
-> VLM zero-shot review
-> embedding anomaly score
-> human review queue
-> DB record
```

其中 ABIDE/PCP 数据用于：

1. 建立视觉 QC 评估集
2. 建立正常样本库
3. 构建 fail/maybe 样本
4. 测试 VLM 提示词
5. 训练 embedding classifier

## 12. 第一版落地任务

建议先做以下 4 件事：

### Step 1：解析 CSV 标签

输入：

```text
data/qc_labels/Phenotypic_V1_0b_preprocessed1.csv
```

输出：

```text
data/qc_labels/abide_pcp_qc_labels.csv
```

包含：

```text
SUB_ID
FILE_ID
SITE_ID
anat_label
func_label
anat_notes
func_notes
qap_metrics
SUB_IN_SMP
```

### Step 2：下载少量 PCP derivatives

先不要全量下载。

建议选择：

```text
pipeline = cpac
strategy = nofilt_noglobal
derivatives = func_mean, func_mask, func_preproc
```

先下载每类：

```text
pass 20 个
fail 20 个
needs_review 20 个
```

### Step 3：生成 QC panel

把 NIfTI 转成 PNG panel：

```text
middle axial / coronal / sagittal slices
mask overlay
functional mean image
```

输出：

```text
data/qc_panels/abide_pcp/
```

### Step 4：跑 VLM baseline

先不用训练。

使用：

```text
Qwen2.5-VL 7B
```

让模型输出：

```text
pass / needs_review / fail
```

再与 rater consensus label 对比。

## 13. 后续路线

```text
ABIDE/PCP labels
-> 构建视觉 panel
-> VLM zero-shot/few-shot
-> embedding + 轻量分类器
-> 人工复核 queue
-> 积累本项目真实标签
-> 再考虑 LoRA 微调
```

重点：当前阶段不要直接做 VLM 微调。先把标签、图像、评估集、错误类型和人工复核机制跑通。
