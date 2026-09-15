# 公开 MRI 异常与 QC 数据来源

## 目标

这些数据用于建立视觉 QC 的真实异常评测集，而不是直接替代 fMRIPrep 的规则 QC。数据按标签可靠性分层：专家评分数据用于独立测试；人工评分有分歧的数据进入复核池；手工脑掩膜用于验证 skull stripping。

## 已接入来源

### MR-ART（OpenNeuro ds004173）

- 148 名健康成人，共 436 张 T1w。
- 同一被试包含 standard、headmotion1、headmotion2，适合构建匹配的正常/异常对照。
- 专家评分为 `1=good, 2=medium, 3=bad`；本项目映射为 `pass / needs_review / fail`。
- 数据自带 MRIQC 指标和 HTML 报告，许可证为 CC0。
- 当前先下载 5 名被试的 15 张影像及对应 MRIQC 文件作为 pilot。
- 数据主页：<https://openneuro.org/datasets/ds004173/versions/1.0.2>
- 数据说明论文：<https://doi.org/10.1038/s41597-022-01694-8>

### 受控运动与运动校正数据（OpenNeuro ds004332）

- 22 名健康成人，包含 still、nod、shake 条件，以及是否启用 prospective motion correction 和 selective reacquisition 的组合。
- 覆盖 T1w、T2w、FLAIR 等模态。
- 两名技师和一名神经放射科医生分别评分；专家分数为 `1=worst, 5=best`。
- 适合评估模型能否识别真实运动伪影，以及是否对运动校正后的图像产生合理排序。
- 数据主页：<https://openneuro.org/datasets/ds004332>

### ABIDE-I / PCP

- 本地已有 1035 名被试、约 111 GB 的 CPAC 功能像衍生数据。
- 784 例两名评分者均为 OK；251 例存在 `maybe/OK`、`fail/OK` 或 `OK/fail` 分歧。
- 分歧样本不能直接当作负样本，只能标为 `needs_review`，经本项目人工平台复核后再进入训练集。
- 它的优势是多中心和功能像规模大，适合测试站点泛化、覆盖不全和功能像伪影。
- PCP 项目仓库：<https://github.com/preprocessed-connectomes-project/abide>

### NFBS 手工脑掩膜

- 125 张 T1w，提供人工修正的 skull-stripped 图像和脑掩膜。
- 这是脑提取的参考标准，不是“异常数据集”。可用于计算 Dice、漏切和多切，补足当前视觉 QC 对脑掩膜错误的验证。
- 完整压缩包约 1.9 GB，暂不作为第一批异常分类数据下载。
- 数据说明与下载：<https://github.com/preprocessed-connectomes-project/NFB_skullstripped>

## 本地目录

影像、逐样本标签和元数据不随源码分发。需要先按数据源条款自行下载。
目录脚本依赖两个元数据仓库，可在项目根目录准备：

```bash
mkdir -p data/public_qc_sources/repos
git clone --depth 1 https://github.com/OpenNeuroDatasets/ds004173.git data/public_qc_sources/repos/ds004173
git clone --depth 1 https://github.com/OpenNeuroDatasets/ds004332.git data/public_qc_sources/repos/ds004332
```

这些仓库的影像通常是 git-annex 指针，不等于下载完成。实际影像使用 OpenNeuro 下载工具获取。
ABIDE 表型表应自行获取并保存在 `data/qc_labels/Phenotypic_V1_0b_preprocessed1.csv`，
然后运行 `scripts/download_abide_pcp_qc_derivatives.py` 生成本地下载 manifest。
没有 ABIDE manifest 时，目录脚本会跳过 ABIDE，而非要求随代码分发表型表。

```text
data/public_qc_sources/
├── repos/                         # 数据集元数据仓库，不含大影像
├── ds004173_mrart_pilot/          # MR-ART 小规模真实影像 pilot
├── public_qc_catalog.csv          # 统一样本目录
└── public_qc_catalog_summary.json # 数量与标签分布
```

运行以下命令可重建统一目录：

```bash
.venv/bin/python scripts/build_public_qc_catalog.py
```

生成可供现有标注平台读取的盲化图像包：

```bash
.venv/bin/python scripts/build_public_qc_packets.py
```

`manifest.jsonl` 不含参考答案；专家标签和 PCP 弱标签单独保存在
`reference_labels.jsonl`，评估时再按 `sample_id` 合并。

## 使用边界

1. 专家分数不是疾病诊断标签，只表示图像质量或伪影严重度。
2. 同一被试的不同扫描必须放在同一个 train/validation/test 分区，防止身份泄漏。
3. MR-ART 和 ds004332 的专家标签优先保留为最终测试集；合成异常主要用于训练和功能测试。
4. ABIDE 的评分分歧不能自动映射为 fail，必须人工复核。
5. 模型第一版保持 shadow mode，不据此自动阻止 MySQL 入库。
