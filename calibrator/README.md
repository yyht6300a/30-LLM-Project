# calibrator：数据准备、特征与校准器训练

所有者：P1（data_prep、features）、P2（training、eval）。

- `data_prep/`：模型输出快照加载与校验、论文过滤规则、类别平衡、50/50 分层切分、10% 验证集。
- `features/`：离线流行度查找（entity → sitelinks、实体对 → 共现），按论文规则做精确实体匹配；
  表外实体显式报告为未找到（不做模糊实体链接，也不做模型自生成兜底）。
- `training/`：MLP 训练与基线复现。
- `eval/`：ECE / Brier / AUROC / Alignment / Confw 指标与阈值标定。

P1 论文复现结果（数据集、流行度统计、RQ1/RQ2 与论文表格对照）见 [P1_REPRODUCTION_REPORT.md](P1_REPRODUCTION_REPORT.md)。

## data_prep：使用说明

直接读取论文仓库的六个压缩快照 `data/model_outputs/*.jsonl.gz`（每个 26,430 行：
movies 10,964 / songs 2,157 / basketball 13,309，合计 158,580 行），不生成中间 CSV，
不复制数据。

### 依赖

Python ≥ 3.10（已在 3.10 与 3.13 上通过全部测试），`numpy`、`pandas`、`scikit-learn`；测试需要
`pytest`。不需要 torch。

### 数据位置

按以下顺序查找（data_prep 可指向论文仓库根目录，或直接指向 `model_outputs` 目录；features 需要根目录）：

1. 函数参数 `repo_dir=...` / 命令行 `--repo-dir`；
2. 环境变量 `POPULAR_BUT_WRONG_REPO`；
3. 默认 `reference/Popular-but-Wrong`（见 docs/03 的 M0，`reference/` 已被 gitignore）。

features 还需要官方仓库的 `res/` 查找表，需在官方仓库内运行一次 `python scripts/materialize_compact_data.py`
生成（M0 第 4 步已包含）。

```bash
export POPULAR_BUT_WRONG_REPO=~/path/to/Popular-but-Wrong   # 或 clone / 软链到 reference/
python -m calibrator.data_prep                                # 六模型清单与过滤统计
python -m calibrator.data_prep --model qwen2.5-7b --dataset movies   # 三种协议的切分统计
python -m calibrator.features                                 # 离线查找与快照特征逐条比对
pytest calibrator                                             # 全部 P1 测试；找不到数据时集成测试自动 skip
```

### P2 最小用法

```python
from calibrator.data_prep import load_split, load_model_outputs, FEATURE_COLUMNS

split = load_split("Qwen2.5-7B", "Movies", protocol="balanced")  # 或 "balanced_train" / "natural"
X_train, y_train = split.X_train, split.y_train   # 原始特征（未标准化），列序 = FEATURE_COLUMNS
X_val, y_val = split.X_val, split.y_val
X_test, y_test = split.X_test, split.y_test
split.meta_test   # 与 X_test 逐行对齐：model/dataset/index/question/reference/response/correct/...
split.stats       # 过滤原因计数、平衡情况、各子集样本数与正确率

# 消融：任意特征子集，按给定顺序
split_c = load_split("qwen2.5-7b", "movies", features=["confidence", "gene_coo"])

# 循环 18 个组合时先加载一次
records = load_model_outputs()
splits = {(m, d): load_split(m, d, records=records)
          for m in ["qwen2.5-7b", "qwen2.5-14b"] for d in ["movies", "songs", "basketball"]}
```

模型名接受论文仓库写法（`Qwen2.5-7B`、`llama8b`、`chatgpt`）或快照文件名
（`qwen2.5-7b`、`llama3-8b`、`gpt-3.5-turbo`），大小写不敏感。

可用特征：`confidence`、`qpop`、`gene_pop`、`gene_coo`（主特征），以及基线用的
`sc_confidence`、`vc_confidence`、`llm_qpop`、`llm_gene_pop`、`llm_coo`。`gt_pop`、`gt_coo`
等依赖标准答案的字段被禁止作为特征（会泄露标签）。

### 过滤规则（与官方 `calibration_experiment.py` 的 `collect_samples` 一致）

按顺序，每条记录只记第一个原因，计数见 `apply_paper_filters()` 的报告或 `split.stats["excluded"]`：

| 原因 | 规则 |
|---|---|
| `empty_response` | `response` 为空（qwen2-7b 共 214 条） |
| `noisy_entity` | `q_occ > 6000`；movies/songs 另加 `gene_occ > 6000`（缺失的 occ 按 0，不过滤） |
| `missing_confidence` | `confidence` 为 null（过滤空回答后实际为 0） |

保留数与官方 `calibration_results.json` 的 18 个组合完全一致（测试中逐一校验）。

### 缺失值语义

快照中 `null` 表示上游不可得，不等于 0。加载后计数字段为可空 `Int64`，原值不改动。
`missing="official"`（默认，复现论文）：

- `gene_pop`：`gene_pop_found=false` 时快照已存 0，训练时按 0；`found=true` 且值为 0 是真实的 0。
- `gene_coo`、`qpop`、`llm_*`：不可得时填 0（与官方一致）。
- `sc_confidence`、`vc_confidence`：不可得的记录在平衡之后、切分之前剔除（与官方逐实验 NaN 过滤一致）。

每个特征是否可得在 `meta_*` 的 `<feature>_available` 列，填 0 的数量在
`stats["zero_filled_unavailable"]`。`missing="drop"` 为敏感性分析选项（非论文口径）：
在平衡之前剔除所选特征任一不可得的记录。

### 切分协议

每个 (model, dataset) 独立处理，`random_state=42`：

| protocol | 对应官方脚本 | 步骤 |
|---|---|---|
| `balanced`（主表） | `calibration_experiment.py` | 正确率不在 [0.25, 0.75] → 对整个保留池下采样到 1:1 → 50/50 分层切分 |
| `balanced_train` | `setting2_natural_test.py` | 自然分布 50/50 分层切分 → 仅训练半区按同一规则平衡；测试集保持自然分布 |
| `natural` | `setting3_natural_train_test.py`、主表附录 | 50/50 分层切分，不平衡 |

三种协议最后都从训练半区分层切出 10% 作验证集（同官方 `run_mlp` 的早停验证集）。
下采样逐位复刻官方 `balance_samples`（`np.random.default_rng(42)`），切分使用与官方相同的
`train_test_split` 调用，因此样本与顺序都和官方一致。`natural` 的切分规模与正确率与官方
`setting3_results.json` 全部 18 个组合一致。

示例：Qwen2.5-7B / Movies 保留 9,578 条（正确 1,749，正确率 0.183）：

| protocol | train (对/错) | val (对/错) | test (对/错) |
|---|---|---|---|
| balanced | 1,574 (787/787) | 175 (88/87) | 1,749 (874/875) |
| balanced_train | 1,575 (788/787) | 175 (87/88) | 4,789 (874/3,915) |
| natural | 4,310 (787/3,523) | 479 (88/391) | 4,789 (874/3,915) |

### 与官方流程的差异（P2 注意）

1. **数据入口**：直接读压缩快照，不经过 `materialize_compact_data.py` 生成的 `res/` 与查找表；
   结果等价（保留数、平衡规模、natural 切分均已对齐）。
2. **标准化与阈值**：官方在完整训练半区（train ∪ val）上拟合 `StandardScaler`，并在其上
   搜索 Alignment 阈值，之后才切出验证集。精确复现时用 `split.X_trainval` / `split.y_trainval`
   拟合 scaler；更严格的做法是只用 `X_train`。scaler 由 P2 负责，任何情况下都不要用测试集。
3. **验证集**：官方只在 MLP 内部、且训练集 > 100 条时切验证集（否则 val = train）；这里总是切，
   保证三者互不重叠。本数据中最小训练半区为 148 条，因此结果不受影响。
4. **不做的官方跳过逻辑**：官方对 < 50 条或正确率不在 [0.02, 0.98] 的组合跳过实验，本数据中不会触发。
5. 论文主表的平均值不含 gpt-3.5-turbo（只平均开源模型），由 P2 在汇总时处理。

## features：离线流行度查找（PopQ / PopGe / RPopGe）

`calibrator.features.PopularityLookup` 把一对 (问题, 模型回答) 映射为校准器的流行度特征。它只读取
官方预计算表，不需要 Wikipedia、Wikidata、模型或 MCP 服务在运行：

| 特征 | 来源 | 键 |
|---|---|---|
| `qpop`（PopQ） | `data/{dataset}.jsonl` 的 `popularity`（Wikidata sitelinks） | 问题实体，区分大小写 |
| `gene_pop`（PopGe） | `res/gt_gene_entity_popularity_*.jsonl`（Wikidata sitelinks） | 清洗后的回答实体，区分大小写 |
| `gene_coo`（RPopGe） | `res/cooccurrence_*.json`（Wikipedia 文档共现） | 小写问题实体 → 小写回答实体 |
| `q_occ` / `gene_occ` | `res/single_occurrence_*.json`（Wikipedia 文档数） | 小写实体，仅用于噪音实体判断 |

### P3 最小用法

```python
from calibrator.features import PopularityLookup

lookup = PopularityLookup()          # 启动时建一次（约 1 秒），之后复用
r = lookup.lookup("Who is the director of the movie The Intouchables", "Eric Toledano")
# {'dataset': 'movies', 'question_entity': 'The Intouchables', 'gene_entity': 'Eric Toledano',
#  'qpop': 60, 'gene_pop': None, 'gene_coo': 2,
#  'qpop_found': True, 'gene_pop_found': False, 'gene_coo_found': True,
#  'q_occ': 63, 'gene_occ': 2, 'noisy_entity': False}

# 校准器输入：顺序同 FEATURE_COLUMNS，查不到的流行度按论文口径填 0（与 P2 训练数据一致）
x = lookup.feature_vector(question, answer, confidence=c)    # np.array([c, qpop, gene_pop, gene_coo])

# 只有实体时需显式给出 dataset
lookup.lookup("The Intouchables", "Eric Toledano", dataset="movies")
```

`dataset` 省略时按问题模板推断（`Who is the director of the movie ` / `Who is the performer of the song ` /
`Where is the birthplace of the basketball player `）；推断不出时 `dataset=None`，所有 `*_found=False`。

### 匹配规则（与快照特征的生成代码一致）

规则取自 `calibration_experiment.py` / `build_compact_release_data.py`：

- 问题实体：`question.replace(模板前缀, "")`。qpop 按原大小写精确匹配，因为 Movies 的 "Coma"/"COMA"
  与 Songs 的 "Change"/"chAngE" 是不同问题、流行度不同；共现与出现次数按小写匹配。
- 回答实体（`clean_entity`）：去换行、截掉第一个 "(" 之后的内容；Basketball 总是取第一个逗号之前的部分，
  Movies/Songs 只在长度 ≤ 20 时这样做；最后去掉首尾非单词字符。
- 噪音实体（`noisy_entity`）：`q_occ > 6000`，或在 Movies/Songs 上 `gene_occ > 6000`（缺失按 0）。
  论文训练时剔除这类样本，对它们的校准结果不可靠。

官方仓库较新的 `code/prepare_before_analysis/build_external_popularity.py` 使用另一套规范化（casefold、
不按逗号截断、换行替换为空格）；如果用它重建查找表，需要重新跑 `python -m calibrator.features` 校验。

### 缺失值与真实的 0

- 表中没有的值返回 `None`，对应的 `*_found=False`；找到的 0 是真实的 0，`*_found=True`。
- `gene_coo_found` 表示该 (问题, 回答) 对在共现表中。官方代码在问题实体存在但实体对缺失时记为 0；
  这里先返回 `None`，由 `feature_vector` 统一填 0。
- 回答清洗后为空（空回答、纯标点）时视为没有回答实体，`gene_*` 均为未找到。

### 覆盖范围

查找表只包含论文六个模型在三个数据集上实际产生过的 (问题, 回答) 对。任意用户问题，或模型给出表中
没出现过的回答，一般都会返回未找到；这时特征按 0 填充，校准结果的可信度会下降。P3 应把 `*_found`
与 `noisy_entity` 一并返回给调用方。不做模糊实体链接，也没有实现 docs/01 中"表外实体用模型自生成流行度兜底"的方案（需要运行中的模型）。

### 与官方快照的一致性

`python -m calibrator.features` 对 158,580 条记录逐条用原始问题与回答重新查找，并与快照中的特征比较：

| 字段 | 结果 |
|---|---|
| `qpop`、`gene_coo`、`q_occ`、`gene_occ` | 158,580 条全部一致，无不一致 |
| `gene_pop` / `gene_pop_found` | 158,365 条一致（151,010 条值相同，7,355 条两边都是查不到、快照存 0）；215 条不同 |
| 结论 | 158,365 / 158,580 条完全一致 |

215 条差异全部是清洗后实体为空的回答：qwen2-7b 有 214 条空回答，qwen2.5-7b Movies 有 1 条回答为 `"/>."`。
原始流行度表把空字符串映射为 6 个 sitelinks，快照因此存了 `gene_pop=6, found=true`；查找器把这种情况
视为没有实体。在 P2 的训练数据中，前 214 条已被空回答过滤掉，只剩 qwen2.5-7b Movies #7717 一条会受影响：
训练特征 gene_pop=6，而查找器给出 0。

注意：官方仓库的 `res/` 由 `materialize_compact_data.py` 从快照反推生成，所以这里的比对验证的是
规范化与匹配规则，而不是 Wikipedia/Wikidata 计数本身。测试中另有端到端检查：对 Qwen2.5-7B 三个数据集的
测试集，`feature_vector` 与 `load_split` 的 `X_test` 逐行一致（只有上面那一条例外）。
