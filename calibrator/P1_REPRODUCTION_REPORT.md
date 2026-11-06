# P1 复现报告：数据、流行度统计与 RQ1/RQ2

论文：《Popular but Wrong》（arXiv 2505.17537v2）。官方仓库 commit `1b1f250`，本地路径
`reference/Popular-but-Wrong`。本报告只覆盖 P1 范围（数据与统计复现）；校准实验（论文表 5–7、
附录表 14–16）由 P2 负责。

## 1. 复现范围与结果

| 步骤 | 命令（在官方仓库内） | 结果 |
|---|---|---|
| 制品完整性 | `python scripts/verify_artifacts.py --level full` | `Artifact verification passed for level=full.` |
| RQ1/RQ2 统计 | `python scripts/reproduce.py --stage core` | 日志 `reproduction_logs/rq1_rq2_statistics.log` |
| Bootstrap 置信区间 | 同上（`bootstrap_ci.py`，1000 次重采样，seed 42） | `bootstrap_results.json` 与官方提交版本逐字节一致 |
| 数值检查点 | `python scripts/check_expected_results.py` | 全部在容差内 |

`check_expected_results.py` 中的校准/分布/迁移检查读取的是官方仓库自带的结果 JSON，P1 没有重跑
这些实验（需要 torch，属于 P2）。

## 2. 数据集与样本数

| 数据集 | 关系 | 问题数（论文表 1） | 复现 |
|---|---|---:|---|
| Movies | movie → director | 10,964 | 一致 |
| Songs | song → performer | 2,157 | 一致 |
| Basketball | player → birthplace | 13,309 | 一致 |

六个模型各 26,430 条回答，合计 158,580 条。不同分析使用不同的过滤口径，样本数不同：

| 口径 | 规则 | 例：Qwen2.5-7B Movies |
|---|---|---:|
| 校准（`calibrator.data_prep`，论文表 5） | 去空回答、去 single occurrence > 6000 的歧义实体，流行度查不到记为 0 | 9,578 |
| RQ 统计（表 4、8、9、10） | 在上面基础上，再去掉标准答案实体为噪音、实体不在共现/出现表中、`gt_pop` 或 `gene_pop` 查不到的样本 | 8,195 |
| RQ 描述统计（表 3、图 5） | 与 RQ 统计相同，但查不到的流行度记为 0，不剔除 | — |

论文 RQ 分析不包括 Basketball 上的 Qwen2.5 系列（准确率 4–8%，见表 9）。

## 3. 准确率与置信度（论文表 9）

18 个 (dataset, model) 组合的 Acc 与 Conf 全部与论文一致（三位小数）。示例：

| 数据集 | 指标 | Llama-3-8B | Qwen2-7B | ChatGPT | Qwen2.5-7B | Qwen2.5-14B | Qwen2.5-32B |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | Acc | 0.715 | 0.401 | 0.942 | 0.210 | 0.469 | 0.437 |
| Movies | Conf | 0.904 | 0.817 | 0.986 | 0.754 | 0.877 | 0.877 |
| Basketball | Acc | 0.134 | 0.098 | 0.350 | 0.042 | 0.074 | 0.078 |

所有模型都过度自信：Conf 远高于 Acc，在 Basketball 上最严重。

## 4. 知识流行度统计（论文表 3）

核心阶段日志不输出表 3。这里按 RQ 描述统计口径从压缩快照重新计算（样本数与日志一致）：

| 指标 | Movies | Songs | Basketball | 与论文 |
|---|---:|---:|---:|---|
| 答对：PopQ / PopGT / RPopGT | 22 / 28 / 9 | 10 / 64 / 17 | 9 / 132 / 3 | 一致 |
| 答错（标准答案）：PopQ / PopGT / RPopGT | 18 / 21 / 5 | 7 / 31.5 / 7 | 7 / 75 / 1 | 一致 |
| 答错（生成答案）：PopGe / RPopGe | 5 / 2 | 39 / 4.5 | 186 / 1 | 一致 |
| PopGe CV | 1.58 | 0.90 | 0.58 | 一致 |
| Answer Unique Ratio（按清洗后实体去重） | 0.789 | 0.821 | 0.142 | 论文 0.787 / 0.821 / 0.133 |

## 5. 流行度与正确性（RQ1）

- **偏相关（论文表 8）**：15 个组合（不含 Basketball 上的 Qwen2.5）全部一致。
  其中 10 个组合以 RPopGT 为最强信号（包括 Songs 的全部 6 个）。例外有两类：Movies 上的 ChatGPT 与 Qwen2.5-7B
  是 PopQ 略高；Basketball 上的 Llama-3-8B 与 Qwen2-7B 是 PopGT 最强。在 Movies 上，控制其余两个信号后
  PopGT 为 −0.041 ~ −0.005。
- **Bootstrap**：gt_coo→acc 的 Spearman 95% CI 全部不含 0（如 Qwen2.5-7B Movies +0.488 [+0.471, +0.502]）。

## 6. 流行度与置信度（RQ2）

- **偏相关（论文表 4）**：Total / Correct / Incorrect 三列，对 RPopGe、PopQ、PopGe 全部一致。
  示例 Qwen2.5-7B Movies：RPopGe +0.419 / +0.178 / +0.195，PopGe +0.239 / +0.156 / +0.310。
- **错误答案更"流行"（论文表 10）**：日志覆盖的组合中，PopGe > PopGT 与 RPopGe > RPopGT 两列全部一致，
  例如 Movies 上 PopGe > PopGT 占错误答案的 44.9%–64.2%，Basketball 上为 68.7%–78.2%。
- **置信度（论文图 5）**：ChatGPT 与 Qwen2.5-7B 答对/答错的置信度均值 ± 标准差（共 24 个数）全部一致，
  如 Qwen2.5-7B Movies 答对 90.9 ± 8.6、答错 69.6 ± 14.3。

## 7. 主要结论（复现确认）

1. 问题与标准答案的共现（RPopGT）比单实体流行度更能预测答对。
2. 模型答错时倾向生成更流行的实体，或与问题实体共现更多的实体；知识越少的领域（Basketball）越明显。
3. 在 Movies 与 Songs 上，置信度主要由问题与生成答案的共现（RPopGe）驱动（Total 列 +0.172 ~ +0.609）；
   Basketball 上 PopQ 更强。在 Movies 上，错误样本的 PopGe→conf 偏相关是正确样本的 1.7–4.9 倍（ChatGPT 除外），
   即模型对"熟悉"的错误答案更自信；Songs 与 Basketball 上这一放大效应不稳定。
   这些信号与正确性的关系不同于它们与置信度的关系，这正是用 PopQ、PopGe、RPopGe 校准置信度的依据。

## 8. 差异与局限

- **表 3 Answer Unique Ratio**：Movies 0.789 对 0.787、Basketball 0.142 对 0.133，其余数值完全一致。
  试过原始回答、清洗后的实体、大小写归一三种去重方式均未消除差异，原因未定，不影响其他结论。
- **口径不统一**：论文表 3 与图 5 把查不到的流行度记为 0，而表 4、8、9、10 剔除这些样本。这是论文本身的口径差异，不是数据问题。
- **未覆盖的条目**：核心日志没有表 10 的 Either / Pop only 两列、Basketball 上的 Qwen2.5 行，也没有图 4 与图 6（需运行 `--stage figures`）。
- **查找表来源**：官方仓库的 `res/` 查找表由 `materialize_compact_data.py` 从压缩快照反推生成，并非原始统计表。
  `calibrator.features` 与快照的一致性校验因此只能验证实体规范化与匹配规则，无法独立验证 Wikipedia/Wikidata 计数本身。
- **空实体伪影**：原始流行度表把空字符串映射为 6 个 sitelinks。快照中 215 条清洗后实体为空的回答因此带有
  `gene_pop=6, found=true`。离线查找把这种情况视为"无实体"，详见 `calibrator/README.md`。
