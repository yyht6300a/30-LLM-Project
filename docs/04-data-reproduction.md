# 数据来源与复现

## 官方仓库

论文官方仓库已开源全部数据与脚本：
`https://github.com/Trustworthy-Information-Access/Popular-but-Wrong`

## 三层数据

1. 源 QA 数据集：`data/movies.jsonl`（10,964）、`data/songs.jsonl`（2,157）、
   `data/basketball.jsonl`（13,309）。每行含 question、reference、popularity、question_entity。
2. 模型输出快照：`data/model_outputs/*.jsonl.gz`，六个模型各 26,430 行，已含 confidence、
   correct 标签与全部流行度字段，可用于纯 CPU 训练校准器。
3. 从零重建脚本：模型响应见 `qa_generation/`，流行度特征见
   `docs/BUILD_POPULARITY_FEATURES.md`（Wikidata sitelinks + 英文 Wikipedia 20231101 快照）。

## 如何应用

- 训练校准器：直接用 `data/model_outputs/qwen2.5-7b.jsonl.gz` 等快照，特征为
  confidence + qpop/gene_pop/gene_coo，标签为 correct。
- 在线部署：用 `qa_generation/` 的推理方式让 vLLM 带 logprobs 重生成，保持训练与部署
  同一模型、同一解码。
- 离线流行度表：按 `docs/BUILD_POPULARITY_FEATURES.md` 的字段与匹配规则构建。

## 字段与论文记号对照

| 仓库字段 | 论文记号 |
|---|---|
| confidence | c |
| qpop / gt_pop / gene_pop | PopQ / PopGT / PopGe |
| gt_coo / gene_coo | RPopGT / RPopGe |
| llm_qpop / llm_gene_pop / llm_coo | dPopQ / dPopGe / dRPopGe |
