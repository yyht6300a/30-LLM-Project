# 项目概述

## 目标

在离线环境中，用中等规模开源模型复现论文的置信度校准器，并把校准能力以 MCP 工具的形式
外挂给 OpenHands agent。agent 对事实问题先生成初步答案并计算模型原始置信度 c，再将问题、
答案与 c 交给 MCP 校准服务；服务用离线流行度表和已训练的多层感知机输出校准后的正确概率，
并按阈值给出回答或检索的建议。

## 背景

论文《Popular but Wrong》发现 LLM 的过度自信与知识流行度相关：幻觉答案往往比标准答案更
流行、与问题实体共现更频繁，且这类答案获得更高的模型置信度。论文用三层 MLP，把原始置信度
c 与三个流行度特征（PopQ、PopGe、RPopGe）映射为校准后的正确概率，显著降低 ECE 并提升
Alignment。

## 已确定的关键决策

- 集成路线：route B（agent 生成答案并算 c，server 负责校准）。
- 生成方式：answer-only 专用生成，保证 logprobs 与答案同属一次调用。
- 置信度来源：vLLM 返回的逐 token logprobs，c = mean(exp(logprob_i))。
- 流行度来源：离线预计算表（Wikidata sitelinks + Wikipedia 共现）。表外实体按论文口径填 0，
  并在 `feature_coverage` 中标记；原计划的"模型自生成兜底"未实现（见 `calibrator/README.md`）。
- 校准器粒度：按 (model, domain) 保存权重。
- 模型：主选 Qwen2.5-7B-Instruct，vLLM 本地 OpenAI 兼容端点，贪心解码。
- 纠偏动作：answer / retrieve 单阈值，阈值在训练半区上按 Alignment 最大化标定（论文附录 B.2），
  随权重导出；无法校准时返回 unknown，由 agent 转检索。
