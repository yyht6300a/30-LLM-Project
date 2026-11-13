# Agent 侧校准问答技术文档

本文描述 P4 在 OpenHands agent 侧实现的 `calibrated_qa` 工具，以及它内部复用的
`search_qa`（`KnowledgeBase`）知识库组件的架构、实现原理、接口形式，汇总冒烟测试
结果，并给出下一步计划（含消融实验方案）。

## 1. 总体架构

两个组件都以 OpenHands SDK 的 `ToolDefinition` 体系实现并注册进工具注册表；agent
实际只暴露 `calibrated_qa`，由主 LLM 通过函数调用触发，低置信时的知识库检索在工具
内部完成。

```text
用户问题
  │ conversation.send_message(question)
  ▼
OpenHands Agent（主 LLM，LiteLLM → vLLM 的 qwen2.5-7b）
  │  调用 calibrated_qa
  ▼
calibrated_qa：生成答案 + 算 c + 调 MCP 校准 + 决策
  └─ 低置信 / unknown 时 → 内部调 KnowledgeBase 检索，返回标准答案
```

两个工具共用的注册机制：

```python
from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool import Tool

register_tool("calibrated_qa", CalibratedQATool)
register_tool("search_qa", SearchTool)   # 注册备用，不加入 agent 工具列表

agent = Agent(
    llm=llm,
    tools=[
        Tool(name="calibrated_qa", params={"data_dir": ..., ...}),
    ],
    system_prompt=AGENT_INSTRUCTIONS,
)
```

说明：OpenHands 的 `add_runtime_tools` 必须在 agent 初始化之后调用，否则会被忽略；
因此采用注册表 + `Tool(name=..., params=...)` 的官方方式，由 `_initialize()` 阶段
通过 `resolve_tool` 调用 `CalibratedQATool.create(**params)` 完成实例化。

## 2. calibrated_qa 工具

文件：[agent_integration/calibrated_qa.py](../agent_integration/calibrated_qa.py)

### 2.1 功能

对实体中心事实问题（电影导演 / 歌曲演唱者 / 篮球球员出生地）生成答案，并返回校准
后的正确概率与决策。它把"取 logprobs → 算 c → 校准 → 低置信时检索"全部封闭在工具
内部，agent 不需要接触 token 概率，也不需要在低置信时自行调用检索工具。

### 2.2 分层实现原理

执行器 `CalibratedQAExecutor` 分五层，每层可独立替换，服务于消融实验：

```text
Layer 0  calculate_confidence    纯函数：logprobs -> c，无 I/O
Layer 1  generate_answer         vLLM answer-only 生成，独占 logprobs
Layer 2  CalibrationClient       stdio MCP 调 calibrate 工具
Layer 3  DecisionPolicy          纯函数：概率/置信度 -> answer|retrieve|unknown
Layer 4  KnowledgeBase           低置信时：本地知识库检索标准答案
```

Layer 0：c 的计算口径与论文一致。

```python
SPECIAL_TOKENS = {"<|im_end|>", "<|endoftext|>"}

def calculate_confidence(logprob_content):
    answer_tokens = [i for i in logprob_content if i.token not in SPECIAL_TOKENS]
    probabilities = [math.exp(i.logprob) for i in answer_tokens]
    return sum(probabilities) / len(probabilities), answer_tokens
```

即 c 是"答案 token 概率的算术平均"，不是 `exp(mean(logprob))`，并排除特殊停止 token。

Layer 1：answer-only 生成。用 OpenAI 兼容客户端直连 vLLM，system prompt 约束只输出
实体名，`temperature=0`、`max_tokens=32`、`logprobs=True`。返回 `GenerationResult`
（question/answer/raw_confidence/model/answer_token_count）。

Layer 2：`CalibrationClient` 每次调用 spawn 一个 `python -m mcp_server.server` 的
stdio 子进程，经 MCP 协议调用 `calibrate` 工具，拿到结构化校准结果。

Layer 3：`DecisionPolicy.decide(mode, calibrated_probability, threshold,
raw_confidence, raw_threshold)`。`mode="calibrated"` 用校准概率对 artifact 阈值，
`mode="raw"` 用原始 c 对 raw_threshold（无校准基线）。这是消融实验的核心开关。

Layer 4：当 `decision` 为 `retrieve` 或 `unknown` 且 `retrieval_enabled=True` 时，
直接调用 `KnowledgeBase.search(question)`，把检索到的标准答案别名放进返回结果，
并在返回文本里给出明确的 `Final answer:`，agent 只需原样引用。

### 2.3 接口形式

输入 Action：

```text
CalibratedQAAction
  question: str          # 原始事实问题
  domain: str | None     # movies | songs | basketball；省略时服务端自动推断
```

输出 Observation：

| 字段 | 类型 | 含义 |
|---|---|---|
| answer | str \| None | 模型生成的答案 |
| raw_confidence | float \| None | 原始置信度 c |
| calibrated_probability | float \| None | 校准后 P(正确) |
| threshold | float \| None | 决策阈值 |
| decision | str \| None | answer / retrieve / unknown |
| decision_source | str \| None | calibrated / raw（消融可定位） |
| status | str \| None | ok / degraded |
| artifact | str \| None | 实际加载的 `<model>__<domain>` 权重 |
| reason / warnings | str / list | 降级原因与提示 |
| request_id / model / domain | — | 可定位性：请求号、模型、领域 |
| feature_values / feature_coverage | dict | 特征值与命中情况 |
| retrieved_references | list[str] | 低置信时检索到的标准答案别名 |
| retrieved_matched_question | str \| None | 命中的规范问题 |
| retrieval_triggered | bool | 本次是否触发了内部检索 |
| latency_ms | int \| None | 耗时 |

下游 MCP `calibrate` 工具的输入为 `question, answer, confidence, model, domain(可选)`，
返回 `status / calibrated_probability / threshold / decision / artifact / feature_values /
feature_coverage / reason / warnings`。该工具定义在
[mcp_server/server.py](../mcp_server/server.py)，推理链为
`CalibrationService.calibrate` → `PopularityLookup` + `ArtifactStore.predict`。

### 2.4 数据流

```text
agent 调 calibrated_qa(question)
  → generate_answer：vLLM(logprobs=True) → answer + logprobs
  → calculate_confidence：c = mean(exp(logprob_i))
  → CalibrationClient.calibrate：spawn mcp_server.server，调 calibrate
       → 实体链接 → 查流行度表 → 组装 [c, PopQ, PopGe, RPopGe] → MLP → P(正确)
  → DecisionPolicy：按 mode 产生 decision
  → （decision ∈ {retrieve, unknown} 时）KnowledgeBase.search → retrieved_references
  → CalibratedQAObservation（含 content 文本 + 结构化字段）回流 agent
```

## 3. search_qa / KnowledgeBase 组件

文件：[agent_integration/search_tool.py](../agent_integration/search_tool.py)

### 3.1 功能

离线知识库检索：从论文三份源数据中查出标准答案别名。`KnowledgeBase` 是
`calibrated_qa` 在低置信时内部复用的数据层；`SearchTool` 则作为独立工具注册备用，
不加入 agent 的工具列表。

### 3.2 实现原理

`KnowledgeBase` 加载 `movies/songs/basketball.jsonl`，建两级索引：

```text
_by_question:  规范化问题 -> 记录
_by_entity:    领域 -> 规范化实体 -> 记录列表
```

检索分两级：

1. 规范问题精确匹配（主路径）：`小写 + 去首尾空格 + 压缩空白` 后查表。
2. 改写问题兜底：先按领域关键词判定领域，再在该领域内取"出现在问题里的最长实体"，
   实体长度下限 3，用于避免普通词误命中。

```python
DOMAIN_KEYWORDS = {
    "movies": ("movie", "film", "director", "directed"),
    "songs": ("song", "performer", "performed", "singer", "sang"),
    "basketball": ("basketball", "player", "birthplace", "born"),
}
```

### 3.3 接口形式

输入 Action：

```text
SearchAction
  question: str          # 要检索的问题
  domain: str | None     # 可选领域提示
```

输出 Observation：

| 字段 | 类型 | 含义 |
|---|---|---|
| found | bool | 是否命中 |
| references | list[str] | 标准答案别名 |
| matched_question | str \| None | 命中的规范问题 |
| question_entity | str \| None | 问题主体实体 |
| domain | str \| None | 命中领域 |

工具参数 `data_dir` 指向 `reference/Popular-but-Wrong/data`。

## 4. 主 agent 编排

主 agent 的 system prompt（`run_calibrated_agent.py` 中的 `AGENT_INSTRUCTIONS`）规定
事实问题调 `calibrated_qa`，然后原样引用其返回的 final answer；检索已由工具内部完成，
agent 不调用其它搜索工具。非事实任务不调 `calibrated_qa`。

主 agent 的 LLM 模型名使用 LiteLLM 限定形式 `openai/qwen2.5-7b`（`M3_AGENT_LLM_MODEL`），
而 vLLM served 名与校准 artifact 的 slug 保持 `qwen2.5-7b`，二者不可混淆。

## 5. 冒烟测试结果

### 5.1 无 vLLM 环境（逻辑层验证）

0_preflight / 1_imports / 2_data / 3_decision / 5_registration 通过；
4_vllm_mcp 与 6_e2e 因 vLLM 不可达被阻断。此轮验证了 SDK 导入、数据索引、决策策略
与工具注册。

### 5.2 量化模型（最终）

环境：4-bit 量化模型，vLLM 0.33，模型
`Qwen/Qwen2.5-7B-Instruct-AWQ`（4-bit）。

| 阶段 | 结果 |
|---|---|
| 0_preflight | PASS（sdk / data / vllm / mcp_server 全部就绪） |
| 1_imports | PASS |
| 2_data | PASS（26,428 条） |
| 3_decision | PASS |
| 4_vllm_mcp | PASS（c=0.7406 → p=0.8123，decision=answer，artifact=qwen2.5-7b__movies） |
| 5_registration | PASS（agent 工具 `calibrated_qa` / `finish`；`search_qa` 注册但未暴露） |
| 6_e2e | PASS（conversation FINISHED，最终回答 Olivier Nakache 与 Éric Toledano） |

关键执行参数与修复记录见
[agent_integration/smoke_run_log.md](../agent_integration/smoke_run_log.md) 与
[model_runtime/vllm/start_qwen25_7b_quantized.sh](../model_runtime/vllm/start_qwen25_7b_quantized.sh)。

## 6. 下一步计划

### 6.1 消融实验方案

目标：证明校准器与检索兜底各自带来可量化的增益，且增益可归因。

| 消融变量 | 取值 | 实现方式 |
|---|---|---|
| 决策源 | calibrated vs raw | `M3_DECISION_MODE=calibrated` / `raw`，`decision_source` 字段区分 |
| 检索兜底 | 开 vs 关 | `retrieval_enabled=True/False`；关时 retrieve/unknown 直接返回 unknown |
| 特征子集 | full vs c-only vs c+RPopGe | 需要 P2 额外导出 c-only 与 c+RPopGe 的部署 artifact（当前仅导出 full） |
| 模型精度 | bf16 vs AWQ 4bit | 分别部署 bf16 与 AWQ 4bit，对比 c 与校准概率分布 |

评测口径（在线，跑 `data/*.jsonl` 的子集）：准确率、拒答率、检索率、覆盖率-准确率
曲线、端到端 Alignment / AUROC / ECE，并记录每次的 `decision_source`、`feature_values`
与 `feature_coverage`，保证可追溯。

建议消融顺序：raw（基线）→ calibrated（主结果）→ calibrated+检索（完整链路），
最后补特征子集与量化精度两项。

### 6.2 端到端评测与工程收尾

1. 在 `evaluation/` 写批量评测脚本：抽 `data/movies.jsonl` 等子集，开/关校准器与
   检索，输出指标表与覆盖率-准确率曲线。
2. 结果回填 `docs/`，更新 README 与 `03-development-guide.md` 的验收标准。
3. 在 `scripts/` 加一键复现与 CI 门禁（至少跑 0/1/2/3/5 这些纯逻辑阶段）。
4. 录制端到端 demo，作为课程项目交付物。

### 6.3 已知风险

- 量化偏移：AWQ 4bit 下 c 与 bf16 不同（示例 0.74 vs 论文约 0.895），严格复现需
  bf16 或重新校准。
- 显存受限时需降低 `max_model_len` 与 `gpu_memory_utilization`，超长问题可能截断。
- 实体链接与离线表覆盖有限，域外问题会返回 `unknown`，需检索或拒答兜底。
