# 逐步开发指南（五人分工）

## 分工总表

| 编号 | 角色 | 负责目录 | 关键交付物 |
|---|---|---|---|
| P1 | 数据工程与复现 | `reference/`、`calibrator/data_prep`、`calibrator/features` | 三份 QA 数据、流行度离线表、复现统计 |
| P2 | 校准器训练 | `calibrator/training`、`calibrator/eval` | MLP 权重、基线对比、阈值 |
| P3 | 服务与模型运行时 | `mcp_server/`、`model_runtime/vllm` | vLLM 端点、MCP calibrate 工具 |
| P4 | Agent 集成 | `agent_integration/openhands` | completion 补丁、answer-only、wrapper、检索兜底 |
| P5 | 评测与工程化 | `evaluation/`、`scripts/`、`docs/` | 端到端评测、文档、CI、集成 demo |

## 里程碑总览

| 阶段 | 内容 | 负责人 | 依赖 |
|---|---|---|---|
| M0 | 环境与仓库就绪 | 全体 | 无 |
| M1 | 数据就绪与论文复现 | P1 | M0 |
| M2 | 校准器训练与评测 | P2 | M1 |
| M3 | MCP 服务与 vLLM | P3 | M0（vLLM 可并行） |
| M4 | OpenHands Agent 集成 | P4 | M2、M3 |
| M5 | 端到端评测与交付 | P5 | M4 |

## M0 环境与仓库就绪（全体，第 1 天）

1. 每人 clone 本项目仓库并进入各自目录。
2. P1 单独克隆论文仓库到 `reference/Popular-but-Wrong`：
   `git clone --depth 1 https://github.com/Trustworthy-Information-Access/Popular-but-Wrong.git reference/Popular-but-Wrong`。
3. 建立 Python 3.10 虚拟环境，安装论文仓库的 `requirements-core.txt`。
4. 运行 `python scripts/materialize_compact_data.py` 与 `python scripts/verify_artifacts.py --level full` 验证数据完整。

验收：`Artifact verification passed for level=full.`

## M1 数据就绪与论文复现（P1，第 1–3 天）

1. 确认 `reference/Popular-but-Wrong/data/` 下三份 QA 数据（movies 10,964、songs 2,157、basketball 13,309）。
2. 在 `calibrator/data_prep/` 编写数据加载与切分脚本：50/50 分层切分、10% 验证集，复现论文口径。
3. 在 `calibrator/features/` 构建离线流行度表：entity → sitelinks、实体对 → 共现频次；实现实体链接与表外兜底。
4. 运行 `python scripts/reproduce.py --stage core`，复现流行度/置信度/正确性的主统计。

产出：数据加载脚本、流行度离线表、复现统计报告。
验收：三份数据行数与论文一致，core 统计在论文容差内。

## M2 校准器训练与评测（P2，第 3–5 天）

1. 在 `calibrator/training/` 复现 MLP(128-64-32)，输入 c + PopQ + PopGe + RPopGe，BCE 损失，早停。
2. 复现基线 Prob / Consis / Verbal / Temp / Platt / Isotonic，并做 MLP(c) 与 MLP(c+RPopGe) 消融。
3. 在 `calibrator/eval/` 计算 ECE、Brier、AUROC、Alignment、Confw，并扫描 answer/retrieve 阈值。
4. 按 (model, domain) 导出权重到 `mcp_server/` 约定的位置。

产出：训练脚本、校准器权重、指标表、阈值配置。
验收：Alignment 与 ECE 复现到论文表 5 的容差内，权重可被 MCP 加载。

## M3 MCP 服务与 vLLM（P3，第 2–4 天，可与 M1/M2 并行）

1. 在 `model_runtime/vllm/` 写启动脚本：本地起 Qwen2.5-7B-Instruct，开启 logprobs。
2. 验证 `choices[0].logprobs.content` 可取，关闭思考，固定贪心解码。
3. 在 `mcp_server/` 实现 calibrate(question, answer, c) 工具：实体链接 → 查表 → 组装特征 → 选权重 → MLP 推理 → 返回概率与决策。
4. 设计降级策略：缺权重、缺特征、模型不匹配时返回 unknown 或原始 c，不阻塞 agent。

产出：vLLM 启动脚本、MCP server 实现、工具 schema。
验收：MCP 工具对样例输入返回校准概率与 decision，错误场景有降级返回。

## M4 OpenHands Agent 集成（P4，第 4–6 天）

1. patch OpenHands 的 LLM completion，传 logprobs=True 并捕获响应。
2. 实现 answer-only 生成：约束模型只输出答案，关思考。
3. 计算 c 并存入事件流/状态；实现 wrapper 工具读取 c 后调用 MCP calibrate。
4. 按 decision 实现直接回答与联网搜索（离线用本地 RAG）。

产出：completion 补丁、answer-only 工具、wrapper 工具、检索兜底。
验收：agent 能完成"生成 → 取 logprobs → 算 c → 校准 → 决策"闭环。

## M5 端到端评测与交付（P5，第 6–7 天）

1. 在 `evaluation/` 对比开/关校准器的准确率、拒答率、检索率，画覆盖率-准确率曲线。
2. 整理 `docs/` 与 README，补全复现说明与架构图。
3. 在 `scripts/` 加一键复现与 CI 门禁。
4. 录制端到端 demo。

产出：评测报告、覆盖率-准确率曲线、完整文档、demo。
验收：校准后准确率或覆盖-准确率曲线上有可量化的正向收益，仓库可一键复现。

## 集成顺序与并行安排

- P1 与 P3 可并行：P3 先做 vLLM 端点联通，不等校准器权重。
- P2 依赖 P1 的特征与数据切分。
- P4 依赖 P2 的权重与 P3 的 MCP 接口。
- P5 在 M4 闭环后收口。

## 常见阻塞点

- logprobs 字段路径或思考模式未关，导致 c 失真。
- 训练与部署模型或解码不一致，导致校准失效。
- 实体链接重名或表覆盖不全，导致流行度特征错误。
- 联网搜索在离线环境不可用，需要本地 RAG 兜底。
