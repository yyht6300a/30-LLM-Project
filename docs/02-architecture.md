# 系统架构

## 组件与职责

| 组件 | 职责 |
|---|---|
| OpenHands Agent | 生成答案、计算 c、调用校准、按决策执行动作 |
| vLLM 端点 | 提供模型生成与逐 token logprobs |
| MCP 校准服务 | 查流行度特征、组装特征、MLP 推理、阈值决策 |
| 离线流行度表 | 提供 PopQ、PopGe、RPopGe |
| 校准器权重 | per (model, domain) 的 MLP |
| 离线复现与训练 | 构造数据与特征、训练 MLP |
| 决策与评测 | 阈值标定、端到端评估 |

## 一次请求的数据流

1. 用户向 agent 提出事实问题。
2. agent 发起 answer-only 生成，请求带 logprobs=True。
3. agent 取 answer 与 logprobs，计算 c，存入会话状态。
4. agent 的 wrapper 工具调用 MCP 的 calibrate(question, answer, c)。
5. MCP 做实体链接，查离线流行度表补齐 PopQ、PopGe、RPopGe。
6. MCP 组装特征向量，按 model 与 domain 选权重，MLP 输出 P(正确)。
7. MCP 按标定阈值给出 answer 或 retrieve 决策。
8. agent 执行决策：直接回答，或联网搜索（离线用本地 RAG）后重答。

## logprobs 通道

校准器的第一输入特征 c 必须由 logprobs 计算得到，因此需要一条从模型到校准器的概率通道。
vLLM 的 OpenAI 兼容接口支持 logprobs；OpenHands 侧需要 patch LLM completion 以传
logprobs=True 并捕获响应，再把 c 存进事件流或运行时状态供 wrapper 工具读取。
