# agent_integration：OpenHands Agent 集成

所有者：P4。

本目录在 OpenHands SDK 上实现事实问答的校准闭环：agent 收到问题后调用
`calibrated_qa`，该工具完成"生成 → 算 c → 校准 → 决策"，并在 decision 为
`retrieve` / `unknown` 时内部做离线知识库检索、把标准答案一并返回。agent 只需
原样引用返回的 final answer，不依赖其自行调用检索工具。

## 目录内容

```text
agent_integration/
├── calibrated_qa.py          # CalibratedQATool：生成 + 算 c + MCP 校准 + 决策 + 内部检索
├── search_tool.py            # SearchTool / KnowledgeBase：离线知识库检索（被 calibrated_qa 复用）
├── run_calibrated_agent.py   # 入口：LLM + 工具注册 + system prompt + 会话
├── smoke_test.py             # 分层冒烟测试（0_preflight ~ 6_e2e）
├── smoke_run_log.md          # 冒烟运行与问题修复记录
└── openhands/                # OpenHands SDK 克隆（已 gitignore）
```

## 两个组件

| 组件 | 功能 | 文件 |
|---|---|---|
| `calibrated_qa` | answer-only 生成 → logprobs 算 c → stdio MCP `calibrate` → 决策；低置信时内部检索并返回标准答案 | [calibrated_qa.py](calibrated_qa.py) |
| `search_qa` | 论文三份源数据做离线知识库（`KnowledgeBase`），精确匹配 + 领域关键词门 + 最长实体兜底；被 calibrated_qa 内部复用，独立注册备用 | [search_tool.py](search_tool.py) |

详细架构、实现原理与接口字段见 [docs/05-agent-tools-technical-doc.md](../docs/05-agent-tools-technical-doc.md)。

## 注册机制

组件通过 SDK 工具注册表接入，在模块导入时 `register_tool`；agent 只暴露
`calibrated_qa` 一个工具（`search_qa` 注册但不加入 agent 工具列表）：

```python
register_tool("calibrated_qa", CalibratedQATool)
register_tool("search_qa", SearchTool)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name="calibrated_qa", params={"data_dir": ..., ...}),
    ],
    system_prompt=AGENT_INSTRUCTIONS,
)
```

注意：不要用 `agent.add_runtime_tools(...)` 在构造后立即注册——OpenHands 会因 agent
尚未初始化而忽略它（会打印 warning）。注册表 + `Tool` spec 是官方路径。

## 入口与配置

运行入口见 [run_calibrated_agent.py](run_calibrated_agent.py)。环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `M3_VLLM_BASE_URL` | `http://127.0.0.1:8000/v1` | vLLM OpenAI 兼容端点 |
| `M3_MODEL` | `qwen2.5-7b` | vLLM served 名，也是校准 artifact 的 slug |
| `M3_AGENT_LLM_MODEL` | `openai/qwen2.5-7b` | 主 agent 的 LiteLLM 限定模型名 |
| `M3_MCP_PYTHON` | `sys.executable` | 启动 stdio MCP server 的 Python |
| `POPULAR_BUT_WRONG_REPO` | `reference/Popular-but-Wrong` | 论文仓库路径（流行度表） |
| `M3_DECISION_MODE` | `calibrated` | `calibrated` 或 `raw`（消融开关） |
| `M3_RAW_THRESHOLD` | `0.5` | raw 模式的决策阈值 |

`calibrated_qa` 还接受 `data_dir`（知识库路径，默认
`reference/Popular-but-Wrong/data`）与 `retrieval_enabled`（默认 `True`，消融
"检索兜底开/关"用）。

## 冒烟测试

[smoke_test.py](smoke_test.py) 按层验证：

```text
0_preflight → 1_imports → 2_data → 3_decision → 4_vllm_mcp → 5_registration → 6_e2e
```

最终结果（Qwen2.5-7B-Instruct-AWQ 4-bit 量化模型）全部 PASS，
过程与修复记录见 [smoke_run_log.md](smoke_run_log.md)。

## 下一步

- 消融：`M3_DECISION_MODE` 切 calibrated/raw，检索兜底开关，特征子集与量化精度对比。
- 端到端评测：批量跑 `data/*.jsonl` 子集，统计准确率、拒答率、检索率、覆盖率-准确率曲线。
