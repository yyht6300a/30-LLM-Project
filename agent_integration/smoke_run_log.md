# 冒烟测试记录

结论：SDK 与依赖问题已全部解决，冒烟 0–6 全部通过；校准侧（MCP + MLP）与完整 e2e
（生成 → 算 c → 校准 → 决策 → 检索/回答）均验证可用，代码架构方案可行。

## 冒烟分层

`smoke_test.py` 按层验证，失败即可定位到具体层：

```text
0_preflight     可行性预检（SDK / 数据 / vLLM / MCP 就绪）
1_imports       openhands.sdk 与 P4 模块导入
2_data          本地知识库加载与检索
3_decision      DecisionPolicy 三路分支
4_vllm_mcp      vLLM 生成 + logprobs 算 c + MCP 校准
5_registration  工具注册
6_e2e           完整会话闭环
```

## 过程中解决的问题

1. `openhands.sdk` 不可导入 → 安装 openhands-sdk 及 anyio / mcp / openai / numpy /
   pandas / scipy / scikit-learn。
2. 运行时工具注册被忽略 → 改用 `register_tool` + `Agent(tools=[Tool(name, params)])`
   的官方注册方式。
3. 主 agent LLM 模型名缺 provider 前缀 → 新增 `M3_AGENT_LLM_MODEL`
   （默认 `openai/qwen2.5-7b`），vLLM served 名与 artifact slug 仍用裸名。
4. vLLM Triton JIT 缺 C 编译器 → 提供 gcc（可用户态安装，免 sudo）。
5. vLLM flashinfer 缺 nvcc → 禁用 flashinfer 采样，回退 PyTorch 采样。
6. 显存受限 → 降低 `max_model_len`、调 `gpu_memory_utilization`、启用 eager 模式。
7. tool calling → vLLM 开启 `--enable-auto-tool-choice --tool-call-parser hermes`。
8. agent 在 decision=retrieve 时漏调 `search_qa` → 把知识库检索折叠进 `calibrated_qa`
   （低置信时工具内部检索并返回标准答案），agent 只需原样引用，不再依赖 LLM 自行调用。

## 最终结果

| 阶段 | 结果 |
|---|---|
| 0_preflight | PASS |
| 1_imports | PASS |
| 2_data | PASS（26,428 条） |
| 3_decision | PASS |
| 4_vllm_mcp | PASS（生成答案 → 算 c → 校准 → decision） |
| 5_registration | PASS（agent 工具 `calibrated_qa` / `finish`；`search_qa` 注册但未暴露） |
| 6_e2e | PASS（会话 FINISHED，agent 判 retrieve 后由 calibrated_qa 内部检索并正确回答） |

## 结论

端到端链路（主 agent 接 vLLM → 生成 + logprobs 算 c → MCP calibrate → 决策 →
检索/回答）在量化模型上完整跑通。

注意：量化会带来校准偏移——AWQ 4bit 的 c 与 bf16 不同，演示链路可用，严格复现
论文数值仍需 bf16 或对量化模型重新校准。
