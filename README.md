# 30-LLM-Project：知识流行度置信度校准器的 Agent 集成

本项目复现 EMNLP2026 论文《Popular but Wrong: Understanding and Mitigating LLM
Overconfidence through Knowledge Popularity》的置信度校准器，并将其封装为 MCP 服务接入
OpenHands agent，使 agent 在事实问答场景中先取得初步答案与模型置信度，经校准器纠偏后，
再决定直接回答还是转为联网检索。

## 仓库结构

```text
30-LLM-Project/
├── docs/                    # 设计文档与开发指南
├── reference/               # 论文官方仓库（单独克隆，已 gitignore）
├── calibrator/              # 数据准备、特征、训练、离线评测
├── mcp_server/              # 校准器 MCP 服务
├── agent_integration/       # OpenHands 集成（补丁与工具）
├── model_runtime/           # vLLM 部署与 logprobs 通道
├── evaluation/              # 端到端评测
└── scripts/                 # 公共脚本与 CI
```

## 五人分工

| 编号 | 角色 | 负责目录 |
|---|---|---|
| P1 | 数据工程与复现 | `reference/`、`calibrator/data_prep`、`calibrator/features` |
| P2 | 校准器训练 | `calibrator/training`、`calibrator/eval` |
| P3 | 服务与模型运行时 | `mcp_server/`、`model_runtime/vllm` |
| P4 | Agent 集成 | `agent_integration/openhands` |
| P5 | 评测与工程化 | `evaluation/`、`scripts/`、`docs/`、git/CI |

## Agent 集成（P4）

在 OpenHands agent 侧实现了两个组件，通过 SDK 工具注册表（`register_tool` +
`Tool(name, params)`）接入；agent 实际只暴露 `calibrated_qa` 一个工具：

- `calibrated_qa`：对事实问题做 answer-only 生成，取 logprobs 算原始置信度 c，经
  stdio MCP 调 `calibrate` 得到 `answer / retrieve / unknown` 决策；当决策为
  `retrieve` / `unknown` 时，工具内部直接做知识库检索，把检索到的标准答案一并返回。
- `search_qa`：离线知识库检索组件（`KnowledgeBase`），被 `calibrated_qa` 在低置信
  时内部复用；同时以独立工具形式注册备用，但不暴露给 agent。

实现见 [agent_integration/calibrated_qa.py](agent_integration/calibrated_qa.py)、
[agent_integration/search_tool.py](agent_integration/search_tool.py)、
[agent_integration/run_calibrated_agent.py](agent_integration/run_calibrated_agent.py)；
架构、接口与实现原理详见 [docs/05-agent-tools-technical-doc.md](docs/05-agent-tools-technical-doc.md)。

## 冒烟测试

冒烟脚本 [agent_integration/smoke_test.py](agent_integration/smoke_test.py) 按层验证：
可行性预检 → SDK 导入 → 数据索引 → 决策策略 → vLLM+MCP → 工具注册 → 完整 e2e。

最终结果（`Qwen/Qwen2.5-7B-Instruct-AWQ` 4-bit 量化模型）：
0–6 全部 PASS；e2e 中 agent 经 `calibrated_qa` 判 `retrieve` 后由其内部检索并
正确回答。完整过程与逐条修复记录见
[agent_integration/smoke_run_log.md](agent_integration/smoke_run_log.md)。

## 本地运行（量化模型）

在带 GPU 的 Linux 环境，先按
[model_runtime/vllm/start_qwen25_7b_quantized.sh](model_runtime/vllm/start_qwen25_7b_quantized.sh)
启动 vLLM 量化服务（需准备装有 vllm 的 Python 环境并设置 `P4_ENV`），再跑冒烟：

```bash
export PYTHONPATH="$PWD"
export M3_MCP_PYTHON="<装有 mcp/numpy/pandas/scipy/sklearn 的 python>"
export POPULAR_BUT_WRONG_REPO="$PWD/reference/Popular-but-Wrong"
export M3_AGENT_LLM_MODEL="openai/qwen2.5-7b"
OPENHANDS_SUPPRESS_BANNER=1 \
python agent_integration/smoke_test.py
```

## 端到端评测（P5）

Qwen2.5-7B-Instruct（bf16，vLLM）在 P2 测试集 2,460 题上，开/关校准器 × 开/关检索兜底。
三个数据集上，开校准器都同时提高准确率、降低拒答率与检索率（检索关时的准确率）：

| 数据集 | 准确率 raw → calibrated | 拒答率 raw → calibrated |
|---|---|---|
| Movies（1,749） | 0.354 → 0.434 | 0.572 → 0.544 |
| Songs（148） | 0.351 → 0.392 | 0.622 → 0.588 |
| Basketball（563） | 0.181 → 0.242 | 0.702 → 0.661 |

完整指标、覆盖率-准确率曲线与注意事项（检索开时知识库检索返回标准答案，准确率为理想检索上限）
见 [evaluation/README.md](evaluation/README.md)。

## 复现

Linux / TC2，Python ≥ 3.12（TC2 先 `module load python312`），在仓库根目录：

```bash
bash scripts/reproduce.sh test   # 建环境、拉论文仓库并生成查找表、跑全部 CPU 测试（即 CI 门禁）
bash scripts/reproduce.sh        # 再复现 P1 论文统计、P2 校准器评测与权重、离线覆盖率-准确率曲线
bash scripts/reproduce.sh gpu    # TC2 登录节点：提交端到端消融作业，结果写入 evaluation/output/
```

端到端 demo：在 GPU 节点运行 `bash -l scripts/demo.sh`。各脚本说明见 [scripts/README.md](scripts/README.md)。

## 文档

- 分步开发计划：[docs/03-development-guide.md](docs/03-development-guide.md)
- 系统架构与架构图：[docs/02-architecture.md](docs/02-architecture.md)
- 数据来源与复现方法：[docs/04-data-reproduction.md](docs/04-data-reproduction.md)
- Agent 双工具技术文档：[docs/05-agent-tools-technical-doc.md](docs/05-agent-tools-technical-doc.md)
- 端到端评测与工程化技术文档（P5）：[docs/06-evaluation-technical-doc.md](docs/06-evaluation-technical-doc.md)
- MCP 服务与 vLLM：[mcp_server/README.md](mcp_server/README.md)
