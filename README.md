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

## 快速开始

完整的分步开发计划见 [docs/03-development-guide.md](docs/03-development-guide.md)。
数据来源与复现方法见 [docs/04-data-reproduction.md](docs/04-data-reproduction.md)。
