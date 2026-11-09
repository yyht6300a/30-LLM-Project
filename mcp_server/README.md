# mcp_server：校准器 MCP 服务

所有者：P3。

# M3：vLLM 与置信度校准 MCP 服务

本文说明 M3 模块的安装、启动、调用和验收方式。

系统流程：

```text
问题 -> vLLM 生成答案和 token logprobs
     -> 计算原始置信度 c
     -> MCP calibrate 工具
     -> 查询流行度特征并运行 MLP
     -> 返回校准概率和 answer/retrieve/unknown 决策
```

## 1. 环境要求

已验证的集群环境：

- Slurm 分区：`MGPU-TC2`
- GPU：NVIDIA A40 46 GB
- CUDA：12.8
- Python：3.12
- 模型：`Qwen/Qwen2.5-7B-Instruct`
- vLLM 模型名：`qwen2.5-7b`

项目使用两个独立虚拟环境：

| 环境 | 用途 | 是否需要 GPU |
|---|---|---:|
| `.venv1` | vLLM、PyTorch、模型推理 | 是 |
| `.venv-m3` | MCP Server、NumPy 推理和测试 | 否 |

分开环境可避免 vLLM 的 PyTorch/CUDA 依赖影响轻量的 MCP 服务。执行命令前，请先确认激活了正确环境。

## 2. 首次安装

以下命令只需执行一次。建议在登录节点创建环境，在计算节点启动 vLLM。

### 2.1 创建 vLLM 环境

```bash
module load python312
module load cuda/12.8.0

cd /absolute/path/to/30-LLM-Project
python3 -m venv .venv1
source .venv1/bin/activate

python -m pip install --upgrade pip
python -m pip install vllm openai
```

检查安装：

```bash
python --version
python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"
```

在 GPU 计算节点上，最后一项应为 `True`。

### 2.2 创建 MCP 环境

```bash
module load python312

cd /absolute/path/to/30-LLM-Project
python3 -m venv .venv-m3
source .venv-m3/bin/activate

python -m pip install --upgrade pip
python -m pip install "mcp[cli]>=2,<3" anyio numpy openai pydantic pytest
```

检查安装并运行 CPU 测试：

```bash
python -c "import anyio, mcp, numpy, openai, pydantic"
pytest -q mcp_server/tests
```

MCP 还依赖论文仓库中的离线数据：

```text
reference/Popular-but-Wrong/data/
reference/Popular-but-Wrong/res/
```

### 2.3 requirement.txt

```bash
虚拟环境venv1: model_runtime/vllm/requirements.txt
虚拟环境venv-m3: mcp_server/requirements.txt
```

## 3. 启动 vLLM

### 3.1 申请 GPU

```bash
srun \
  --partition=MGPU-TC2 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=02:00:00 \
  --pty bash -l
```

### 3.2 在计算节点启动服务

```bash
module load python312
module load cuda/12.8.0

cd /absolute/path/to/30-LLM-Project
source .venv-vllm/bin/activate

export M3_HF_HOME="/absolute/project-or-scratch/path/huggingface"
mkdir -p "$M3_HF_HOME"

model_runtime/vllm/start_qwen25_7b.sh
```

默认地址为 `http://127.0.0.1:8000`。健康检查：

```bash
curl --fail http://127.0.0.1:8000/v1/models
```

服务只监听 `127.0.0.1`，调用程序应与 vLLM 位于同一计算节点。

可通过以下变量覆盖默认配置：

```text
M3_MODEL_ID
M3_SERVED_MODEL_NAME
M3_VLLM_HOST
M3_VLLM_PORT
M3_MAX_MODEL_LEN
M3_GPU_MEMORY_UTILIZATION
```

## 4. 启动 MCP Server

另开一个位于同一节点的终端，激活 MCP 环境：

```bash
cd /absolute/path/to/30-LLM-Project
source .venv-m3/bin/activate

export PYTHONPATH="$PWD"
export POPULAR_BUT_WRONG_REPO="$PWD/reference/Popular-but-Wrong"

python -m mcp_server.server
```

MCP Server 使用 stdio，不监听端口。启动后无输出并持续等待客户端是正常现象。日志不得写入 stdout，以免破坏 JSON-RPC 协议；应写入 stderr。

MCP host 配置示例：

```json
{
  "mcpServers": {
    "popular-but-wrong-calibrator": {
      "command": "/absolute/path/to/30-LLM-Project/.venv-m3/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/30-LLM-Project",
      "env": {
        "PYTHONPATH": "/absolute/path/to/30-LLM-Project",
        "POPULAR_BUT_WRONG_REPO": "/absolute/path/to/30-LLM-Project/reference/Popular-but-Wrong"
      }
    }
  }
}
```

## 5. `calibrate` 工具

### 输入 Schema

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `question` | string | 是 | 原始问题，建议保留数据集中的规范文本 |
| `answer` | string | 是 | vLLM 生成的纯答案 |
| `confidence` | number | 是 | 原始置信度，范围 `[0,1]` |
| `model` | string | 是 | artifact 模型名，如 `qwen2.5-7b` |
| `domain` | string / null | 否 | `movies`、`songs` 或 `basketball`；省略时尝试自动推断 |

原始置信度必须按答案 token 概率的算术平均计算：

```python
confidence = sum(math.exp(token.logprob) for token in answer_tokens) / len(answer_tokens)
```

计算时排除 `<|im_end|>`、`<|endoftext|>` 等特殊停止 token。不要使用序列概率，也不要使用 `exp(mean(logprob))`。

### 主要输出

| 字段 | 说明 |
|---|---|
| `status` | `ok` 或 `degraded` |
| `raw_confidence` | vLLM 原始置信度 |
| `calibrated_probability` | MLP 校准后的正确概率；无法校准时为 `null` |
| `threshold` | P2 导出的决策阈值 |
| `decision` | `answer`、`retrieve` 或 `unknown` |
| `artifact` | 实际使用的 `<model>__<domain>` artifact |
| `reason` | 降级原因 |
| `warnings` | 缺特征、实体噪声等提示 |
| `feature_values` | `confidence`、`qpop`、`gene_pop`、`gene_coo` |
| `feature_coverage` | 三个流行度特征是否成功命中 |

当校准成功时：

```python
decision = "answer" if calibrated_probability >= threshold else "retrieve"
```

### 调用示例

```python
import anyio
from mcp import Client, StdioServerParameters


async def main():
    server = StdioServerParameters(
        command="/absolute/path/to/30-LLM-Project/.venv-m3/bin/python",
        args=["-m", "mcp_server.server"],
        env={
            "PYTHONPATH": "/absolute/path/to/30-LLM-Project",
            "POPULAR_BUT_WRONG_REPO": (
                "/absolute/path/to/30-LLM-Project/reference/Popular-but-Wrong"
            ),
        },
    )

    async with Client(server) as client:
        result = await client.call_tool(
            "calibrate",
            arguments={
                "question": "Who is the director of the movie The Intouchables",
                "answer": "Olivier Nakache",
                "confidence": 0.8954265,
                "model": "qwen2.5-7b",
                "domain": "movies",
            },
        )
        print(result.structured_content)


anyio.run(main)
```

返回示例：

```json
{
  "status": "ok",
  "raw_confidence": 0.8954265,
  "calibrated_probability": 0.9668866,
  "threshold": 0.3449275,
  "decision": "answer",
  "artifact": "qwen2.5-7b__movies",
  "reason": null,
  "warnings": []
}
```

概率取决于输入、流行度特征和 artifact，示例值不是固定期望值。

## 6. 降级策略

输入不符合 Schema 时，MCP 返回工具错误，例如：

- `question` 或 `model` 为空；
- `confidence` 不是数字或不在 `[0,1]`；
- `domain` 不在支持列表中。

运行时资源缺失时，服务返回结构化降级结果，不让 Agent 崩溃：

| 场景 | `reason` | 处理方式 |
|---|---|---|
| 无法推断领域 | `domain_not_inferred` | 返回 `unknown` |
| 答案为空 | `empty_answer` | 返回 `unknown` |
| 实体噪声过高 | `noisy_entity` | 返回 `unknown` |
| 三个流行度特征全部缺失 | `no_popularity_features` | 返回 `unknown` |
| 部分流行度特征缺失 | `partial_feature_coverage` | 缺失值补 0，返回概率和决策，同时标记 `degraded` |
| 找不到权重 | `artifact_not_found` | 返回原始分数和 `unknown` |
| artifact 损坏 | `artifact_invalid` | 返回原始分数和 `unknown` |

`fallback_score` 是原始置信度，不是校准概率，不能与 MLP 阈值直接比较。上游收到 `unknown` 后应进入检索、拒答或其他兜底流程。

## 7. 已知限制

1. **实体链接较朴素。** 当前主要依赖问题模板和字符串清洗，别名、缩写、重音字符、同名实体、多实体答案或解释性回答可能无法命中。
2. **仅支持三个领域。** 当前只支持电影导演、歌曲表演者和篮球球员出生地问题。
3. **离线表覆盖有限。** 流行度表来自论文快照，不会自动更新；新问题或新答案可能没有特征。
4. **训练与部署协议必须一致。** 模型版本、prompt、chat template、解码参数或 tokenizer 变化后，应重新验证或重新训练校准器。
5. **Basketball 上的 Qwen2.5 结果仅供实验。** 论文未将该组合纳入主要后续分析。
6. **MCP 仅支持 stdio。** 当前没有 HTTP、认证、TLS 或多租户能力。
7. **vLLM 随 Slurm 作业结束。** 作业超时、抢占或节点故障后，端点会消失。

## 8. 测试与验收

### CPU 测试

```bash
source .venv-m3/bin/activate
pytest -q mcp_server/tests
```

### 已启动 vLLM 时的闭环测试

```bash
source .venv-m3/bin/activate
python -m mcp_server.e2e_smoke_test
```

### Slurm 一键验收

```bash
export M3_PROJECT_ROOT="/absolute/path/to/30-LLM-Project"
export M3_HF_HOME="/absolute/project-or-scratch/path/huggingface"

sbatch \
  --export=ALL,M3_PROJECT_ROOT="$M3_PROJECT_ROOT",M3_HF_HOME="$M3_HF_HOME" \
  mcp_server/m3_e2e.slurm
```

检查作业状态：

```bash
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

通过标准：

- 日志包含 `M3 end-to-end smoke test passed`；
- 日志包含 `M3 Slurm acceptance passed`；
- 作业状态为 `COMPLETED`；
- 退出码为 `0:0`。

M3 的最终验收项：

1. vLLM 返回答案和逐 token logprobs；
2. 原始置信度计算符合论文定义；
3. MCP 发布 `calibrate` 工具及 Schema；
4. 在线 NumPy 推理与 P2 离线推理误差小于 `1e-6`；
5. 正常输入返回校准概率和 `answer/retrieve`；
6. 缺特征、缺权重和实体噪声场景返回结构化降级；
7. Slurm 验收作业以 `COMPLETED`、`0:0` 结束。

