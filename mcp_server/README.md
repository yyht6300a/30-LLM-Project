# mcp_server：校准器 MCP 服务

所有者：P3。

# Qwen2.5-7B vLLM 运行时与 M3/M4 接入说明

本文档说明如何在学校 GPU 集群上启动 Qwen2.5-7B-Instruct、获取逐 token logprobs，并将生成结果交给 M3 MCP 校准服务。


---

## 1. 系统数据流

完整流程如下：

```text
用户问题
  ↓
OpenHands 调用 vLLM 生成短答案
  ↓
vLLM 返回 answer 和逐 token logprobs
  ↓
M4 计算原始置信度 c
  ↓
M4 调用 MCP calibrate 工具
  ↓
M3 查询 PopQ、PopGe、RPopGe
  ↓
加载对应 (model, domain) 的 MLP 权重
  ↓
返回 calibrated_probability 和 decision
  ↓
OpenHands 执行 answer / retrieve
```

M3 提供两个能力：

1. OpenAI-compatible vLLM API：

```text
http://127.0.0.1:8000/v1
```

2. stdio MCP Server：

```bash
python -m mcp_server.server
```

---

## 2. 已验证运行环境

当前配置已在学校 GPU 集群验证：

| 项目 | 配置 |
|---|---|
| 调度系统 | Slurm |
| GPU 分区 | `MGPU-TC2` |
| GPU | NVIDIA A40 46GB |
| GPU 数量 | 1 |
| CUDA | 12.8 |
| 模型 | `Qwen/Qwen2.5-7B-Instruct` |
| 对外模型名 | `qwen2.5-7b` |
| 推理精度 | BF16 |
| vLLM 环境 | `.venv-vllm` |
| MCP 环境 | `.venv-m3` |

Qwen2.5-7B 可以在单张 A40 上运行，不需要 tensor parallel，也不应默认使用量化模型。

量化会改变 token 概率分布，可能导致 P2 训练出的校准器失效。

---

## 3. 目录与文件

M3 相关文件：

```text
model_runtime/vllm/
├── start_qwen25_7b.sh       # 启动 vLLM
├── smoke_test.py            # vLLM 与 logprobs 测试
├── m3_e2e.slurm             # Slurm 一键端到端验收
└── README.md

mcp_server/
├── artifact_store.py        # 加载 P2 metadata、scaler 和 NPZ 权重
├── service.py               # 特征查询与校准业务逻辑
├── server.py                # MCP calibrate 工具
├── e2e_smoke_test.py        # vLLM → MCP 真实闭环测试
└── tests/
```

模型权重位于：

```text
calibrator/training/artifacts/
```

P1 离线特征资源位于：

```text
reference/Popular-but-Wrong/
```

---

## 4. 启动 vLLM

### 4.1 申请交互式 GPU

```bash
srun \
  --partition=MGPU-TC2 \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=32G \
  --time=02:00:00 \
  --pty bash -l
```

进入计算节点后：

```bash
module load python312
module load cuda/12.8.0

nvidia-smi
```

### 4.2 配置 Hugging Face 缓存

必须将缓存放在学校允许的 project 或 scratch 路径中：

```bash
export M3_HF_HOME="<project-or-scratch>/zhiao001/huggingface"
mkdir -p "$M3_HF_HOME"
```

不要把模型缓存提交到 Git。

### 4.3 启动服务

```bash
cd /path/to/30-LLM-Project
source .venv1/bin/activate

model_runtime/vllm/start_qwen25_7b.sh
```

默认配置：

| 参数 | 默认值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-7B-Instruct` |
| served model name | `qwen2.5-7b` |
| host | `127.0.0.1` |
| port | `8000` |
| dtype | `bfloat16` |
| max model length | `4096` |
| GPU memory utilization | `0.85` |
| seed | `42` |

检查服务：

```bash
curl --fail http://127.0.0.1:8000/v1/models
```

---

## 5. M4 调用 vLLM 的固定协议

M4 必须显式设置以下参数：

```python
response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer with only the requested entity name. "
                "Do not explain your answer."
            ),
        },
        {
            "role": "user",
            "content": canonical_question,
        },
    ],
    temperature=0,
    top_p=1,
    max_tokens=32,
    logprobs=True,
)
```

关键要求：

- `model` 必须是 `qwen2.5-7b`。
- 必须设置 `logprobs=True`。
- 必须使用贪心解码：`temperature=0`。
- 问题放在 user message 中。
- answer-only 指令放在 system message 中。
- 不要给 canonical question 添加额外前缀、后缀或问号。
- 不允许输出解释、推理过程或多段文本。

### 为什么不能修改 canonical question

P1 的 `PopularityLookup` 使用论文问题模板识别领域和实体，例如：

```text
Who is the director of the movie The Intouchables
```

如果改成：

```text
Please answer: Who is the director of the movie The Intouchables?
```

可能无法正确查询 `PopQ`，导致校准结果降级。

---

## 6. 原始置信度 c 的计算

论文中的原始置信度定义为：

```text
生成答案所有 token 概率的算术平均
```

公式：

```python
c = mean(exp(token_logprob))
```

参考实现：

```python
import math

SPECIAL_TOKENS = {
    "<|im_end|>",
    "<|endoftext|>",
}

items = response.choices[0].logprobs.content

answer_tokens = [
    item
    for item in items
    if item.token not in SPECIAL_TOKENS
]

if not answer_tokens:
    raise ValueError("No answer tokens with logprobs")

confidence = sum(
    math.exp(item.logprob)
    for item in answer_tokens
) / len(answer_tokens)
```

必须排除：

```text
<|im_end|>
<|endoftext|>
```

不能使用以下计算方式：

```python
math.exp(mean(logprobs))       # 错误：这是几何平均
math.prod(probabilities)       # 错误：这是序列概率
```

计算结果必须满足：

```text
0 <= confidence <= 1
```

---

## 7. MCP calibrate 工具

MCP Server 启动命令：

```bash
source .venv-m3/bin/activate
python -m mcp_server.server
```

这是 stdio server。直接运行后没有输出并一直阻塞是正常现象，因为它在等待 MCP Host 发送请求。

不要在 MCP Server 中使用 `print()` 输出日志，因为 stdout 是协议通道。

### 7.1 工具输入

工具名称：

```text
calibrate
```

输入：

```json
{
  "question": "Who is the director of the movie The Intouchables",
  "answer": "Olivier Nakache",
  "confidence": 0.8954265,
  "model": "qwen2.5-7b",
  "domain": "movies"
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `question` | string | 原始 canonical question |
| `answer` | string | 当前 vLLM 生成的答案 |
| `confidence` | float | 按论文公式计算的原始置信度 |
| `model` | string | 当前必须使用 `qwen2.5-7b` |
| `domain` | string/null | `movies`、`songs` 或 `basketball` |

如果 `domain` 省略，服务会尝试从论文问题模板推断。

为了让行为更明确，M4 推荐显式传递 domain。

### 7.2 正常返回

```json
{
  "status": "ok",
  "model": "qwen2.5-7b",
  "domain": "movies",
  "raw_confidence": 0.8954265,
  "fallback_score": 0.8954265,
  "calibrated_probability": 0.9668866,
  "threshold": 0.3449275,
  "decision": "answer",
  "artifact": "qwen2.5-7b__movies",
  "reason": null,
  "warnings": [],
  "question_entity": "The Intouchables",
  "answer_entity": "Olivier Nakache",
  "feature_values": {
    "confidence": 0.8954265,
    "qpop": 60.0,
    "gene_pop": 15.0,
    "gene_coo": 9.0
  },
  "feature_coverage": {
    "qpop": true,
    "gene_pop": true,
    "gene_coo": true
  },
  "noisy_entity": false
}
```

---

## 8. M4 决策处理规则

推荐使用以下保守策略：

```python
if result["status"] == "ok":
    if result["decision"] == "answer":
        return_original_answer()
    elif result["decision"] == "retrieve":
        perform_retrieval()

else:
    # degraded 状态不能被当作正常校准成功
    perform_retrieval_or_fallback()
```

建议决策表：

| status | decision | M4 行为 |
|---|---|---|
| `ok` | `answer` | 直接采用原始答案 |
| `ok` | `retrieve` | 执行联网搜索或本地 RAG |
| `degraded` | `unknown` | 优先检索 |
| `degraded` | `answer/retrieve` | 表示部分特征缺失；默认仍建议检索 |
| MCP 调用失败 | 无 | 不阻塞 Agent，进入检索或原始答案兜底 |

不要只检查 `decision` 而忽略 `status`。

### 检索不可用时

如果检索服务不可用：

1. 保留原始答案；
2. 记录 fallback 原因；
3. 不要将 `fallback_score` 描述成校准概率；
4. 不要让 Agent 因校准服务异常而崩溃。

---

## 9. MCP 子进程配置

MCP SDK v2 的 stdio 子进程只继承有限环境变量，因此必须显式提供项目路径。

示例：

```python
from mcp import Client, StdioServerParameters

server = StdioServerParameters(
    command="/absolute/path/30-LLM-Project/.venv-m3/bin/python",
    args=["-m", "mcp_server.server"],
    env={
        "PYTHONPATH": "/absolute/path/30-LLM-Project",
        "POPULAR_BUT_WRONG_REPO": (
            "/absolute/path/30-LLM-Project/"
            "reference/Popular-but-Wrong"
        ),
    },
)
```

M4 接入 OpenHands 时，应使用集群上的真实绝对路径。

---

## 10. Slurm 部署注意事项

vLLM 默认绑定：

```text
127.0.0.1:8000
```

这意味着 OpenHands/M4 客户端必须与 vLLM 运行在同一个计算节点。

推荐方式：

```text
同一个 Slurm 作业
├── vLLM 后台进程
├── OpenHands Agent
└── MCP stdio 子进程
```

不要默认从另一个计算节点访问 `127.0.0.1:8000`。

如果将 vLLM 改为 `0.0.0.0`，必须同时考虑：

- 集群网络策略；
- 端口冲突；
- 身份验证；
- 未授权访问；
- 学校安全规定。

当前项目默认不开放远程端口。

---

## 11. 一键端到端验收

设置缓存：

```bash
export M3_HF_HOME="<project-or-scratch>/zhiao001/huggingface"
```

提交：

```bash
sbatch \
  --export=ALL,M3_HF_HOME="$M3_HF_HOME" \
  model_runtime/vllm/m3_e2e.slurm
```

查看作业：

```bash
squeue -j <JOB_ID>
```

查看输出：

```bash
tail -f m3-e2e-<JOB_ID>.out
```

查看模型日志：

```bash
tail -f logs/vllm-<JOB_ID>.log
```

验收成功时，输出末尾包含：

```text
M3 end-to-end smoke test passed
M3 Slurm acceptance passed
```

最终检查：

```bash
sacct \
  -j <JOB_ID> \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

期望：

```text
State     COMPLETED
ExitCode  0:0
```

---

## 12. 本地测试命令

MCP 与业务逻辑测试不需要 GPU：

```bash
source .venv-m3/bin/activate
pytest -q mcp_server/tests
```

vLLM 基础测试需要在 GPU 节点运行：

```bash
source .venv-vllm/bin/activate
python model_runtime/vllm/smoke_test.py
```

真实闭环测试：

```bash
source .venv-m3/bin/activate
python -m mcp_server.e2e_smoke_test
```

---

## 13. 当前支持范围与限制

当前离线 popularity 表主要覆盖论文的三类问题：

| domain | 问题类型 |
|---|---|
| `movies` | 电影 → 导演 |
| `songs` | 歌曲 → 表演者 |
| `basketball` | 篮球运动员 → 出生地 |

限制：

1. 任意开放领域问题可能无法命中 popularity 表。
2. 在线答案与论文归档答案写法差异过大时，可能缺少 `PopGe` 或 `RPopGe`。
3. 缺失单个特征时，服务按论文协议补 0，并返回 `degraded`。
4. 所有 popularity 特征都缺失时，decision 为 `unknown`。
5. Qwen2.5 系列在 Basketball 数据上的原始准确率很低，该组合应视为实验性支持。
6. 不允许使用其他模型的校准权重代替当前模型权重。

---

