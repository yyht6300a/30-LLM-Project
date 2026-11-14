# scripts：公共脚本与 CI

所有者：P5。

| 文件 | 作用 |
|---|---|
| `reproduce.sh` | 一键复现：建环境、拉论文仓库并生成查找表、跑测试、复现 P1/P2 结果与离线曲线；`gpu` 参数在 TC2 提交端到端消融作业 |
| `requirements.txt` | CPU 环境 `.venv-eval` 的依赖（测试、复现、评测客户端、demo agent） |
| `demo.sh` | 端到端 demo：在 GPU 节点起 bf16 vLLM（含工具调用参数）并运行 OpenHands agent |

CI 门禁见 `.github/workflows/ci.yml`：每次 push / PR 在 GitHub 上运行 `bash scripts/reproduce.sh test`。

## 用法

```bash
# Linux / TC2 登录节点，Python >= 3.12（TC2 先 module load python312）
bash scripts/reproduce.sh test   # 环境 + 论文数据 + 全部 CPU 测试（CI 跑的就是这一步）
bash scripts/reproduce.sh        # 上一步 + P1 论文统计 + P2 校准器评测与权重 + 离线覆盖率-准确率曲线
bash scripts/reproduce.sh gpu    # 在 TC2 提交端到端消融作业（evaluation/run_ablation.slurm）
```

P2 重新生成的评测报告与权重写入 `reproduce_output/`，不覆盖仓库中
端到端评测实际使用的权重，可与 `calibrator/eval/output/`、`calibrator/training/artifacts/`
对比。端到端消融结果写入 `evaluation/output/`，说明见 [evaluation/README.md](../evaluation/README.md)。

## TC2 复现结果（2026-11-13）

环境：TC2 计算节点 TC2N02，`module load python312`，`.venv-eval` 按 `requirements.txt` 安装
（torch 2.15.0+cpu），论文仓库 commit `1b1f250`。`bash scripts/reproduce.sh` 跑完 P1 后，
srun 交互会话在 P2 阶段被 Slurm 取消（22:00:01），P2 与离线曲线三条命令在新的 srun 会话中手动补跑：

```bash
python -m calibrator.eval --output-dir reproduce_output/calibrator_eval
python -m calibrator.training --artifacts-dir reproduce_output/artifacts
python -m evaluation.coverage_accuracy
```

原始输出提交在 [reproduce_output/](../reproduce_output/)：`p1/`（P1 统计与 bootstrap 日志、
`bootstrap_results.json`）、`calibrator_eval/`（P2 评测报告）、`artifacts/`（P2 重新训练的 18 套权重）。

结论：P1 统计与仓库一致；P2 重新训练的权重数值不是逐位相同，但 18 套中 14 套测试集 Alignment
相同，4 套（均为 movies）各差 1 道测试题。端到端评测使用的 qwen2.5-7b 三套：权重最大差
0.018，决策阈值差 < 0.001，测试集 Alignment / AUROC / Brier 相同，只有 basketball 的 ECE
0.053 → 0.050。

### P1 论文统计

`reproduce.py --stage core` 输出与 [P1_REPRODUCTION_REPORT.md](../calibrator/P1_REPRODUCTION_REPORT.md) 一致，例如：

| 检查项 | TC2 输出 | P1 报告 |
|---|---|---|
| 表 0 Movies × Qwen2.5-7B：Acc / Conf | 0.210 / 0.754 | 0.210 / 0.754 |
| 表 0 Movies × llama8b：Acc / Conf | 0.715 / 0.904 | 0.715 / 0.904 |
| 表 6 Movies 错误/正确样本 PopGe→conf 偏相关比（除 ChatGPT） | 1.7×–4.9× | 1.7–4.9 倍 |
| 表 9 Movies 错误答案 PopGe > PopGT 比例 | 0.449–0.642 | 44.9%–64.2% |

bootstrap 置信区间（`bootstrap_ci.py`）同样跑完，结果写入论文仓库 `code/analysis_correlation/bootstrap_results.json`。

### P2 校准器权重（测试集 Alignment）

| 模型 / 数据集 | 仓库 | TC2 | 差异 |
|---|---|---|---|
| llama3-8b / movies | 0.92424 | 0.92403 | −1 题（n_test = 4765） |
| llama3-8b / songs | 0.8505 | 0.8505 | 0 |
| llama3-8b / basketball | 0.6985 | 0.6985 | 0 |
| qwen2-7b / movies | 0.91839 | 0.91860 | +1 题 |
| qwen2-7b / songs | 0.8981 | 0.8981 | 0 |
| qwen2-7b / basketball | 0.6943 | 0.6943 | 0 |
| gpt-3.5-turbo / movies | 0.8152 | 0.8152 | 0 |
| gpt-3.5-turbo / songs | 0.7852 | 0.7852 | 0 |
| gpt-3.5-turbo / basketball | 0.7844 | 0.7844 | 0 |
| **qwen2.5-7b / movies** | 0.9485 | 0.9485 | 0 |
| **qwen2.5-7b / songs** | 0.9257 | 0.9257 | 0 |
| **qwen2.5-7b / basketball** | 0.7140 | 0.7140 | 0 |
| qwen2.5-14b / movies | 0.93367 | 0.93388 | +1 题 |
| qwen2.5-14b / songs | 0.9203 | 0.9203 | 0 |
| qwen2.5-14b / basketball | 0.6755 | 0.6755 | 0 |
| qwen2.5-32b / movies | 0.93665 | 0.93686 | +1 题 |
| qwen2.5-32b / songs | 0.9050 | 0.9050 | 0 |
| qwen2.5-32b / basketball | 0.7367 | 0.7367 | 0 |

`calibration_report.md` 与仓库版本的 diff 只在 MLP 行（基线 Prob / Temp / Platt / Isotonic 等完全一致）：

```diff
-| MLP(c+RPopGe) | 0.903 | 0.881 | 0.791 | 0.935 | 0.924 | 0.925 |
-| MLP(full) | 0.924 | 0.918 | 0.815 | 0.949 | 0.934 | 0.937 |
+| MLP(c+RPopGe) | 0.902 | 0.881 | 0.791 | 0.935 | 0.925 | 0.924 |
+| MLP(full) | 0.924 | 0.919 | 0.815 | 0.949 | 0.934 | 0.937 |
...
-| MLP(full) | 0.879 |
+| MLP(full) | 0.880 |
```

差异来源未核实；P2 训练时的 torch 版本没有记录，CPU 上 MLP 训练的浮点细节随版本变化，
可能让个别边界样本翻转。

### 离线覆盖率–准确率曲线（qwen2.5-7b）

与仓库中的 [coverage_accuracy_qwen2.5-7b.md](../evaluation/output/coverage_accuracy_qwen2.5-7b.md) 完全一致。
注意这一步读的是仓库中的权重（`calibrator/training/artifacts/`），不是上面重新训练的权重：

| 数据集 | 分数 | cov 0.2 | cov 0.4 | cov 0.6 | cov 0.8 | cov 1.0 |
|---|---|---|---|---|---|---|
| movies | raw c | 0.949 | 0.859 | 0.752 | 0.615 | 0.500 |
| movies | MLP(full) | 0.991 | 0.976 | 0.830 | 0.623 | 0.500 |
| songs | raw c | 0.967 | 0.933 | 0.764 | 0.613 | 0.500 |
| songs | MLP(full) | 1.000 | 0.950 | 0.831 | 0.622 | 0.500 |
| basketball | raw c | 0.566 | 0.619 | 0.604 | 0.557 | 0.499 |
| basketball | MLP(full) | 0.823 | 0.770 | 0.678 | 0.590 | 0.499 |

### 未测部分

`gpu` 档中「`.venv1` 不存在时自动创建」未从零测试；`.venv1` 是按
`model_runtime/vllm/requirements.txt` 手动建的，端到端消融作业即用它跑通（见 [evaluation/README.md](../evaluation/README.md)）。

## demo

```bash
srun --partition=MGPU-TC2 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash -l
bash -l scripts/demo.sh
```
