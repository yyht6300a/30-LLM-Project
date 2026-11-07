# calibrator/training 与 eval：P2 使用说明

所有者：P2。复现论文的校准器训练与评测，并向 P3 导出可加载的权重。

## 环境

`Python ≥ 3.10`、`numpy`、`pandas`、`scikit-learn`、`pytorch`（CPU 版即可）。
训练数据来自论文仓库 `data/model_outputs/*.jsonl.gz`，按 `data_prep` 的约定放在
`reference/Popular-but-Wrong/` 下，或设置环境变量 `POPULAR_BUT_WRONG_REPO`。

## 运行

```bash
# 全部 18 个 (model, dataset) 组合的评测，输出 JSON + Markdown 报告
python -m calibrator.eval --output-dir calibrator/eval/output

# 只评测部署模型 Qwen2.5-7B
python -m calibrator.eval --models qwen2.5-7b

# 训练并导出部署用 MLP(full) 权重（P3 的 MCP 加载这些文件）
python -m calibrator.training --models qwen2.5-7b --datasets movies songs basketball
```

## 复现范围

- MLP 用 PyTorch 精确复现论文架构：三层 (128, 64, 32)，每层 BatchNorm + ReLU +
  Dropout(0.3)，AdamW(lr 5e-4, weight_decay 1e-4)，BCE 损失，10% 验证集早停。
- 基线：Prob、Consis、Verbal、Temp（温度缩放）、Platt、Isotonic。
- 单信号阈值：PopQ、PopGe、RPopGe。
- MLP 消融：`MLP(c)`、`MLP(c+RPopGe)`、`MLP(full)`（= c + PopQ + PopGe + RPopGe）。
- 指标：Alignment、AUROC、ECE、Brier、Confw。
- 阈值：在训练半区（train ∪ val）上扫描唯一值、最大化准确率（论文附录 B.2）。

## 导出格式（供 P3）

`python -m calibrator.training` 在 `artifacts/` 下为每个 (model, dataset) 生成：

- `<model>__<dataset>.pt`：完整 PyTorch state_dict（复现与审计用）。
- `<model>__<dataset>.npz`：把 BatchNorm 折叠进线性层后的 NumPy 推理权重，P3 用
  `calibrator.training.mlp.load_folded` 加载即可，**不需要 torch**。
- `<model>__<dataset>.json`：元数据，含 `features`（列顺序）、`scaler_mean` /
  `scaler_scale`（推理前标准化输入）、`threshold`（answer/retrieve 决策阈值）与
  `test_metrics`。
- `manifest.json`：全部组合的索引。

P3 推理流程：按 `features` 顺序取 `[confidence, qpop, gene_pop, gene_coo]`，用
`scaler_mean` / `scaler_scale` 标准化，`load_folded` 加载 `.npz` 后预测 P(正确)，
再与 `threshold` 比较得出 answer/retrieve 决策。未命中流行度表的实体按 0 填（与
训练一致）。

`artifacts/` 已加入 `.gitignore`，二进制权重不进仓库；一行命令即可重新生成。
