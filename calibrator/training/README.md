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

# 训练并导出三种特征集的部署权重（full / c / c_rpopge）
python -m calibrator.training --models qwen2.5-7b --datasets movies songs basketball

# 只导出全特征集（保持旧命名）
python -m calibrator.training --variants full
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

`python -m calibrator.training` 在 `artifacts/` 下为每个 (model, dataset, variant)
生成一组文件。variant 有三档，对应消融实验的特征子集：

| variant | 特征 | 文件名 |
|---|---|---|
| `full` | confidence + qpop + gene_pop + gene_coo | `<model>__<dataset>` |
| `c` | confidence | `<model>__<dataset>__c` |
| `c_rpopge` | confidence + gene_coo | `<model>__<dataset>__c_rpopge` |

每档包含：

- `<stem>.pt`：完整 PyTorch state_dict（复现与审计用）。
- `<stem>.npz`：把 BatchNorm 折叠进线性层后的 NumPy 推理权重，用
  `calibrator.training.mlp.load_folded` 加载，**不需要 torch**。
- `<stem>.json`：元数据，含 `features`（列顺序）、`variant`、`scaler_mean` /
  `scaler_scale`、`threshold`、`test_metrics`。
- `manifest.json`：按 `model/dataset/variant` 索引全部组合。

`full` 不带后缀，保持与旧版 `mcp_server.artifact_store.ArtifactStore` 的兼容；消融
用的 `c` / `c_rpopge` 带后缀，P3/P4 需要按 `variant` 选择对应的 `<stem>` 加载。

P3 推理流程：按 JSON 里的 `features` 顺序取对应特征，用 `scaler_mean` /
`scaler_scale` 标准化，`load_folded` 加载 `.npz` 后预测 P(正确)，再与 `threshold`
比较得出 answer/retrieve 决策。未命中流行度表的实体按 0 填（与训练一致）。

权重（`.pt` / `.npz` / `.json`）已提交到仓库；如需更新，运行
`python -m calibrator.training` 重新生成即可。
