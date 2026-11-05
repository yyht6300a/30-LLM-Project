# calibrator：数据准备、特征与校准器训练

所有者：P1（data_prep、features）、P2（training、eval）。

- `data_prep/`：QA 数据加载、50/50 分层切分、10% 验证集。
- `features/`：流行度离线表（entity → sitelinks、实体对 → 共现）、实体链接、表外兜底。
- `training/`：MLP 训练与基线复现。
- `eval/`：ECE / Brier / AUROC / Alignment / Confw 指标与阈值标定。
