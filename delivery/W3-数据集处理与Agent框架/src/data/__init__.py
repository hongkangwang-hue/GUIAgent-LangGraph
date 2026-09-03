"""数据集处理模块（M2 交付物 1）。

把公开 GUI 数据集统一为同一 schema，为 M3 的 grounding 训练集与零样本
评测集做准备。

子模块职责：

- `schema`  —— 统一样本定义与 JSONL 读写，是本模块的唯一对外数据契约
- `loaders` —— 各数据集的装载器，把原始格式翻译成 `UnifiedSample`
- `clean`   —— 清洗规则，每条规则单独计数，便于在报告里说明剔除了什么
- `stats`   —— 统计聚合
- `charts`  —— 4 张 Matplotlib 图表
- `split`   —— M3 前置件①：训练 / 验证集划定与冻结

原始数据放在 `data/raw/`，中间产物放 `data/processed/`，两者都不入库
（见 `.gitignore`）。
"""
