"""微调模块 —— M3。

`dataset.py` 生成训练数据，`train_lora.py` 跑 QLoRA 训练。
两者之间只通过 JSONL 文件耦合：训练可以在租来的 GPU 上跑，
那台机器不需要装 `data/` 那一整套依赖。
"""
