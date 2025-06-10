from dataclasses import dataclass, field
import transformers

@dataclass
class CustomArguments(transformers.TrainingArguments):
    """
    自定义 TrainingArguments：包含 LoRA 和数据处理相关参数。
    """
    # LoRA 的矩阵秩
    lora_r: int = field(default=8)
    # 数据预处理并行进程数
    num_proc: int = field(default=1)
    # 文本最大序列长度
    max_seq_length: int = field(default=32)
    # 验证策略，'no' 表示不验证
    eval_strategy: str = field(default="steps")
    # 验证步长
    eval_steps: int = field(default=100)
    # 随机种子
    seed: int = field(default=0)
    # 优化器类型
    optim: str = field(default="adamw_torch")
    # 训练轮数
    num_train_epochs: int = field(default=2)
    # 每设备批大小
    per_device_train_batch_size: int = field(default=1)
    # 学习率
    learning_rate: float = field(default=5e-5)
    # 权重衰减
    weight_decay: float = field(default=0)
    # 预热步数
    warmup_steps: int = field(default=10)
    # 学习率调度器类型
    lr_scheduler_type: str = field(default="linear")
    # 梯度检查点
    gradient_checkpointing: bool = field(default=False)
    # bf16 混合精度
    bf16: bool = field(default=True)
    # 梯度累加
    gradient_accumulation_steps: int = field(default=1)
    # 日志记录步长
    logging_steps: int = field(default=3)
    # 保存策略
    save_strategy: str = field(default="steps")
    # 保存步长
    save_steps: int = field(default=3)
    # 最多保留 checkpoint 数量
    save_total_limit: int = field(default=2)
