# 导入 Hugging Face transformers 库
import transformers

# 导入 PyTorch
import torch

# 导入 Hugging Face 的 Trainer 类（通用训练框架）和 AutoTokenizer（自动加载 tokenizer）
from transformers import Trainer, AutoTokenizer

# 导入 accelerate 工具，PartialState 用于多卡/分布式训练支持
from accelerate import PartialState

# 导入 peft 库（用于参数高效微调，如 LoRA）
from peft import LoraConfig, TaskType, get_peft_model

# 导入 Hugging Face 的因果语言模型（AutoModelForCausalLM）和量化配置（BitsAndBytesConfig）
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 导入自定义数据预处理方法
from data.preprocessing import load_text_datasets, tokenize_function, group_texts

# 导入自定义模型加载与配置函数
from models.loader import build_bnb_config, get_tokenizer, get_quantized_model, apply_lora

# 导入自定义训练参数配置类
from configs.training_args import CustomArguments

# 导入关闭警告工具
from utils.logging import disable_warnings

# 主函数，训练流程入口
def main():
    # 关闭训练过程中的非必要警告信息
    disable_warnings()

    # 解析自定义训练参数（通过命令行输入）
    parser = transformers.HfArgumentParser(CustomArguments)
    training_args, = parser.parse_args_into_dataclasses()

    # 读取模型路径
    model_path = training_args.model_name_or_path

    # 构建 BitsAndBytes（bnb）量化配置，用于加载低精度模型
    bnb_config = build_bnb_config()

    # 加载分词器
    tokenizer = get_tokenizer(model_path)

    # 加载量化后的预训练模型
    model = get_quantized_model(model_path, bnb_config)

    # 给模型应用 LoRA（低秩适配器）微调，只训练部分参数
    model = apply_lora(
        model,
        training_args.lora_r,  # LoRA 中的秩（r），控制可训练参数量
        [
            # 只训练 transformer 中的这几个投影层，节省显存
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ],
    )

    # 加载并预处理训练集和验证集
    train_ds, eval_ds = load_text_datasets(
        training_args.train_data_dir,  # 训练数据路径
        training_args.eval_data_dir,   # 验证数据路径
    )

    # 分词处理（支持多进程）
    with training_args.main_process_first(desc="tokenize"):
        train_ds = train_ds.map(
            tokenize_function(tokenizer),   # 分词函数
            remove_columns=["text"],        # 删除原始文本列，节省内存
            num_proc=training_args.num_proc, # 并行进程数
        )
        eval_ds = eval_ds.map(
            tokenize_function(tokenizer),
            remove_columns=["text"],
            num_proc=training_args.num_proc,
        )

    # 拼接文本，按 max_seq_length 分组，适配语言模型输入格式
    with training_args.main_process_first(desc="group_texts"):
        train_ds = train_ds.map(
            lambda ex: group_texts(ex, training_args.max_seq_length),
            batched=True, num_proc=training_args.num_proc,
        )
        eval_ds = eval_ds.map(
            lambda ex: group_texts(ex, training_args.max_seq_length),
            batched=True, num_proc=training_args.num_proc,
        )

    # 创建 Hugging Face Trainer（训练核心）
    trainer = Trainer(
        model=model,             # 训练模型（已加 LoRA）
        args=training_args,      # 训练参数
        train_dataset=train_ds,  # 训练集
        eval_dataset=eval_ds,    # 验证集
    )

    # 开始训练
    trainer.train()

    # 保存微调后的模型
    trainer.save_model(training_args.output_dir)

# 程序入口
if __name__ == '__main__':
    main()
