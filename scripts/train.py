import transformers
import torch
from transformers import Trainer, AutoTokenizer
from accelerate import PartialState
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from data.preprocessing import load_text_datasets, tokenize_function, group_texts
from models.loader import build_bnb_config, get_tokenizer, get_quantized_model, apply_lora
from configs.training_args import CustomArguments
from utils.logging import disable_warnings

# 主函数
def main():
    # 关闭警告
    disable_warnings()

    # 解析自定义训练参数
    parser = transformers.HfArgumentParser(CustomArguments)
    training_args, = parser.parse_args_into_dataclasses()

    # 构造路径与配置
    model_path = training_args.model_name_or_path
    bnb_config = build_bnb_config()

    # 加载分词器与模型
    tokenizer = get_tokenizer(model_path)
    model = get_quantized_model(model_path, bnb_config)
    model = apply_lora(
        model,
        training_args.lora_r,
        [
            "q_proj","v_proj","k_proj","o_proj",
            "gate_proj","down_proj","up_proj"
        ],
    )

    # 加载并预处理数据
    train_ds, eval_ds = load_text_datasets(
        training_args.train_data_dir,
        training_args.eval_data_dir,
    )
    # 分词
    with training_args.main_process_first(desc="tokenize"):
        train_ds = train_ds.map(
            tokenize_function(tokenizer),
            remove_columns=["text"],
            num_proc=training_args.num_proc,
        )
        eval_ds = eval_ds.map(
            tokenize_function(tokenizer),
            remove_columns=["text"],
            num_proc=training_args.num_proc,
        )
    # 分组
    with training_args.main_process_first(desc="group_texts"):
        train_ds = train_ds.map(
            lambda ex: group_texts(ex, training_args.max_seq_length),
            batched=True, num_proc=training_args.num_proc,
        )
        eval_ds = eval_ds.map(
            lambda ex: group_texts(ex, training_args.max_seq_length),
            batched=True, num_proc=training_args.num_proc,
        )
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == '__main__':
    main()
