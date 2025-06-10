import torch
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import TaskType, LoraConfig, get_peft_model

# 构建量化配置
def build_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# 加载分词器
def get_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)

# 加载量化模型
def get_quantized_model(model_path, bnb_config):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
    )

# 应用 LoRA 微调
def apply_lora(model, r, modules, alpha=16, dropout=0.05):
    peft_conf = LoraConfig(
        r=r,
        target_modules=modules,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )
    model = get_peft_model(model, peft_conf)
    model.print_trainable_parameters()
    return model
