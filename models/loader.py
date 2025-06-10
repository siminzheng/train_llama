import torch  # 导入 PyTorch，进行张量运算和模型加载

from accelerate import PartialState  # 导入 accelerate，用于分布式设备管理

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# AutoTokenizer: 自动加载对应的分词器
# AutoModelForCausalLM: 自动加载因果语言模型（支持文本生成）
# BitsAndBytesConfig: 用于配置量化参数

from peft import TaskType, LoraConfig, get_peft_model
# peft: 参数高效微调库，支持 LoRA 等技术
# TaskType: 任务类型定义
# LoraConfig: LoRA 微调参数配置
# get_peft_model: 将 LoRA 配置应用到模型上


# 构建 4-bit 量化配置
def build_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,  # 启用 4-bit 量化，显著降低显存占用
        bnb_4bit_use_double_quant=True,  # 使用双重量化，进一步优化存储效率
        bnb_4bit_quant_type="nf4",  # 使用 nf4（normal float 4-bit）量化类型，兼顾精度和效率
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用 bfloat16，节省显存且适配 GPU
    )


# 加载分词器
def get_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)  # 从 Hugging Face Hub 或本地路径加载分词器


# 加载 4-bit 量化模型
def get_quantized_model(model_path, bnb_config):
    return AutoModelForCausalLM.from_pretrained(
        model_path,  # 指定模型路径（可以是 llama、baichuan、mistral 等）
        low_cpu_mem_usage=True,  # 启用低 CPU 内存加载模式，减少内存占用
        quantization_config=bnb_config,  # 使用构建好的 4-bit 量化配置
        device_map={"": PartialState().process_index},  # 自动将当前进程的模型加载到合适的 GPU 上
    )


# 应用 LoRA 参数高效微调
def apply_lora(model, r, modules, alpha=16, dropout=0.05):
    peft_conf = LoraConfig(
        r=r,  # LoRA 的秩，控制可训练参数数量（越小越省显存）
        target_modules=modules,  # 需要注入 LoRA 的目标模块，一般为投影层（q_proj, v_proj 等）
        task_type=TaskType.CAUSAL_LM,  # 指定任务类型为因果语言模型（生成任务）
        lora_alpha=alpha,  # LoRA 缩放系数，影响 LoRA 输出的权重调整幅度
        lora_dropout=dropout,  # LoRA dropout，防止过拟合
    )
    model = get_peft_model(model, peft_conf)  # 将 LoRA 配置应用到模型中
    model.print_trainable_parameters()  # 打印当前模型中可训练参数的比例，确认是否注入成功
    return model  # 返回注入 LoRA 后的模型
