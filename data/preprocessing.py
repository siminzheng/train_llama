from datasets import load_dataset
from itertools import chain

# 加载文本数据函数
# train_dir: 训练数据目录，eval_dir: 验证数据目录
def load_text_datasets(train_dir, eval_dir):
    train_ds = load_dataset("text", data_dir=train_dir, split="train")
    eval_ds = load_dataset("text", data_dir=eval_dir, split="train")
    return train_ds, eval_ds

# 分词函数生成器，返回对 example 中 text 列进行分词的函数
def tokenize_function(tokenizer):
    def fn(examples):
        return tokenizer(examples["text"])
    return fn

# 按 max_seq_length 切分文本块，并生成 labels
def group_texts(examples, block_size):
    # 将每个字段展平
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_len = len(concatenated[list(examples.keys())[0]])
    # 丢弃不够整块的尾部
    total_len = (total_len // block_size) * block_size
    # 切分
    result = {
        k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items()
    }
    # labels 与 input_ids 相同
    result["labels"] = result["input_ids"].copy()
    return result
