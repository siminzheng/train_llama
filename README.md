# train_llama
手动训练一个Llama3模型的简单示例

```text
模型架构：
llama-lora-finetune/
├── README.md
├── requirements.txt
├── configs/
│   └── training_args.py
├── data/
│   └── preprocessing.py
├── models/
│   └── loader.py
├── utils/
│   └── logging.py
└── scripts/
    └── train.py
```
# llama-lora-finetune

基于 Meta-Llama-3-Instruct 的 LoRA+BitsAndBytes 量化低资源微调示例。

## 快速开始
```bash
pip install -r requirements.txt

python scripts/train.py \
  --model_name_or_path /data04/llama3/Meta-Llama-3.1-8B-Instruct \
  --train_data_dir /home/xuepeng/pretrain_test/train_data \
  --eval_data_dir /home/xuepeng/pretrain_test/eval_data
```


