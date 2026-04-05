# ─────────────────────────────────────────────
#  finetune.py — NUST Bank LoRA Fine-tuning (FIXED)
# ─────────────────────────────────────────────

# 🔥 IMPORTANT: Fix CUDA fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

assert torch.cuda.is_available(), "No GPU found!"
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
# CONFIG (SAFE FOR T4)
# ─────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR     = "/content/lora_bank_model_qwenx"
ENRICHED_JSONL = "preprocessed_data/bank_data_master.jsonl"
MOBILE_JSON    = "bank_qa_categories.json"

MAX_SEQ_LEN   = 512        # 🔥 reduced from 1024
BATCH_SIZE    = 1          # 🔥 reduced
GRAD_ACCUM    = 16         # keeps effective batch same
LEARNING_RATE = 2e-4
EPOCHS        = 3
USE_4BIT      = True

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_enriched_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except:
                continue

            if "structured_data" in data:
                q = data["structured_data"].get("question") or data["structured_data"].get("Question")
                a = data["structured_data"].get("answer")   or data["structured_data"].get("Answer")
            else:
                q = data.get("question") or data.get("Question") or data.get("query")
                a = data.get("answer")   or data.get("Answer")   or data.get("response")

            if q and a:
                records.append({"question": q.strip(), "answer": a.strip()})
    return records


def load_mobile_json(path):
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for category in data.get("categories", []):
            for qa in category.get("questions", []):
                q = qa.get("question", "").strip()
                a = qa.get("answer", "").strip()
                if q and a:
                    records.append({"question": q, "answer": a})
    except:
        print("Mobile file missing, skipping.")
    return records


print("Loading data...")
excel_data  = load_enriched_jsonl(ENRICHED_JSONL)
mobile_data = load_mobile_json(MOBILE_JSON)
all_data    = excel_data + mobile_data

print(f"Loaded {len(all_data)} samples")
assert len(all_data) > 0

# ─────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────
def format_and_tokenize(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False
    )

    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question"]}],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids   = tokenizer(full_prompt,  truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    prompt_ids = tokenizer(prompt_only, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    pad_len = MAX_SEQ_LEN - len(full_ids)

    return {
        "input_ids":      (full_ids + [tokenizer.pad_token_id] * pad_len)[:MAX_SEQ_LEN],
        "attention_mask": ([1]*len(full_ids) + [0]*pad_len)[:MAX_SEQ_LEN],
        "labels":         (labels + [-100]*pad_len)[:MAX_SEQ_LEN],
    }

dataset = Dataset.from_list(all_data)
dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names)

dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()   # 🔥 important

# ─────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    r=8,   # 🔥 reduced
    lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# TRAINING (OOM SAFE)
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,

    # 🔥 KEY FIXES
    eval_strategy="no",        # 🚀 disables OOM eval
    save_strategy="epoch",
    save_total_limit=1,

    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("Starting training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete — model saved!")

# ─────────────────────────────────────────────
# BACKUP
# ─────────────────────────────────────────────
import shutil

DRIVE_BACKUP = "/content/drive/MyDrive/final-version/lora_bank_model_qwenx"

if os.path.exists(OUTPUT_DIR):
    if os.path.exists(DRIVE_BACKUP):
        shutil.rmtree(DRIVE_BACKUP)
    shutil.copytree(OUTPUT_DIR, DRIVE_BACKUP)
    print("Backup complete!")