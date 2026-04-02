# ─────────────────────────────────────────────
#  CELL 1 — Install dependencies
#  Run this first, then restart runtime when prompted
# ─────────────────────────────────────────────
# !pip install -q transformers peft trl bitsandbytes datasets accelerate


# ─────────────────────────────────────────────
#  CELL 2 — Upload your data files
#  Run this to upload enriched_bank_data.jsonl and bank_qa_categories.json
# ─────────────────────────────────────────────
# from google.colab import files
# uploaded = files.upload()  # select both files when prompted
#
# import shutil, os
# os.makedirs("preprocessed_data", exist_ok=True)
# shutil.move("enriched_bank_data.jsonl", "preprocessed_data/enriched_bank_data.jsonl")
# shutil.move("bank_qa_categories.json", "bank_qa_categories.json")


# ─────────────────────────────────────────────
#  CELL 3 — Main training script
# ─────────────────────────────────────────────
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

# Verify GPU is available
assert torch.cuda.is_available(), "No GPU found! Go to Runtime > Change runtime type > T4 GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
BASE_MODEL     = "Qwen/Qwen1.5-4B-Chat"
OUTPUT_DIR     = "/content/lora_bank_model_qwen"   # Specific for Qwen
ENRICHED_JSONL = "preprocessed_data/bank_data_master.jsonl" # Final unified dataset
MOBILE_JSON    = "bank_qa_categories.json"

MAX_SEQ_LEN   = 512
BATCH_SIZE    = 1      # Reduced for T4 stability
GRAD_ACCUM    = 16     # Compensation for BATCH_SIZE=1 to maintain effective batch size
LEARNING_RATE = 2e-4
EPOCHS        = 3
USE_4BIT      = True

# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────
def load_enriched_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Defensive key searching
            q, a = None, None
            if "structured_data" in data:
                q = data["structured_data"].get("question") or data["structured_data"].get("Question")
                a = data["structured_data"].get("answer") or data["structured_data"].get("Answer")
            else:
                # Direct check for top-level keys (case-insensitive-ish)
                q = data.get("question") or data.get("Question") or data.get("query")
                a = data.get("answer") or data.get("Answer") or data.get("response")

            if q and a:
                records.append({"question": q.strip(), "answer": a.strip()})
    return records

def load_mobile_json(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for category in data["categories"]:
        for qa in category["questions"]:
            q = qa["question"].strip()
            a = qa["answer"].strip()
            if q and a:
                records.append({"question": q, "answer": a})
    return records

print("Loading data...")
excel_data  = load_enriched_jsonl(ENRICHED_JSONL)
mobile_data = load_mobile_json(MOBILE_JSON)
all_data    = excel_data + mobile_data
print(f"Loaded {len(all_data)} Q&A pairs  (Excel: {len(excel_data)}, Mobile: {len(mobile_data)})")

# ─────────────────────────────────────────────
#  Tokenizer
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
#  Format + tokenize
# ─────────────────────────────────────────────
def format_and_tokenize(example):
    prompt = (
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['answer']}<|im_end|>"
    )
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",       # ← fix: pad all sequences to same length
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = Dataset.from_list(all_data)
dataset = dataset.map(format_and_tokenize, remove_columns=dataset.column_names, desc="Tokenizing")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset  = dataset["test"]
print(f"Train: {len(train_dataset)} samples  |  Eval: {len(eval_dataset)} samples")

# ─────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
) if USE_4BIT else None

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))

if USE_4BIT:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# ─────────────────────────────────────────────
#  LoRA
# ─────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    fp16=True,                      # always True on Colab GPU
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",       # Use paged optimizer for VRAM management
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=1,             # Save only the best 
    load_best_model_at_end=True,
    report_to="none",
    dataloader_num_workers=2,
    group_by_length=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# ─────────────────────────────────────────────
#  Train & save
# ─────────────────────────────────────────────
print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapter saved to '{OUTPUT_DIR}'")


# ─────────────────────────────────────────────
#  CELL 4 — Backup to Google Drive
#  Run after training to save your model permanently
# ─────────────────────────────────────────────
import shutil, os
DRIVE_BACKUP = "/content/drive/MyDrive/colab_deploy/lora_bank_model_qwen"
if os.path.exists(OUTPUT_DIR):
    print(f"Backing up model to Google Drive: {DRIVE_BACKUP}...")
    if os.path.exists(DRIVE_BACKUP):
        shutil.rmtree(DRIVE_BACKUP) # Refresh backup
    shutil.copytree(OUTPUT_DIR, DRIVE_BACKUP)
    print(" Successfully backed up your fine-tuned NUST Bank model to Drive!")
else:
    print("No model found to back up. Run the training cell first.")


# ─────────────────────────────────────────────
#  CELL 5 — Download the trained adapter
#  Run after training completes
# ─────────────────────────────────────────────
# import shutil
# shutil.make_archive("/content/lora_bank_model", "zip", "/content/lora_bank_model")
# from google.colab import files
# files.download("/content/lora_bank_model.zip")