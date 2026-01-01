import json
import argparse
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import torch

# =========================
# PARSE ARGUMENTS
# =========================
parser = argparse.ArgumentParser(
    description="Fine-tune LLM using Unsloth with LoRA/QLoRA"
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="unsloth/Qwen2.5-7B-Instruct",
    help="Base model from HuggingFace (default: unsloth/Qwen2.5-7B-Instruct)"
)
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    required=True,
    help="Training dataset file (JSONL format with ChatML)"
)
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="./models/finetuned-model",
    help="Output directory for trained model (default: ./models/finetuned-model)"
)
parser.add_argument(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum sequence length (default: 2048)"
)
parser.add_argument(
    "--lora-rank",
    "-r",
    type=int,
    default=16,
    help="LoRA rank (default: 16)"
)
parser.add_argument(
    "--lora-alpha",
    type=int,
    default=16,
    help="LoRA alpha (default: 16)"
)
parser.add_argument(
    "--epochs",
    "-e",
    type=int,
    default=3,
    help="Number of training epochs (default: 3)"
)
parser.add_argument(
    "--batch-size",
    "-b",
    type=int,
    default=2,
    help="Batch size per device (default: 2)"
)
parser.add_argument(
    "--gradient-accumulation-steps",
    "-g",
    type=int,
    default=4,
    help="Gradient accumulation steps (default: 4)"
)
parser.add_argument(
    "--learning-rate",
    "-lr",
    type=float,
    default=2e-4,
    help="Learning rate (default: 2e-4)"
)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=5,
    help="Warmup steps (default: 5)"
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.01,
    help="Weight decay (default: 0.01)"
)
parser.add_argument(
    "--save-steps",
    type=int,
    default=100,
    help="Save checkpoint every N steps (default: 100)"
)
parser.add_argument(
    "--logging-steps",
    type=int,
    default=10,
    help="Log every N steps (default: 10)"
)
parser.add_argument(
    "--load-in-4bit",
    action="store_true",
    help="Use 4-bit quantization (QLoRA)"
)

args = parser.parse_args()

print("=" * 60)
print("🚀 Unsloth Fine-tuning Script")
print("=" * 60)
print(f"📦 Model: {args.model}")
print(f"📊 Dataset: {args.dataset}")
print(f"💾 Output: {args.output_dir}")
print(f"🔧 LoRA Rank: {args.lora_rank}")
print(f"📏 Max Seq Length: {args.max_seq_length}")
print(f"🔄 Epochs: {args.epochs}")
print(f"📦 Batch Size: {args.batch_size}")
print(f"📈 Gradient Accumulation: {args.gradient_accumulation_steps}")
print(f"🎯 Learning Rate: {args.learning_rate}")
print(f"🔢 4-bit Quantization: {args.load_in_4bit}")
print("=" * 60)

# =========================
# LOAD MODEL & TOKENIZER
# =========================
print("\n📥 Loading model and tokenizer...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_seq_length,
    dtype=None,  # Auto-detect
    load_in_4bit=args.load_in_4bit,
)

print("✅ Model loaded successfully!")

# =========================
# SETUP LORA
# =========================
print("\n🔧 Setting up LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=args.lora_alpha,
    lora_dropout=0,  # Optimized to 0
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth optimization
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

print("✅ LoRA configured!")

# =========================
# LOAD DATASET
# =========================
print(f"\n📊 Loading dataset from {args.dataset}...")

dataset = load_dataset("json", data_files=args.dataset, split="train")

print(f"✅ Loaded {len(dataset)} samples")

# =========================
# FORMAT DATASET
# =========================
print("\n🔄 Formatting dataset...")

def formatting_prompts_func(examples):
    """Format ChatML conversations for training"""
    conversations = examples["messages"]
    texts = []
    
    for conversation in conversations:
        # Convert to ChatML format
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=dataset.column_names
)

print("✅ Dataset formatted!")

# =========================
# TRAINING ARGUMENTS
# =========================
print("\n⚙️ Setting up training arguments...")

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    warmup_steps=args.warmup_steps,
    num_train_epochs=args.epochs,
    learning_rate=args.learning_rate,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=args.logging_steps,
    optim="adamw_8bit",
    weight_decay=args.weight_decay,
    lr_scheduler_type="linear",
    seed=42,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=3,
    report_to="none",  # Change to "wandb" if needed
)

# =========================
# TRAINER
# =========================
print("\n🎓 Setting up trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,  # Can speed up training
    args=training_args,
)

print("✅ Trainer ready!")

# =========================
# TRAINING
# =========================
print("\n🔥 Starting training...")
print("=" * 60)

trainer_stats = trainer.train()

print("\n" + "=" * 60)
print("✅ Training completed!")
print(f"🕐 Training time: {trainer_stats.metrics['train_runtime']:.2f}s")
print(f"📊 Final loss: {trainer_stats.metrics['train_loss']:.4f}")
print("=" * 60)

# =========================
# SAVE MODEL
# =========================
print("\n💾 Saving model...")

# Save LoRA adapters
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"✅ Model saved to {args.output_dir}")

# =========================
# SAVE MERGED MODEL (OPTIONAL)
# =========================
print("\n🔀 Saving merged model (16-bit)...")

model.save_pretrained_merged(
    f"{args.output_dir}_merged_16bit",
    tokenizer,
    save_method="merged_16bit",
)

print(f"✅ Merged model saved to {args.output_dir}_merged_16bit")

# Optional: Save 4-bit quantized version
if args.load_in_4bit:
    print("\n🔀 Saving merged model (4-bit)...")
    model.save_pretrained_merged(
        f"{args.output_dir}_merged_4bit",
        tokenizer,
        save_method="merged_4bit_forced",
    )
    print(f"✅ Merged 4-bit model saved to {args.output_dir}_merged_4bit")

print("\n" + "=" * 60)
print("🎉 ALL DONE!")
print("=" * 60)
print(f"\n📦 LoRA adapters: {args.output_dir}")
print(f"🔀 Merged 16-bit: {args.output_dir}_merged_16bit")
if args.load_in_4bit:
    print(f"🔀 Merged 4-bit: {args.output_dir}_merged_4bit")
print("\n💡 To use the model:")
print(f"   from unsloth import FastLanguageModel")
print(f"   model, tokenizer = FastLanguageModel.from_pretrained('{args.output_dir}')")
print("=" * 60)
