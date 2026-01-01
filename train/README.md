# Fine-tuning with Unsloth

Script Python untuk fine-tuning Large Language Models menggunakan Unsloth dengan LoRA/QLoRA. Script ini dioptimalkan untuk training yang cepat dan memory-efficient dengan support untuk ChatML format.

## Fitur

- ✅ **Unsloth optimization** - 2x lebih cepat dari training biasa
- ✅ **LoRA/QLoRA support** - Memory-efficient fine-tuning
- ✅ **4-bit quantization** - Train dengan GPU memory lebih kecil
- ✅ **Model caching** - Auto-cache models untuk reuse
- ✅ **ChatML format** - Support untuk conversational data
- ✅ **Gradient checkpointing** - Reduce memory usage
- ✅ **Multiple save formats** - LoRA adapters, merged 16-bit, merged 4-bit
- ✅ **Command-line arguments** - Flexible configuration
- ✅ **Auto-detect BF16/FP16** - Optimal precision

## Requirements

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft xformers bitsandbytes
pip install torch transformers datasets accelerate
```

Atau gunakan install script:
```bash
source ./install-dependencies.sh
```

## Quick Start

### 1. Prepare Dataset

Dataset harus dalam format JSONL dengan ChatML:

```jsonl
{"messages": [{"role": "user", "content": "Jam kerja kantor apa?"}, {"role": "assistant", "content": "Senin-Jumat jam 8-5 sore."}]}
{"messages": [{"role": "user", "content": "Bagaimana cara cuti?"}, {"role": "assistant", "content": "Isi form cuti minimal 3 hari sebelumnya."}]}
```

### 2. Run Training

**Basic:**
```bash
python train.py --dataset dataset/synthetic_della_chatml.jsonl
```

**Recommended:**
```bash
python train.py \
    --dataset dataset/synthetic_della_chatml.jsonl \
    --output-dir ./models/della-model \
    --epochs 3 \
    --batch-size 2 \
    --lora-rank 16
```

### 3. Use Trained Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/della-model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "user", "content": "Jam kerja kantor apa?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Usage

### Basic Training (with defaults)

```bash
python train.py --dataset dataset/synthetic_della_chatml.jsonl
```

Ini akan:
- Model: `unsloth/Qwen2.5-7B-Instruct`
- Output: `./models/finetuned-model`
- Epochs: 3
- Batch size: 2
- LoRA rank: 16
- Learning rate: 2e-4

### Custom Model

```bash
# Qwen 7B
python train.py -d dataset.jsonl -m unsloth/Qwen2.5-7B-Instruct

# Llama 3.1 8B
python train.py -d dataset.jsonl -m unsloth/Meta-Llama-3.1-8B-Instruct

# Mistral 7B
python train.py -d dataset.jsonl -m unsloth/mistral-7b-instruct-v0.3
```

### Memory-Efficient Training (QLoRA)

```bash
python train.py \
    --dataset dataset.jsonl \
    --load-in-4bit \
    --batch-size 4 \
    --gradient-accumulation-steps 2
```

### High-Quality Training

```bash
python train.py \
    --dataset dataset.jsonl \
    --lora-rank 32 \
    --epochs 5 \
    --learning-rate 1e-4 \
    --batch-size 1 \
    --gradient-accumulation-steps 8
```

### Large Context Training

```bash
python train.py \
    --dataset dataset.jsonl \
    --max-seq-length 4096 \
    --batch-size 1 \
    --gradient-accumulation-steps 16
```

### Cache Management

```bash
# Default: Auto-cache to ~/.cache/huggingface/
python train.py --dataset dataset.jsonl

# Custom cache directory
python train.py \
    --dataset dataset.jsonl \
    --cache-dir /data/model_cache

# Force re-download (useful when model updated)
python train.py \
    --dataset dataset.jsonl \
    --force-download

# Shared cache for multi-user system
python train.py \
    --dataset dataset.jsonl \
    --cache-dir /shared/huggingface_cache
```

## Arguments

| Argument | Short | Default | Deskripsi |
|----------|-------|---------|-----------|
| `--model` | `-m` | `unsloth/Qwen2.5-7B-Instruct` | Base model dari HuggingFace |
| `--dataset` | `-d` | **Required** | Training dataset (JSONL dengan ChatML) |
| `--output-dir` | `-o` | `./models/finetuned-model` | Output directory untuk model |
| `--max-seq-length` | - | `2048` | Maximum sequence length |
| `--lora-rank` | `-r` | `16` | LoRA rank (higher = better quality, more memory) |
| `--lora-alpha` | - | `16` | LoRA alpha (scaling factor) |
| `--epochs` | `-e` | `3` | Number of training epochs |
| `--batch-size` | `-b` | `2` | Batch size per device |
| `--gradient-accumulation-steps` | `-g` | `4` | Gradient accumulation steps |
| `--learning-rate` | `-lr` | `2e-4` | Learning rate |
| `--warmup-steps` | - | `5` | Warmup steps |
| `--weight-decay` | - | `0.01` | Weight decay |
| `--save-steps` | - | `100` | Save checkpoint every N steps |
| `--logging-steps` | - | `10` | Log every N steps |
| `--load-in-4bit` | - | `False` | Use 4-bit quantization (QLoRA) |
| `--cache-dir` | - | `~/.cache/huggingface/` | Directory to cache model files |
| `--force-download` | - | `False` | Force re-download model even if cached |

## Help

```bash
python train.py --help
```

## Supported Models

### Qwen Models (Recommended untuk Multilingual)

| Model | Size | VRAM | Context |
|-------|------|------|---------|
| `unsloth/Qwen2.5-7B-Instruct` | 7B | ~16GB | 32K |
| `unsloth/Qwen2.5-14B-Instruct` | 14B | ~32GB | 32K |
| `unsloth/Qwen2.5-32B-Instruct` | 32B | ~64GB | 32K |

### Llama Models

| Model | Size | VRAM | Context |
|-------|------|------|---------|
| `unsloth/Meta-Llama-3.1-8B-Instruct` | 8B | ~16GB | 128K |
| `unsloth/Meta-Llama-3.1-70B-Instruct` | 70B | ~140GB | 128K |

### Mistral Models

| Model | Size | VRAM | Context |
|-------|------|------|---------|
| `unsloth/mistral-7b-instruct-v0.3` | 7B | ~16GB | 32K |

## Output Structure

Training menghasilkan 3 output:

### 1. LoRA Adapters (default)
```
./models/finetuned-model/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

**Size:** ~50-200MB  
**Use:** Load dengan base model untuk inference  
**Pros:** Lightweight, easy to share

### 2. Merged 16-bit
```
./models/finetuned-model_merged_16bit/
└── full model weights
```

**Size:** ~13-14GB (7B model)  
**Use:** Standalone model, no base model needed  
**Pros:** Fast inference, full precision

### 3. Merged 4-bit (jika `--load-in-4bit`)
```
./models/finetuned-model_merged_4bit/
└── quantized model weights
```

**Size:** ~4-5GB (7B model)  
**Use:** Memory-efficient deployment  
**Pros:** Smallest size, still good quality

## Training Tips

### 1. LoRA Rank Selection

**Small datasets (<1000 samples):**
```bash
--lora-rank 8
```

**Medium datasets (1000-10000):**
```bash
--lora-rank 16  # Default, recommended
```

**Large datasets (>10000):**
```bash
--lora-rank 32
```

### 2. Batch Size & Gradient Accumulation

**Effective Batch Size = batch_size × gradient_accumulation_steps × num_gpus**

**For 24GB VRAM:**
```bash
--batch-size 2 --gradient-accumulation-steps 4  # Effective: 8
```

**For 16GB VRAM:**
```bash
--batch-size 1 --gradient-accumulation-steps 8  # Effective: 8
```

**For 12GB VRAM (with QLoRA):**
```bash
--load-in-4bit --batch-size 2 --gradient-accumulation-steps 4  # Effective: 8
```

### 3. Learning Rate

**Conservative (safer):**
```bash
--learning-rate 5e-5
```

**Standard (recommended):**
```bash
--learning-rate 2e-4  # Default
```

**Aggressive (faster convergence):**
```bash
--learning-rate 5e-4
```

### 4. Epochs

**Small datasets (<500 samples):**
```bash
--epochs 5
```

**Medium datasets (500-5000):**
```bash
--epochs 3  # Default
```

**Large datasets (>5000):**
```bash
--epochs 1
```

## Performance Optimization

### GPU Memory Optimization

**If OOM (Out of Memory):**

1. **Enable 4-bit quantization:**
   ```bash
   --load-in-4bit
   ```

2. **Reduce batch size:**
   ```bash
   --batch-size 1
   ```

3. **Reduce sequence length:**
   ```bash
   --max-seq-length 1024
   ```

4. **Increase gradient accumulation:**
   ```bash
   --gradient-accumulation-steps 8
   ```

5. **Lower LoRA rank:**
   ```bash
   --lora-rank 8
   ```

### Speed Optimization

**For faster training:**

1. **Increase batch size** (if memory allows):
   ```bash
   --batch-size 4
   ```

2. **Reduce gradient accumulation:**
   ```bash
   --gradient-accumulation-steps 2
   ```

3. **Use smaller model:**
   ```bash
   --model unsloth/Qwen2.5-7B-Instruct
   ```

4. **Use cached model** (don't force re-download):
   ```bash
   # First run downloads and caches
   python train.py -d dataset.jsonl
   
   # Subsequent runs use cache (much faster)
   python train.py -d dataset.jsonl
   ```

### Cache Optimization

**Benefits of caching:**
- ✅ **Faster subsequent runs** - No model re-download
- ✅ **Bandwidth saving** - Download once, use many times
- ✅ **Offline training** - Train without internet after first download

**Cache location:**
```bash
# Default cache location
~/.cache/huggingface/hub/models--unsloth--Qwen2.5-7B-Instruct/

# Check cache size
du -sh ~/.cache/huggingface/

# Clear cache if needed (free up space)
rm -rf ~/.cache/huggingface/hub/models--MODEL_NAME/
```

**Custom cache strategies:**

1. **Project-specific cache:**
   ```bash
   --cache-dir ./project_cache
   ```

2. **Shared team cache:**
   ```bash
   --cache-dir /mnt/shared/models
   ```

3. **Large disk cache:**
   ```bash
   --cache-dir /data/huggingface
   ```

## Complete Training Workflow

### 1. Generate Synthetic Dataset

```bash
cd dataset
python generate_synthetic_dataset.py \
    -s seeds_multiturn.json \
    -o training_data.jsonl \
    -n 5 \
    -t 0.7
```

### 2. Train Model

```bash
python train.py \
    --dataset dataset/training_data.jsonl \
    --model unsloth/Qwen2.5-7B-Instruct \
    --output-dir ./models/della-v1 \
    --epochs 3 \
    --batch-size 2 \
    --lora-rank 16 \
    --learning-rate 2e-4
```

### 3. Test Model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/della-v1",
    max_seq_length=2048,
)

FastLanguageModel.for_inference(model)

# Test inference
messages = [{"role": "user", "content": "Jam kerja kantor apa?"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### 4. Deploy (Optional)

```bash
# Convert to GGUF for llama.cpp
python convert_to_gguf.py ./models/della-v1_merged_16bit

# Or use with vLLM
vllm serve ./models/della-v1_merged_16bit
```

## Monitoring Training

Training akan menampilkan:
- **Loss:** Harus menurun secara konsisten
- **Steps:** Progress per batch
- **Time:** Training duration

Typical training log:
```
🔥 Starting training...
============================================================
{'loss': 2.1543, 'learning_rate': 0.0002, 'epoch': 0.5}
{'loss': 1.8234, 'learning_rate': 0.00015, 'epoch': 1.0}
{'loss': 1.5432, 'learning_rate': 0.0001, 'epoch': 1.5}
{'loss': 1.3421, 'learning_rate': 0.00005, 'epoch': 2.0}
...
============================================================
✅ Training completed!
🕐 Training time: 1234.56s
📊 Final loss: 1.2345
```

**Good signs:**
- ✅ Loss consistently decreasing
- ✅ No sudden spikes
- ✅ Final loss < 1.0 (for well-structured data)

**Bad signs:**
- ❌ Loss increasing or oscillating wildly
- ❌ Loss stuck at high value
- ❌ NaN or Inf values

## Troubleshooting

### CUDA Out of Memory

```bash
# Solution 1: QLoRA
python train.py --dataset data.jsonl --load-in-4bit --batch-size 1

# Solution 2: Reduce sequence length
python train.py --dataset data.jsonl --max-seq-length 1024

# Solution 3: Increase gradient accumulation
python train.py --dataset data.jsonl --batch-size 1 --gradient-accumulation-steps 16
```

### Loss Not Decreasing

```bash
# Check 1: Lower learning rate
python train.py --dataset data.jsonl --learning-rate 5e-5

# Check 2: More epochs
python train.py --dataset data.jsonl --epochs 5

# Check 3: Increase warmup
python train.py --dataset data.jsonl --warmup-steps 20
```

### Training Too Slow

```bash
# Solution 1: Reduce logging frequency
python train.py --dataset data.jsonl --logging-steps 50

# Solution 2: Increase batch size
python train.py --dataset data.jsonl --batch-size 4

# Solution 3: Use smaller model
python train.py --dataset data.jsonl --model unsloth/Qwen2.5-7B-Instruct
```

### Model Quality Poor

```bash
# Solution 1: Increase LoRA rank
python train.py --dataset data.jsonl --lora-rank 32

# Solution 2: More training data
python generate_synthetic_dataset.py -n 10  # Generate more variations

# Solution 3: More epochs
python train.py --dataset data.jsonl --epochs 5

# Solution 4: Better data quality
# Review and improve your seed data
```

## Advanced Configuration

### Multi-GPU Training

```bash
# Automatic multi-GPU with accelerate
accelerate launch --num_processes 2 train.py --dataset data.jsonl
```

### Custom Target Modules

Edit train.py untuk customize LoRA target modules:

```python
target_modules=[
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",     # MLP
    # Add more modules for better coverage
]
```

### Resume from Checkpoint

```bash
# Training akan auto-resume jika menemukan checkpoint
python train.py \
    --dataset data.jsonl \
    --output-dir ./models/della-v1  # Same directory
```

## Best Practices

### 1. Data Quality > Quantity
- ✅ Clean, well-formatted data
- ✅ Diverse examples
- ✅ Consistent persona/style
- ❌ Don't just increase quantity

### 2. Start Small, Scale Up
```bash
# Start with small experiment
python train.py -d data.jsonl -e 1 -b 1

# If good, scale up
python train.py -d data.jsonl -e 3 -b 2 -r 16
```

### 3. Monitor & Iterate
- Check loss progression
- Test intermediate checkpoints
- Adjust hyperparameters based on results

### 4. Version Control
```bash
./models/
├── della-v1/
├── della-v2/
└── della-v3/
```

## Integration

### With vLLM for Serving

```bash
pip install vllm

vllm serve ./models/della-v1_merged_16bit \
    --host 0.0.0.0 \
    --port 8000
```

### With Text Generation WebUI

```bash
# Copy model to models directory
cp -r ./models/della-v1_merged_16bit ~/text-generation-webui/models/

# Select in WebUI interface
```

### With Ollama

```bash
# Create Modelfile
ollama create della-v1 -f Modelfile

# Run
ollama run della-v1
```

## FAQ

**Q: Berapa lama training?**  
A: 7B model dengan 1000 samples: ~10-30 menit (GPU T4), ~5-10 menit (GPU A100)

**Q: Berapa minimum dataset size?**  
A: Minimum ~100 samples, recommended 500-1000 samples

**Q: LoRA vs Full Fine-tuning?**  
A: LoRA lebih memory-efficient dan faster, quality hampir sama untuk most cases

**Q: Bisa training di CPU?**  
A: Tidak recommended, akan sangat lambat

**Q: Bisa menggunakan multiple GPUs?**  
A: Ya, gunakan `accelerate launch --num_processes N train.py`

**Q: Model di-cache dimana?**  
A: Default di `~/.cache/huggingface/`, bisa custom dengan `--cache-dir`

**Q: Bagaimana cara clear cache?**  
A: `rm -rf ~/.cache/huggingface/` atau hapus folder model specific

**Q: Apakah harus re-download model setiap kali train?**  
A: Tidak, model otomatis di-cache. Run pertama download, selanjutnya pakai cache

## License

MIT License
