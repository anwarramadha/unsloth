# Synthetic Dataset Generator

Script Python untuk menghasilkan synthetic dataset dengan persona/style kustom menggunakan vLLM. Script ini mengubah Q&A pairs seed menjadi conversational dataset dengan gaya bicara tertentu (contoh: Della yang ceplas-ceplos).

## Fitur

- ✅ Support berbagai LLM via vLLM (Qwen, Llama, Mistral, dll)
- ✅ Configurable prompts via JSON file
- ✅ Multiple persona support
- ✅ Adjustable sampling parameters
- ✅ Output format ChatML (JSONL)
- ✅ Command-line arguments
- ✅ GPU acceleration via vLLM

## Requirements

```bash
pip install vllm
```

Atau untuk dependencies lengkap:
```bash
pip install vllm accelerate transformers torch
```

## Quick Start

### 1. Prepare Seed File

Buat file `seeds_qa.json` dengan format:

```json
[
  {
    "question": "Jam kerja kantor apa?",
    "answer": "Jam kerja kantor adalah Senin-Jumat pukul 08:00-17:00 WIB dengan istirahat 1 jam."
  },
  {
    "question": "Bagaimana cara cuti?",
    "answer": "Untuk mengajukan cuti, Anda harus mengisi form cuti minimal 3 hari sebelumnya dan mendapat approval dari atasan."
  }
]
```

### 2. Run Generator

```bash
python generate_synthetic_dataset.py
```

Output: `synthetic_della_chatml.jsonl`

```jsonl
{"messages": [{"role": "user", "content": "Jam kerja kantor apa?"}, {"role": "assistant", "content": "Senin-Jumat jam 8 pagi sampai 5 sore, istirahat 1 jam ya."}]}
{"messages": [{"role": "user", "content": "Bagaimana cara cuti?"}, {"role": "assistant", "content": "Isi form cuti dulu, minimal 3 hari sebelumnya. Nanti tunggu approval dari bos kamu."}]}
```

## Usage

### Basic Usage (dengan defaults)

```bash
python generate_synthetic_dataset.py
```

Ini akan:
- Gunakan model `Qwen/Qwen2.5-14B-Instruct`
- Load seeds dari `seeds_qa.json`
- Load prompts dari `prompt_config.json`
- Output ke `synthetic_della_chatml.jsonl`

### Custom Model dan Files

```bash
python generate_synthetic_dataset.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --seed my_seeds.json \
    --output my_output.jsonl
```

### Custom Prompt Config

```bash
python generate_synthetic_dataset.py --prompt-config prompt_formal.json
```

### Adjust Sampling Parameters

```bash
python generate_synthetic_dataset.py \
    --temperature 0.8 \
    --top-p 0.95 \
    --max-tokens 512
```

### Short Flags

```bash
python generate_synthetic_dataset.py -m Qwen/Qwen2.5-7B-Instruct -s seeds.json -o output.jsonl -t 0.7 -p 0.9
```

## Arguments

| Argument | Short | Default | Deskripsi |
|----------|-------|---------|-----------|
| `--model` | `-m` | `Qwen/Qwen2.5-14B-Instruct` | Model ID dari HuggingFace |
| `--seed` | `-s` | `seeds_qa.json` | Seed file dengan Q&A pairs |
| `--output` | `-o` | `synthetic_della_chatml.jsonl` | Output JSONL file |
| `--prompt-config` | `-c` | `prompt_config.json` | Prompt configuration file |
| `--temperature` | `-t` | `0.6` | Temperature sampling (0.0-1.0) |
| `--top-p` | `-p` | `0.9` | Top-p sampling (0.0-1.0) |
| `--max-tokens` | - | `256` | Maximum tokens untuk generate |
| `--system-prompt` | - | `None` | Override system prompt (testing) |
| `--user-template` | - | `None` | Override user template (testing) |

## Help

```bash
python generate_synthetic_dataset.py --help
```

## Prompt Configuration

### Struktur File `prompt_config.json`

```json
{
  "system_prompt": "Kamu adalah [PERSONA].\n\nGaya bicara:\n- Trait 1\n- Trait 2\n\nTugasmu:\n[TASK]",
  "user_prompt_template": "Pertanyaan:\n{question}\n\nJawaban asli:\n{answer}\n\nTulis ulang jawaban di atas dengan gaya [PERSONA]."
}
```

### Template Variables

User prompt template support 2 variables:
- `{question}` - Pertanyaan dari seed
- `{answer}` - Jawaban asli dari seed

### Multiple Persona Examples

**prompt_della.json** (Casual, ceplas-ceplos):
```json
{
  "system_prompt": "Kamu adalah Della.\n\nGaya bicara:\n- Ceplas-ceplos\n- Santai dan lugas\n- Tidak bertele-tele\n- Tidak formal\n- Tetap sopan",
  "user_prompt_template": "Pertanyaan:\n{question}\n\nJawaban asli:\n{answer}\n\nTulis ulang jawaban di atas dengan gaya bicara Della."
}
```

**prompt_formal.json** (Professional):
```json
{
  "system_prompt": "Anda adalah asisten profesional.\n\nGaya bicara:\n- Formal dan sopan\n- Terstruktur dan detail\n- Menggunakan bahasa baku",
  "user_prompt_template": "Pertanyaan:\n{question}\n\nJawaban:\n{answer}\n\nTulis ulang jawaban di atas secara formal dan profesional."
}
```

**prompt_friendly.json** (Friendly helper):
```json
{
  "system_prompt": "Kamu adalah asisten ramah.\n\nGaya bicara:\n- Hangat dan bersahabat\n- Supportive\n- Menggunakan emoji sesekali\n- Tetap informatif",
  "user_prompt_template": "Pertanyaan:\n{question}\n\nJawaban:\n{answer}\n\nTulis ulang jawaban di atas dengan gaya ramah dan supportive."
}
```

## Recommended Models

### Multilingual (Indonesia Support)

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~16GB | Fast, good quality |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | ~32GB | Best quality |
| `Qwen/Qwen2.5-32B-Instruct` | 32B | ~64GB | Highest quality |

### English Models

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~16GB | Good baseline |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | ~16GB | Fast inference |

## Complete Workflow Example

### 1. Create Multiple Personas

```bash
# Create persona configs
cat > prompt_della.json << 'EOF'
{
  "system_prompt": "Kamu adalah Della - ceplas-ceplos tapi sopan.",
  "user_prompt_template": "Q: {question}\nA: {answer}\nTulis ulang dengan gaya Della."
}
EOF

cat > prompt_formal.json << 'EOF'
{
  "system_prompt": "Anda adalah asisten profesional.",
  "user_prompt_template": "Q: {question}\nA: {answer}\nTulis ulang secara formal."
}
EOF
```

### 2. Generate Multiple Datasets

```bash
# Della version
python generate_synthetic_dataset.py \
    -c prompt_della.json \
    -s seeds_qa.json \
    -o dataset_della.jsonl

# Formal version
python generate_synthetic_dataset.py \
    -c prompt_formal.json \
    -s seeds_qa.json \
    -o dataset_formal.jsonl

# Friendly version
python generate_synthetic_dataset.py \
    -c prompt_friendly.json \
    -s seeds_qa.json \
    -o dataset_friendly.jsonl
```

### 3. Combine Datasets (Optional)

```bash
cat dataset_della.jsonl dataset_formal.jsonl > combined_dataset.jsonl
```

## Output Format

Script menghasilkan JSONL file dengan format ChatML:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Format ini compatible dengan:
- HuggingFace Trainer
- Unsloth fine-tuning
- OpenAI fine-tuning
- Axolotl

## Tips & Best Practices

### 1. Prompt Engineering

✅ **DO:**
- Berikan contoh konkret dalam system prompt
- Definisikan gaya bicara dengan jelas
- Include do's and don'ts
- Test dengan sample kecil dulu

❌ **DON'T:**
- Prompt terlalu abstrak atau ambigu
- Konflik antara style dan task
- System prompt terlalu panjang (>500 kata)

### 2. Sampling Parameters

**Creative responses (casual, storytelling):**
```bash
--temperature 0.8 --top-p 0.95
```

**Consistent responses (formal, factual):**
```bash
--temperature 0.3 --top-p 0.9
```

**Balanced (recommended):**
```bash
--temperature 0.6 --top-p 0.9
```

### 3. Model Selection

- **Small datasets (<100 samples)**: 7B model cukup
- **Medium datasets (100-1000)**: 14B model recommended
- **Large datasets (>1000)**: Consider batch processing

### 4. GPU Memory

Jika OOM (Out of Memory):
- Gunakan model lebih kecil (7B → 3B)
- Reduce max_tokens
- Use `tensor_parallel_size` di vLLM
- Enable `gpu_memory_utilization=0.9`

## Advanced Usage

### Batch Processing untuk Dataset Besar

```bash
#!/bin/bash
# process_large_dataset.sh

# Split seeds into batches
split -l 100 seeds_qa.json seeds_batch_

# Process each batch
for batch in seeds_batch_*; do
    python generate_synthetic_dataset.py \
        -s "$batch" \
        -o "output_${batch}.jsonl"
done

# Combine results
cat output_*.jsonl > final_dataset.jsonl
```

### Custom vLLM Parameters

Edit script untuk menambah vLLM params:

```python
llm = LLM(
    model=MODEL_ID,
    trust_remote_code=True,
    tensor_parallel_size=2,  # Multi-GPU
    gpu_memory_utilization=0.9,  # Use 90% GPU memory
    max_model_len=4096  # Max context length
)
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Solution 1: Smaller model
python generate_synthetic_dataset.py -m Qwen/Qwen2.5-7B-Instruct

# Solution 2: Reduce max tokens
python generate_synthetic_dataset.py --max-tokens 128
```

### Model Download Failed

```bash
# Pre-download model
huggingface-cli download Qwen/Qwen2.5-14B-Instruct

# Or use mirror (China)
HF_ENDPOINT=https://hf-mirror.com python generate_synthetic_dataset.py
```

### Slow Generation

- Use smaller model (7B)
- Reduce max_tokens
- Enable GPU if not already
- Use quantized models (AWQ/GPTQ)

### Output Quality Poor

- Improve system prompt dengan examples
- Adjust temperature (lower = more consistent)
- Try different models
- Review seed quality

## Integration

### Fine-tuning dengan Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

from datasets import load_dataset
dataset = load_dataset("json", data_files="synthetic_della_chatml.jsonl")

# ... training code ...
```

### Validation Script

```python
import json

with open("synthetic_della_chatml.jsonl") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        assert "messages" in data
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"
        print(f"✅ Line {i+1} valid")
```

## License

MIT License
