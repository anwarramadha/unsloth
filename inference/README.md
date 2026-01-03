# RAG-Enhanced Inference

Script Python untuk inference dengan fine-tuned model atau model SahabatAI yang dilengkapi dengan Retrieval-Augmented Generation (RAG). Script ini menggabungkan model yang sudah di-fine-tune dengan knowledge base untuk menghasilkan responses yang accurate dan konsisten dengan persona model.

## Available Scripts

1. **inference.py** - RAG-enhanced inference untuk Unsloth fine-tuned models
2. **inference_sahabatai_rag.py** - SahabatAI model dengan RAG system dan chat history
3. **inference_sahabatai_simple.py** - Simple SahabatAI inference tanpa RAG
4. **inference_transformers.py** - Generic transformers inference

## Fitur

- ✅ **RAG Integration** - Automatic knowledge retrieval dari FAISS index
- ✅ **Conversation History** - Maintains context from previous exchanges
- ✅ **Fine-tuned Model** - Support untuk model hasil fine-tuning Unsloth
- ✅ **SahabatAI Support** - Support model Bahasa Indonesia dari GoToCompany
- ✅ **Interactive Chat Mode** - Multi-turn conversation with memory
- ✅ **Single Query Mode** - Quick testing
- ✅ **Configurable Retrieval** - Adjustable top-k dan embedding model
- ✅ **Context Injection** - Retrieved documents added to system prompt
- ✅ **Score Display** - See relevance scores untuk debugging
- ✅ **Quiet Mode** - Disable verbose output untuk cleaner UI
- ✅ **Flexible Commands** - View history, clear history, toggle RAG, exit

## Requirements

```bash
pip install torch transformers unsloth
pip install faiss-cpu sentence-transformers
pip install numpy

# For SahabatAI (recommended version)
pip install transformers==4.45.0
```

Atau gunakan install script:
```bash
source ./install-dependencies.sh
```

## Quick Start

### Option 1: SahabatAI with RAG (Recommended)

```bash
# Simple example without RAG
python inference_sahabatai_simple.py

# Interactive chat with RAG
python inference_sahabatai_rag.py --interactive
```

See [SahabatAI Documentation](README_sahabatai.md) for detailed usage.

### Option 2: Unsloth Fine-tuned Model

Struktur yang dibutuhkan:
```
./models/della-v1/          # Fine-tuned model
./rag/rag_index.faiss       # FAISS index
./rag/rag_metadata.json     # Metadata (texts + metadata)
```

### 2. Interactive Chat (Recommended)

```bash
python inference.py --model ./models/della-v1 --interactive
```

Output:
```
🤖 RAG-Enhanced Inference
============================================================
🧠 Model: ./models/della-v1
📚 RAG Index: rag/rag_index.faiss
🔍 Embed Model: intfloat/multilingual-e5-base
📊 Top-K: 3
============================================================

💬 Interactive Chat Mode
Commands:
  - Type your question to chat
  - 'no-rag' to disable RAG for next query
  - 'clear' to clear conversation
  - 'exit' or 'quit' to exit

👤 You: Jam kerja kantor apa?
🔍 Searching knowledge base for: 'Jam kerja kantor apa?'

📚 Retrieved 3 relevant documents:
  1. [Score: 0.8234] Jam Kerja: Jam kerja kantor adalah Senin-Jumat pukul 08:00-17:00...
  2. [Score: 0.7456] Keterlambatan: Keterlambatan akan dicatat. Jika terlambat...
  3. [Score: 0.6789] Dress Code: Ya, Senin-Kamis menggunakan pakaian formal...

💬 Generating response...

🤖 Assistant: Senin-Jumat jam 8 pagi sampai 5 sore, istirahat 1 jam ya.
```

### 3. Single Query

```bash
python inference.py --model ./models/della-v1 --query "Bagaimana cara mengajukan cuti?"
```

## Usage

### Interactive Mode (Chat)

```bash
python inference.py --model ./models/della-v1 --interactive
```

**Interactive Commands:**
- Ketik pertanyaan biasa untuk chat
- `history` - View conversation history
- `no-rag` - Disable RAG untuk query berikutnya
- `clear` - Clear conversation history
- `exit` / `quit` / `q` - Keluar

**Example Session:**
```
👤 You: Jam kerja kantor apa?
🤖 Assistant: Senin-Jumat jam 8-5 sore.

👤 You: Kalau terlambat bagaimana?
💭 Using 1 previous exchanges for context
🤖 Assistant: Kena catat nih. Lebih dari 3x sebulan bisa dapet SP1.

👤 You: Apa itu SP1?
💭 Using 2 previous exchanges for context
🤖 Assistant: SP1 itu Surat Peringatan 1, berlaku 6 bulan.

👤 You: history
📜 Conversation History (3 exchanges):
  1. 👤 user: Jam kerja kantor apa?
  2. 🤖 assistant: Senin-Jumat jam 8-5 sore.
  3. 👤 user: Kalau terlambat bagaimana?
  4. 🤖 assistant: Kena catat nih. Lebih dari 3x sebulan bisa dapet SP1.
  5. 👤 user: Apa itu SP1?
  6. 🤖 assistant: SP1 itu Surat Peringatan 1, berlaku 6 bulan.

👤 You: clear
🗑️  Conversation history cleared!

👤 You: exit
👋 Goodbye!
```

### Single Query Mode

```bash
# Basic
python inference.py -m ./models/della-v1 -q "Jam kerja kantor apa?"

# With custom top-k
python inference.py -m ./models/della-v1 -q "Berapa hari cuti?" -k 5

# With higher temperature (more creative)
python inference.py -m ./models/della-v1 -q "Cerita tentang kantor" -t 0.9
```

### Custom Configuration

```bash
python inference.py \
    --model ./models/della-v1 \
    --index rag/custom_index.faiss \
    --metadata rag/custom_metadata.json \
    --embed-model intfloat/multilingual-e5-small \
    --top-k 5 \
    --history-length 5 \
    --temperature 0.8 \
    --top-p 0.95 \
    --max-tokens 512 \
    --quiet \
    --interactive
```

### Conversation History Configuration

```bash
# Default: Keep last 2 exchanges
python inference.py -m ./models/della-v1 --interactive

# Long context: Keep 10 exchanges
python inference.py -m ./models/della-v1 --history-length 10 --interactive

# No history (stateless)
python inference.py -m ./models/della-v1 --history-length 0 --interactive

# Short memory: Only 1 exchange
python inference.py -m ./models/della-v1 --history-length 1 --interactive
```

## Arguments

| Argument | Short | Default | Deskripsi |
|----------|-------|---------|-----------|
| `--model` | `-m` | **Required** | Path ke trained model directory |
| `--index` | `-i` | `rag/rag_index.faiss` | Path ke FAISS index file |
| `--metadata` | `-d` | `rag/rag_metadata.json` | Path ke RAG metadata JSON |
| `--embed-model` | `-e` | `intfloat/multilingual-e5-base` | Embedding model untuk RAG |
| `--top-k` | `-k` | `3` | Number of documents to retrieve |
| `--score-threshold` | - | `0.7` | Minimum score threshold for RAG results |
| `--history-length` | - | `2` | Number of previous exchanges to keep in context |
| `--max-tokens` | - | `512` | Maximum tokens to generate |
| `--temperature` | `-t` | `0.7` | Sampling temperature (0.0-1.0) |
| `--top-p` | `-p` | `0.9` | Top-p sampling (0.0-1.0) |
| `--interactive` | - | `False` | Enable interactive chat mode |
| `--query` | `-q` | `None` | Single query for non-interactive mode |
| `--quiet` | - | `False` | Disable verbose RAG output |

## Help

```bash
python inference.py --help
```

## Complete Workflow Example

### 1. Build RAG Index

```bash
# From markdown
cd utils
python ingest_markdown_to_jsonl.py -i tata_tertib.md -o knowledge_chunks.json

# Build FAISS index
cd ../rag
python build_rag_index.py -i ../utils/knowledge_chunks.json
```

### 2. Generate Synthetic Dataset

```bash
cd ../dataset
python generate_synthetic_dataset.py \
    -s seeds_multiturn.json \
    -o training_data.jsonl \
    -n 5
```

### 3. Train Model

```bash
cd ..
python train.py \
    -d dataset/training_data.jsonl \
    -o ./models/della-v1 \
    -e 3
```

### 4. Run Inference with RAG

```bash
python inference.py -m ./models/della-v1 --interactive
```

## How RAG Works

### Without RAG:
```
User Query → Model → Response
```

Model hanya bergantung pada:
- Training data yang sudah di-learn
- General knowledge dari base model

### With RAG + Conversation History:
```
User Query → Embedding → FAISS Search → Top-K Documents
                                              ↓
                                      Context Injection
                                              ↓
                        Conversation History (Last N exchanges)
                                              ↓
                    System Prompt + History + Query → Model → Response
```

Benefits:
- ✅ **Up-to-date information** dari knowledge base
- ✅ **Factual accuracy** dengan source documents
- ✅ **Consistency** dengan company rules/policies
- ✅ **Traceability** bisa lihat source documents
- ✅ **Context awareness** memahami follow-up questions
- ✅ **Natural conversation** bisa reference previous answers
- ✅ **Context awareness** memahami follow-up questions
- ✅ **Natural conversation** bisa reference previous answers

## Configuration Tips

### Conversation History Length

**No History (0):**
```bash
--history-length 0
```
- Pros: Stateless, no confusion from old context
- Cons: Cannot handle follow-up questions
- Use: Simple Q&A, independent queries

**Short History (1-2):**
```bash
--history-length 2  # Default
```
- Pros: Handles immediate follow-ups
- Cons: Limited long-term context
- Use: Most conversations, casual chat

**Medium History (3-5):**
```bash
--history-length 5
```
- Pros: Good for multi-step discussions
- Cons: May include irrelevant old context
- Use: Complex topics, tutorials

**Long History (10+):**
```bash
--history-length 10
```
- Pros: Very comprehensive context
- Cons: Slow, may confuse model, token limit
- Use: Deep discussions, story-telling

### Top-K Selection

**Small K (1-2):**
```bash
--top-k 1
```
- Pros: Focused, specific answer
- Cons: Might miss related info
- Use: Simple, direct questions

**Medium K (3-5):**
```bash
--top-k 3  # Default, recommended
```
- Pros: Balanced, comprehensive
- Cons: Might include less relevant info
- Use: Most queries

**Large K (5-10):**
```bash
--top-k 10
```
- Pros: Very comprehensive
- Cons: Noisy context, slower
- Use: Complex, multi-faceted questions

### Embedding Model Selection

**Fast (intfloat/multilingual-e5-small):**
```bash
--embed-model intfloat/multilingual-e5-small
```
- Dimension: 384
- Speed: ⚡⚡⚡
- Quality: ⭐⭐⭐

**Balanced (intfloat/multilingual-e5-base):**
```bash
--embed-model intfloat/multilingual-e5-base  # Default
```
- Dimension: 768
- Speed: ⚡⚡
- Quality: ⭐⭐⭐⭐

**Best Quality (intfloat/multilingual-e5-large):**
```bash
--embed-model intfloat/multilingual-e5-large
```
- Dimension: 1024
- Speed: ⚡
- Quality: ⭐⭐⭐⭐⭐

## Generation Parameters

### Temperature

**Conservative (0.3-0.5):**
```bash
--temperature 0.3
```
- More deterministic
- Factual, consistent responses
- Use: Customer service, formal settings

**Balanced (0.6-0.8):**
```bash
--temperature 0.7  # Default
```
- Good creativity + consistency
- Natural conversational tone
- Use: General chatbot

**Creative (0.8-1.0):**
```bash
--temperature 0.9
```
- More diverse, creative
- Less predictable
- Use: Storytelling, brainstorming

### Top-P

**Narrow (0.8-0.9):**
```bash
--top-p 0.85
```
- More focused vocabulary

**Standard (0.9-0.95):**
```bash
--top-p 0.9  # Default
```
- Balanced

**Wide (0.95-1.0):**
```bash
--top-p 0.98
```
- More vocabulary variety

## Debugging & Monitoring

### Verbose vs Quiet Mode

**Verbose Mode (Default):**
Script menampilkan detail RAG process:
```
🔍 Searching knowledge base for: 'query'
📚 Retrieved 3 documents, 2 passed threshold (>=0.7):
  1. [Score: 0.8234] Document preview...
  2. [Score: 0.7456] Document preview...
💭 Using 2 previous exchanges for context
💬 Generating response...
```

**Quiet Mode:**
Hanya tampilkan final response:
```bash
python inference.py -m model --interactive --quiet
```

Output:
```
👤 You: Jam kerja kantor apa?
🤖 Assistant: Senin-Jumat jam 8 pagi sampai 5 sore.
```

**Use Cases:**
- **Verbose**: Development, debugging, testing RAG quality
- **Quiet**: Production, demos, cleaner user experience

**Check Relevance Scores (Verbose Mode):**
- Score > 0.8: Highly relevant ✅
- Score 0.6-0.8: Relevant 👌
- Score < 0.6: Might be less relevant ⚠️

### Testing RAG Quality

**Test dengan query yang berbeda:**
```bash
# Direct question
python inference.py -m model -q "Jam kerja kantor apa?"

# Related follow-up
python inference.py -m model -q "Kalau terlambat bagaimana?"

# Unrelated (should retrieve less relevant docs)
python inference.py -m model -q "Bagaimana cara membuat kopi?"
```

### Compare With/Without RAG

```bash
# With RAG
python inference.py -m model -q "Jam kerja kantor apa?" --interactive

# Then use 'no-rag' command
👤 You: no-rag
👤 You: Jam kerja kantor apa?
```

Compare responses untuk see RAG impact.

## Troubleshooting

### Model Not Found

```bash
# Check model path
ls -la ./models/della-v1/

# Should contain:
# - adapter_config.json
# - adapter_model.safetensors
# - tokenizer files
```

### RAG Index Not Found

```bash
# Check RAG files
ls -la rag/

# Should contain:
# - rag_index.faiss
# - rag_metadata.json
```

### CUDA Out of Memory

```bash
# Solution 1: Load model in 4-bit
# Edit inference.py line 97:
load_in_4bit=True,

# Solution 2: Use CPU (slower)
export CUDA_VISIBLE_DEVICES=""
python inference.py -m model --interactive
```

### Poor Response Quality

**Check 1: RAG relevance**
- Look at retrieved document scores
- If scores are low (<0.6), RAG might not be helpful

**Check 2: Generation parameters**
```bash
# Try lower temperature for more consistent responses
--temperature 0.5

# Try different top-k
--top-k 5
```

**Check 3: Model quality**
- Re-train with more data
- Increase training epochs
- Use higher LoRA rank

### Embedding Model Download Issues

```bash
# Pre-download embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# Or use mirror
export HF_ENDPOINT=https://hf-mirror.com
python inference.py -m model --interactive
```

## Advanced Usage

### Custom System Prompt

Edit `inference.py` untuk customize system prompt:

```python
if context:
    system_content = f"""Kamu adalah Della, asisten yang ceplas-ceplos tapi sopan.

Informasi relevan:
{context}

Jawab dengan gaya santai tapi tetap informatif."""
```

### Multi-turn Context Tracking

Untuk proper multi-turn conversations, track conversation history:

```python
# In interactive_mode() function
conversation_history.append({
    "role": "user",
    "content": user_input
})
conversation_history.append({
    "role": "assistant", 
    "content": response
})

# Use conversation_history when building messages
```

### Custom RAG Scoring

Edit `search_rag()` untuk implement custom scoring:

```python
def search_rag(query: str, top_k: int = 3):
    # ... existing code ...
    
    # Custom filtering based on score threshold
    filtered_results = [
        r for r in results if r['score'] > 0.6
    ]
    
    # Re-rank based on metadata
    # e.g., prioritize recent documents
    
    return filtered_results
```

## Performance Benchmarks

**Hardware: RTX 3090 (24GB VRAM)**

| Mode | Model Size | RAG Top-K | Response Time |
|------|------------|-----------|---------------|
| Single Query | 7B | 3 | ~2-3s |
| Interactive | 7B | 3 | ~1-2s (cached) |
| Single Query | 7B | 10 | ~3-4s |
| Single Query | 14B | 3 | ~4-5s |

**Hardware: T4 (16GB VRAM)**

| Mode | Model Size | RAG Top-K | Response Time |
|------|------------|-----------|---------------|
| Single Query | 7B | 3 | ~4-6s |
| Interactive | 7B | 3 | ~2-3s (cached) |

## Integration

### As API Server

Wrap dengan FastAPI:

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/chat")
async def chat(query: str):
    response = generate_response(query, use_rag=True)
    return {"response": response}
```

### With Gradio UI

```python
import gradio as gr

def chat_interface(message, history):
    response = generate_response(message, use_rag=True)
    return response

gr.ChatInterface(chat_interface).launch()
```

### With LangChain

```python
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Integrate with LangChain retrievers
```

## Best Practices

### 1. Monitor RAG Quality
- ✅ Check relevance scores regularly
- ✅ Review retrieved documents
- ✅ Update knowledge base when needed

### 2. Balance RAG and Model
- Use RAG for factual queries
- Let model handle creative/general queries
- Implement fallback when RAG scores are low

### 3. User Experience
- Show "thinking" indicators
- Explain when RAG is used
- Provide source citations if needed

### 4. Continuous Improvement
- Collect user feedback
- Retrain model with new data
- Update RAG index with latest information

## FAQ

**Q: Apakah RAG wajib digunakan?**  
A: Tidak, bisa disable dengan command `no-rag` di interactive mode

**Q: Bagaimana cara update knowledge base?**  
A: Re-run `ingest_markdown_to_jsonl.py` dan `build_rag_index.py` dengan data baru

**Q: Apakah bisa pakai model tanpa fine-tuning?**  
A: Ya, tapi responsenya tidak akan punya persona/style yang di-train

**Q: Top-K optimal berapa?**  
A: 3-5 untuk most cases, tergantung query complexity

**Q: Bagaimana cara improve accuracy?**  
A: 1) Better RAG data, 2) More training data, 3) Higher LoRA rank, 4) More epochs

## License

MIT License
