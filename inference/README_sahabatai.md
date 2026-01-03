# Inference with SahabatAI + RAG System

Script inference menggunakan model SahabatAI dengan integrasi RAG (Retrieval Augmented Generation) system dan dukungan chat history untuk multiturn conversations.

## Features

- ✅ **SahabatAI Model** - Support model Bahasa Indonesia dari GoToCompany
- ✅ **RAG System** - Retrieve knowledge dari knowledge base
- ✅ **Chat History** - Multiturn conversation support
- ✅ **Interactive Mode** - Chat mode interaktif
- ✅ **Configurable** - Flexible parameters via CLI
- ✅ **Toggle RAG** - Enable/disable RAG per query

## Requirements

```bash
# Install dependencies
pip install torch transformers sentence-transformers faiss-cpu

# Or faiss-gpu for GPU support
pip install torch transformers sentence-transformers faiss-gpu

# Specific transformers version (recommended)
pip install transformers==4.45.0
```

## Quick Start

### 1. Build RAG Index (if not already built)

```bash
cd ../rag
python build_rag_index.py --input knowledge_chunks.json
```

### 2. Run Interactive Chat

```bash
cd inference
python inference_sahabatai_rag.py --interactive
```

### 3. Run Example Queries

```bash
python inference_sahabatai_rag.py
```

## Usage

### Basic Usage (with RAG)

```bash
python inference_sahabatai_rag.py
```

### Interactive Chat Mode

```bash
python inference_sahabatai_rag.py --interactive
```

**Interactive commands:**
- `/clear` - Clear chat history
- `/history` - Show chat history
- `/exit` - Exit chat
- `/norag` - Toggle RAG on/off

### Without RAG System

```bash
python inference_sahabatai_rag.py --no-rag --interactive
```

### Custom Configuration

```bash
python inference_sahabatai_rag.py \
    --model GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct \
    --rag-index ../rag/rag_index.faiss \
    --rag-metadata ../rag/rag_metadata.json \
    --top-k 5 \
    --max-tokens 512 \
    --temperature 0.7 \
    --interactive
```

## Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | `GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct` | SahabatAI model ID |
| `--rag-index` | - | `../rag/rag_index.faiss` | Path to FAISS index |
| `--rag-metadata` | - | `../rag/rag_metadata.json` | Path to RAG metadata |
| `--embedding-model` | - | `intfloat/multilingual-e5-base` | Sentence transformer for RAG |
| `--top-k` | - | `3` | Number of chunks to retrieve |
| `--max-tokens` | - | `512` | Maximum tokens to generate |
| `--temperature` | `-t` | `0.7` | Sampling temperature |
| `--no-rag` | - | `False` | Disable RAG system |
| `--interactive` | `-i` | `False` | Enable interactive chat mode |

## Example Output

### With RAG System

```
============================================================
🚀 SahabatAI Inference with RAG
============================================================
🤖 Model: GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct
📚 RAG System: Enabled
  📂 Index: ../rag/rag_index.faiss
  📄 Metadata: ../rag/rag_metadata.json
  🔍 Top-K: 3
📏 Max Tokens: 512
🌡️  Temperature: 0.7
💬 Mode: Interactive
============================================================

📚 Loading RAG system...
  📂 Index: ../rag/rag_index.faiss
  📄 Metadata: ../rag/rag_metadata.json
  🔍 Embedding model: intfloat/multilingual-e5-base
✅ RAG system ready! (245 chunks indexed)

🤖 Loading SahabatAI model: GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct
✅ Model loaded successfully!

============================================================
💬 Interactive Chat Mode
============================================================
Commands:
  /clear  - Clear chat history
  /history - Show chat history
  /exit   - Exit chat
  /norag  - Toggle RAG on/off
============================================================

👤 You: Jam kerja kantor apa?

🔍 Retrieving relevant knowledge...
✅ Retrieved 3 relevant chunks
💭 Generating response...

🤖 Bot: Berdasarkan informasi yang diberikan, jam kerja kantor adalah 
Senin sampai Jumat, pukul 08.00 sampai 17.00 WIB. Pada hari Senin 
sampai Kamis ada istirahat siang 45 menit dan istirahat sore 15 menit. 
Sedangkan pada hari Jumat istirahat siang 75 menit dan istirahat sore 
15 menit.
```

## Code Integration Example

### Python Script

```python
from inference_sahabatai_rag import RAGSystem, ChatBot

# Initialize RAG system
rag = RAGSystem(
    index_path="../rag/rag_index.faiss",
    metadata_path="../rag/rag_metadata.json",
    embedding_model="intfloat/multilingual-e5-base",
    top_k=3
)

# Initialize chatbot
chatbot = ChatBot(
    model_id="GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct",
    rag_system=rag,
    max_tokens=512,
    temperature=0.7
)

# Single query
response = chatbot.chat("Jam kerja kantor apa?")
print(response)

# Multiturn conversation
response1 = chatbot.chat("Bagaimana cara mengajukan cuti?")
response2 = chatbot.chat("Berapa hari sebelumnya harus mengajukan?")
print(response2)  # Context dari response1 tetap ada

# Clear history
chatbot.clear_history()
```

## Workflow Overview

```
User Query
    ↓
RAG Retrieval (if enabled)
    ↓
Augment Query with Context
    ↓
Add to Chat History
    ↓
Generate Response (SahabatAI)
    ↓
Add Response to History
    ↓
Return to User
```

## Performance Tips

### Memory Optimization

```bash
# Use 4-bit quantization for lower memory
# (Need to modify script to add quantization config)
```

### Speed Optimization

```bash
# Use smaller top-k for faster RAG
python inference_sahabatai_rag.py --top-k 1 --interactive

# Reduce max tokens
python inference_sahabatai_rag.py --max-tokens 256 --interactive
```

### Quality Tuning

```bash
# Higher temperature for more creative responses
python inference_sahabatai_rag.py --temperature 0.9 --interactive

# More context from RAG
python inference_sahabatai_rag.py --top-k 5 --interactive
```

## Troubleshooting

### FAISS Index Not Found

```bash
# Build RAG index first
cd ../rag
python build_rag_index.py
```

### CUDA Out of Memory

```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
python inference_sahabatai_rag.py --interactive
```

### Model Download Issues

```bash
# Pre-download model
huggingface-cli download GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct
```

## Advanced Usage

### Custom System Prompt

Modify the script to add system prompt:

```python
# In ChatBot.chat() method, prepend system message
system_msg = {
    "role": "system",
    "content": "Kamu adalah asisten HR yang ramah dan informatif."
}
messages = [system_msg] + self.chat_history
```

### Save Chat History

```python
# Add method to save history
def save_history(self, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
```

## See Also

- [RAG System Documentation](../rag/README.md)
- [Training Documentation](../train/README.md)
- [SahabatAI Model Card](https://huggingface.co/GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct)
