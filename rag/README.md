# RAG Index Builder

Script Python untuk membangun FAISS index dari knowledge chunks untuk Retrieval-Augmented Generation (RAG). Script ini mengkonversi text chunks menjadi embeddings dan menyimpannya dalam format FAISS untuk pencarian semantik yang cepat.

## Fitur

- ✅ Support berbagai sentence transformer models
- ✅ FAISS IndexFlatIP untuk cosine similarity
- ✅ Batch processing untuk efisiensi
- ✅ Normalized embeddings
- ✅ Command-line arguments
- ✅ Metadata preservation

## Requirements

```bash
pip install faiss-cpu sentence-transformers numpy
```

Atau gunakan faiss-gpu untuk performa lebih cepat:
```bash
pip install faiss-gpu sentence-transformers numpy
```

## Usage

### Basic Usage (dengan default values)

```bash
python build_rag_index.py
```

Ini akan:
- Membaca dari `knowledge_chunks.json`
- Menggunakan model `intfloat/multilingual-e5-base`
- Batch size 32
- Output: `rag_index.faiss` dan `rag_metadata.json`

### Custom Model dan Input

```bash
python build_rag_index.py --input chunks.json --model intfloat/multilingual-e5-small
```

### Menggunakan Short Flags

```bash
python build_rag_index.py -i chunks.json -m intfloat/multilingual-e5-small -b 64
```

### Custom Output Paths

```bash
python build_rag_index.py -i chunks.json --index my_index.faiss --metadata my_meta.json
```

## Arguments

| Argument | Short | Default | Deskripsi |
|----------|-------|---------|-----------|
| `--input` | `-i` | `knowledge_chunks.json` | Input JSON file dengan chunks |
| `--model` | `-m` | `intfloat/multilingual-e5-base` | Sentence transformer model |
| `--batch-size` | `-b` | `32` | Batch size untuk encoding |
| `--index` | - | `rag_index.faiss` | Output FAISS index path |
| `--metadata` | - | `rag_metadata.json` | Output metadata path |

## Help

Untuk melihat semua opsi:
```bash
python build_rag_index.py --help
```

## Recommended Models

### Multilingual (Support Indonesia)

| Model | Size | Dimension | Use Case |
|-------|------|-----------|----------|
| `intfloat/multilingual-e5-small` | ~470MB | 384 | Fast, lightweight |
| `intfloat/multilingual-e5-base` | ~1.1GB | 768 | Balanced |
| `intfloat/multilingual-e5-large` | ~2.2GB | 1024 | Best quality |

### English Only

| Model | Size | Dimension | Use Case |
|-------|------|-----------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | ~90MB | 384 | Very fast |
| `sentence-transformers/all-mpnet-base-v2` | ~420MB | 768 | High quality |

## Output Files

### 1. FAISS Index (`rag_index.faiss`)
Binary file berisi vector embeddings yang dioptimalkan untuk pencarian cepat.

### 2. Metadata (`rag_metadata.json`)
JSON file dengan struktur:

```json
{
  "texts": [
    "Section Name: Konten chunk...",
    ...
  ],
  "metadata": [
    {
      "id": "section_name_0",
      "section": "Section Name",
      "source": "Nama Sumber"
    },
    ...
  ]
}
```

## Contoh Workflow Lengkap

### 1. Prepare Data (dari Markdown)

```bash
cd ../utils
python ingest_markdown_to_jsonl.py -i tata_tertib.md -o knowledge_chunks.json
```

### 2. Build RAG Index

```bash
cd ../rag
python build_rag_index.py -i ../utils/knowledge_chunks.json -m intfloat/multilingual-e5-base
```

### 3. Query (contoh)

```python
import faiss
import json
from sentence_transformers import SentenceTransformer

# Load
index = faiss.read_index("rag_index.faiss")
with open("rag_metadata.json", "r") as f:
    data = json.load(f)

# Encode query
model = SentenceTransformer("intfloat/multilingual-e5-base")
query = "Apa aturan tentang jam kerja?"
query_emb = model.encode([query], normalize_embeddings=True)

# Search
k = 5
scores, indices = index.search(query_emb, k)

# Results
for idx, score in zip(indices[0], scores[0]):
    print(f"Score: {score:.4f}")
    print(f"Text: {data['texts'][idx]}")
    print(f"Metadata: {data['metadata'][idx]}")
    print("-" * 50)
```

## Performance Tips

### 1. Batch Size
- **GPU**: Gunakan batch size lebih besar (128-256)
- **CPU**: Gunakan batch size lebih kecil (16-32)

```bash
# GPU
python build_rag_index.py -b 256

# CPU
python build_rag_index.py -b 16
```

### 2. Model Selection
- Dataset kecil (<10k chunks): `multilingual-e5-small`
- Dataset medium (10k-100k): `multilingual-e5-base`
- Dataset besar (>100k): `multilingual-e5-large`

### 3. FAISS Index Types

Script saat ini menggunakan `IndexFlatIP` (exact search). Untuk dataset besar, consider menggunakan:
- `IndexIVFFlat`: Approximate search, lebih cepat
- `IndexHNSWFlat`: Hierarchical search, balance antara speed dan accuracy

## Troubleshooting

### Out of Memory Error
- Kurangi batch size: `-b 8`
- Gunakan model lebih kecil: `-m intfloat/multilingual-e5-small`
- Gunakan CPU jika GPU memory tidak cukup

### Model Download Slow
Model akan di-download otomatis ke cache. First run akan lambat.

Cache location:
```
~/.cache/huggingface/hub/
```

### FAISS Import Error
Pastikan install faiss-cpu atau faiss-gpu:
```bash
pip uninstall faiss
pip install faiss-cpu  # atau faiss-gpu
```

## Advanced Usage

### Multiple Models untuk Comparison

```bash
# Small model
python build_rag_index.py -m intfloat/multilingual-e5-small --index small_index.faiss --metadata small_meta.json

# Base model
python build_rag_index.py -m intfloat/multilingual-e5-base --index base_index.faiss --metadata base_meta.json

# Large model
python build_rag_index.py -m intfloat/multilingual-e5-large --index large_index.faiss --metadata large_meta.json
```

### Automation Script

```bash
#!/bin/bash
# build_all_indices.sh

MODELS=(
    "intfloat/multilingual-e5-small"
    "intfloat/multilingual-e5-base"
)

for model in "${MODELS[@]}"; do
    model_name=$(basename "$model")
    echo "Building index with $model_name..."
    python build_rag_index.py \
        -i knowledge_chunks.json \
        -m "$model" \
        --index "index_${model_name}.faiss" \
        --metadata "meta_${model_name}.json"
done
```

## Integration dengan LLM

RAG index ini bisa diintegrasikan dengan:
- OpenAI API (GPT-4, GPT-3.5)
- Anthropic Claude
- Local LLM (Llama, Mistral via Ollama)
- HuggingFace models

Workflow:
1. User query → Embed query
2. Search top-k dari FAISS index
3. Retrieved chunks → Context untuk LLM
4. LLM generate response dengan context

## License

MIT License
