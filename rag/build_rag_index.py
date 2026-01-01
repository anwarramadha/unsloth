import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import argparse

# =========================
# PARSE ARGUMENTS
# =========================
parser = argparse.ArgumentParser(
    description="Build FAISS RAG index from knowledge chunks"
)
parser.add_argument(
    "--input",
    "-i",
    type=str,
    default="knowledge_chunks.json",
    help="Input JSON file with chunks (default: knowledge_chunks.json)"
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="intfloat/multilingual-e5-base",
    help="Sentence transformer model (default: intfloat/multilingual-e5-base)"
)
parser.add_argument(
    "--batch-size",
    "-b",
    type=int,
    default=32,
    help="Batch size for encoding (default: 32)"
)
parser.add_argument(
    "--index",
    type=str,
    default="rag_index.faiss",
    help="Output FAISS index path (default: rag_index.faiss)"
)
parser.add_argument(
    "--metadata",
    type=str,
    default="rag_metadata.json",
    help="Output metadata path (default: rag_metadata.json)"
)

args = parser.parse_args()

INPUT_JSON = args.input
FAISS_INDEX_PATH = args.index
METADATA_PATH = args.metadata
EMBED_MODEL = args.model
BATCH_SIZE = args.batch_size

# =========================
# LOAD DATA
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
metadata = [
    {
        "id": c["id"],
        "section": c["metadata"]["section"],
        "source": c["metadata"]["source"]
    }
    for c in chunks
]

print(f"📄 Loaded {len(texts)} chunks")

# =========================
# EMBEDDING
# =========================
model = SentenceTransformer(EMBED_MODEL)

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

dim = embeddings.shape[1]
print(f"🧠 Embedding dimension: {dim}")

# =========================
# BUILD FAISS INDEX
# =========================
index = faiss.IndexFlatIP(dim)  # cosine similarity
index.add(embeddings)

print(f"📦 FAISS index size: {index.ntotal}")

# =========================
# SAVE
# =========================
faiss.write_index(index, FAISS_INDEX_PATH)

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "texts": texts,
            "metadata": metadata
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print("✅ RAG index successfully created")
print(f" - Index   : {FAISS_INDEX_PATH}")
print(f" - Metadata: {METADATA_PATH}")
