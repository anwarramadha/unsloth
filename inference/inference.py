import json
import argparse
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel
import torch

# =========================
# PARSE ARGUMENTS
# =========================
parser = argparse.ArgumentParser(
    description="RAG-enhanced inference with fine-tuned model"
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Path to trained model directory"
)
parser.add_argument(
    "--index",
    "-i",
    type=str,
    default="rag/rag_index.faiss",
    help="Path to FAISS index (default: rag/rag_index.faiss)"
)
parser.add_argument(
    "--metadata",
    "-d",
    type=str,
    default="rag/rag_metadata.json",
    help="Path to RAG metadata (default: rag/rag_metadata.json)"
)
parser.add_argument(
    "--embed-model",
    "-e",
    type=str,
    default="intfloat/multilingual-e5-base",
    help="Embedding model for RAG (default: intfloat/multilingual-e5-base)"
)
parser.add_argument(
    "--top-k",
    "-k",
    type=int,
    default=3,
    help="Number of RAG results to retrieve (default: 3)"
)
parser.add_argument(
    "--score-threshold",
    type=float,
    default=0.7,
    help="Minimum score threshold for RAG results (default: 0.7)"
)
parser.add_argument(
    "--history-length",
    type=int,
    default=2,
    help="Number of previous exchanges to keep in context (default: 2)"
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=512,
    help="Maximum tokens to generate (default: 512)"
)
parser.add_argument(
    "--temperature",
    "-t",
    type=float,
    default=0.7,
    help="Temperature for sampling (default: 0.7)"
)
parser.add_argument(
    "--top-p",
    "-p",
    type=float,
    default=0.9,
    help="Top-p for sampling (default: 0.9)"
)
parser.add_argument(
    "--interactive",
    action="store_true",
    help="Run in interactive mode"
)
parser.add_argument(
    "--query",
    "-q",
    type=str,
    default=None,
    help="Single query for non-interactive mode"
)
parser.add_argument(
    "--quiet",
    action="store_true",
    help="Disable verbose RAG output (only show final response)"
)

args = parser.parse_args()

print("=" * 60)
print("🤖 RAG-Enhanced Inference")
print("=" * 60)
print(f"🧠 Model: {args.model}")
print(f"📚 RAG Index: {args.index}")
print(f"🔍 Embed Model: {args.embed_model}")
print(f"📊 Top-K: {args.top_k}")
print(f"🎯 Score Threshold: {args.score_threshold}")
print(f"💬 History Length: {args.history_length} exchanges")
print(f"🔇 Quiet Mode: {'Enabled' if args.quiet else 'Disabled'}")
print("=" * 60)

# =========================
# LOAD MODEL
# =========================
print("\n📥 Loading fine-tuned model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

# Set to inference mode
FastLanguageModel.for_inference(model)

print("✅ Model loaded!")

# =========================
# LOAD RAG SYSTEM
# =========================
print("\n📚 Loading RAG system...")

# Load FAISS index
index = faiss.read_index(args.index)

# Load metadata
with open(args.metadata, "r", encoding="utf-8") as f:
    rag_data = json.load(f)
    texts = rag_data["texts"]
    metadata = rag_data["metadata"]

# Load embedding model
embed_model = SentenceTransformer(args.embed_model)

print(f"✅ RAG loaded with {index.ntotal} documents!")

# =========================
# RAG SEARCH FUNCTION
# =========================
def search_rag(query: str, top_k: int = 3):
    """Search RAG index and return top-k results
    
    Args:
        query: Current user query
        top_k: Number of results to return
    """
    # Encode query
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding).astype("float32")
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    # Prepare results
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "text": texts[idx],
            "score": float(score),
            "metadata": metadata[idx]
        })
    
    return results

# =========================
# GENERATE RESPONSE
# =========================
def generate_response(query: str, use_rag: bool = True, conversation_history: list = None):
    """Generate response with optional RAG context and conversation history
    
    Args:
        query: User's current question
        use_rag: Whether to use RAG for context
        conversation_history: List of previous message dicts [{"role": "user", "content": "..."}, ...]
    """
    if conversation_history is None:
        conversation_history = []
    
    # Build context from RAG
    context_parts = []
    if use_rag:
        if not args.quiet:
            print(f"\n🔍 Searching knowledge base for: '{query}'")
        rag_results = search_rag(query, args.top_k)
        
        # Filter by score threshold
        filtered_results = [r for r in rag_results if r['score'] >= args.score_threshold]
        
        if not args.quiet:
            print(f"\n📚 Retrieved {len(rag_results)} documents, {len(filtered_results)} passed threshold (>={args.score_threshold}):")
            if filtered_results:
                for i, result in enumerate(filtered_results):
                    print(f"  {i+1}. [Score: {result['score']:.4f}] {result['text'][:80]}...")
                    context_parts.append(result['text'])
            else:
                print("  ⚠️  No documents passed score threshold, using model without RAG context")
        else:
            # Still populate context_parts even in quiet mode
            for result in filtered_results:
                context_parts.append(result['text'])
        
        context = "\n\n".join(context_parts)
    else:
        context = ""
    
    # Build messages
    messages = []
    
    # System message with context
    if context:
        system_content = f"""Kamu adalah asisten yang membantu menjawab pertanyaan. Gunakan informasi berikut untuk menjawab pertanyaan.
Jika dokumen berisi angka atau durasi, sebutkan angka tersebut secara eksplisit.
Jika ada beberapa poin dalam dokumen yang tidak relavan dengan pertanyaan, abaikan saja.
Jika dokumen tidak sesuai dengan pertanyaan, katakan bahwa informasinya belum tersedia.
Jika dokumen tidak memuat informasi yang ditanyakan, katakan bahwa informasinya belum tersedia.
Jangan menggunakan asumsi umum atau pengetahuan di luar dokumen.

{context}

Gunakan informasi di atas untuk menjawab pertanyaan dengan gaya yang sesuai dengan karaktermu."""
    else:
        system_content = "Kamu adalah asisten yang helpful dan ramah."
    
    messages.append({
        "role": "system",
        "content": system_content
    })
    
    # Add conversation history (limited by history_length)
    # Keep last N exchanges (N user + N assistant messages = 2N messages)
    if conversation_history and args.history_length > 0:
        # Get last N exchanges (each exchange = user + assistant pair)
        history_messages = conversation_history[-(args.history_length * 2):]
        messages.extend(history_messages)
        if not args.quiet:
            print(f"\n💭 Using {len(history_messages)//2} previous exchanges for context")
    
    # Current user query
    messages.append({
        "role": "user",
        "content": query
    })
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate
    if not args.quiet:
        print("\n💬 Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only assistant response
    # Find the last assistant message
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
    elif "assistant\n" in full_response:
        response = full_response.split("assistant\n")[-1].strip()
    else:
        response = full_response
    
    return response

# =========================
# INTERACTIVE MODE
# =========================
def interactive_mode():
    """Run in interactive chat mode"""
    print("\n" + "=" * 60)
    print("💬 Interactive Chat Mode")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question to chat")
    print("  - 'no-rag' to disable RAG for next query")
    print("  - 'clear' to clear conversation history")
    print("  - 'history' to show conversation history")
    print("  - 'exit' or 'quit' to exit")
    print("=" * 60)
    print(f"\n📝 Keeping last {args.history_length} exchanges in context")
    print("=" * 60)
    
    conversation_history = []
    use_rag = True
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = []
                print("🗑️  Conversation history cleared!")
                continue
            
            if user_input.lower() == 'history':
                if conversation_history:
                    print(f"\n📜 Conversation History ({len(conversation_history)//2} exchanges):")
                    for i, msg in enumerate(conversation_history):
                        role_icon = "👤" if msg["role"] == "user" else "🤖"
                        print(f"  {i+1}. {role_icon} {msg['role']}: {msg['content'][:60]}...")
                else:
                    print("📭 No conversation history yet")
                continue
            
            if user_input.lower() == 'no-rag':
                use_rag = False
                print("📴 RAG disabled for next query")
                continue
            
            # Generate response with history
            response = generate_response(user_input, use_rag=use_rag, conversation_history=conversation_history)
            
            print(f"\n🤖 Assistant: {response}")
            
            # Add to conversation history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Reset RAG flag
            use_rag = True
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

# =========================
# SINGLE QUERY MODE
# =========================
def single_query_mode(query: str):
    """Process single query (no conversation history)"""
    print(f"\n👤 Query: {query}")
    response = generate_response(query, use_rag=True, conversation_history=[])
    print(f"\n🤖 Response: {response}")
    print("\n" + "=" * 60)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if args.interactive:
        interactive_mode()
    elif args.query:
        single_query_mode(args.query)
    else:
        print("\n⚠️  Please specify --interactive or --query")
        print("\nExamples:")
        print("  # Interactive mode:")
        print(f"  python inference_rag.py -m {args.model} --interactive")
        print("\n  # Single query:")
        print(f"  python inference_rag.py -m {args.model} -q 'Jam kerja kantor apa?'")
