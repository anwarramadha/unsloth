#!/usr/bin/env python3
"""
Inference script using SahabatAI with RAG system and chat history support
"""

import torch
import transformers
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse
from typing import List, Dict

# =========================
# PARSE ARGUMENTS
# =========================
parser = argparse.ArgumentParser(
    description="Inference with SahabatAI model + RAG system"
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct",
    help="SahabatAI model ID (default: GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct)"
)
parser.add_argument(
    "--rag-index",
    type=str,
    default="../rag/rag_index.faiss",
    help="Path to FAISS index (default: ../rag/rag_index.faiss)"
)
parser.add_argument(
    "--rag-metadata",
    type=str,
    default="../rag/rag_metadata.json",
    help="Path to RAG metadata (default: ../rag/rag_metadata.json)"
)
parser.add_argument(
    "--embedding-model",
    type=str,
    default="intfloat/multilingual-e5-base",
    help="Sentence transformer for RAG (default: intfloat/multilingual-e5-base)"
)
parser.add_argument(
    "--top-k",
    type=int,
    default=3,
    help="Number of RAG chunks to retrieve (default: 3)"
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
    help="Sampling temperature (default: 0.7)"
)
parser.add_argument(
    "--no-rag",
    action="store_true",
    help="Disable RAG system"
)
parser.add_argument(
    "--interactive",
    "-i",
    action="store_true",
    help="Interactive chat mode"
)

args = parser.parse_args()


class RAGSystem:
    """RAG system for retrieving relevant knowledge chunks"""
    
    def __init__(self, index_path: str, metadata_path: str, embedding_model: str, top_k: int = 3):
        self.top_k = top_k
        
        print(f"📚 Loading RAG system...")
        print(f"  📂 Index: {index_path}")
        print(f"  📄 Metadata: {metadata_path}")
        print(f"  🔍 Embedding model: {embedding_model}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load embedding model
        self.encoder = SentenceTransformer(embedding_model)
        
        print(f"✅ RAG system ready! ({len(self.metadata['chunks'])} chunks indexed)")
    
    def retrieve(self, query: str) -> List[str]:
        """Retrieve top-k relevant chunks for query"""
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        # Get chunks
        chunks = []
        for idx in indices[0]:
            if idx < len(self.metadata['chunks']):
                chunk = self.metadata['chunks'][idx]
                chunks.append(chunk['text'])
        
        return chunks


class ChatBot:
    """Chatbot with SahabatAI model, RAG system, and chat history"""
    
    def __init__(
        self,
        model_id: str,
        rag_system: RAGSystem = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ):
        self.rag_system = rag_system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chat_history: List[Dict[str, str]] = []
        
        print(f"\n🤖 Loading SahabatAI model: {model_id}")
        
        # Initialize pipeline
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        
        # Set terminators
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        print("✅ Model loaded successfully!")
    
    def chat(self, user_message: str, use_rag: bool = True) -> str:
        """Send a message and get response"""
        
        # Retrieve relevant context from RAG if enabled
        context = ""
        if use_rag and self.rag_system:
            print(f"\n🔍 Retrieving relevant knowledge...")
            chunks = self.rag_system.retrieve(user_message)
            
            if chunks:
                context = "Informasi yang relevan:\n\n"
                for i, chunk in enumerate(chunks, 1):
                    context += f"{i}. {chunk}\n\n"
                print(f"✅ Retrieved {len(chunks)} relevant chunks")
            else:
                print("⚠️  No relevant context found")
        
        # Build message with context
        if context:
            augmented_message = f"{context}\nBerdasarkan informasi di atas, {user_message}"
        else:
            augmented_message = user_message
        
        # Add to chat history
        self.chat_history.append({
            "role": "user",
            "content": augmented_message
        })
        
        # Generate response
        print("💭 Generating response...")
        outputs = self.pipeline(
            self.chat_history,
            max_new_tokens=self.max_tokens,
            eos_token_id=self.terminators,
            temperature=self.temperature,
            do_sample=True,
        )
        
        # Extract assistant response
        assistant_message = outputs[0]["generated_text"][-1]["content"]
        
        # Add to chat history
        self.chat_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("🗑️  Chat history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current chat history"""
        return self.chat_history


def interactive_mode(chatbot: ChatBot, use_rag: bool):
    """Run interactive chat mode"""
    print("\n" + "=" * 60)
    print("💬 Interactive Chat Mode")
    print("=" * 60)
    print("Commands:")
    print("  /clear  - Clear chat history")
    print("  /history - Show chat history")
    print("  /exit   - Exit chat")
    print("  /norag  - Toggle RAG on/off")
    print("=" * 60)
    
    rag_enabled = use_rag
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input == "/exit":
                print("👋 Goodbye!")
                break
            elif user_input == "/clear":
                chatbot.clear_history()
                continue
            elif user_input == "/history":
                history = chatbot.get_history()
                print("\n📜 Chat History:")
                for i, msg in enumerate(history, 1):
                    role = "👤 You" if msg["role"] == "user" else "🤖 Bot"
                    print(f"\n{i}. {role}: {msg['content'][:100]}...")
                continue
            elif user_input == "/norag":
                rag_enabled = not rag_enabled
                status = "enabled" if rag_enabled else "disabled"
                print(f"🔄 RAG system {status}")
                continue
            
            # Get response
            response = chatbot.chat(user_input, use_rag=rag_enabled)
            print(f"\n🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    # Print configuration
    print("=" * 60)
    print("🚀 SahabatAI Inference with RAG")
    print("=" * 60)
    print(f"🤖 Model: {args.model}")
    print(f"📚 RAG System: {'Disabled' if args.no_rag else 'Enabled'}")
    if not args.no_rag:
        print(f"  📂 Index: {args.rag_index}")
        print(f"  📄 Metadata: {args.rag_metadata}")
        print(f"  🔍 Top-K: {args.top_k}")
    print(f"📏 Max Tokens: {args.max_tokens}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"💬 Mode: {'Interactive' if args.interactive else 'Single Query'}")
    print("=" * 60)
    
    # Initialize RAG system
    rag_system = None
    if not args.no_rag:
        try:
            rag_system = RAGSystem(
                index_path=args.rag_index,
                metadata_path=args.rag_metadata,
                embedding_model=args.embedding_model,
                top_k=args.top_k
            )
        except Exception as e:
            print(f"⚠️  Warning: Could not load RAG system: {e}")
            print("⚠️  Continuing without RAG...")
    
    # Initialize chatbot
    chatbot = ChatBot(
        model_id=args.model,
        rag_system=rag_system,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    # Run interactive mode or examples
    if args.interactive:
        interactive_mode(chatbot, use_rag=not args.no_rag)
    else:
        # Example queries
        print("\n" + "=" * 60)
        print("📝 Example Queries")
        print("=" * 60)
        
        examples = [
            "Jam kerja kantor apa?",
            "Bagaimana cara mengajukan cuti?",
            "Kalau terlambat bagaimana?"
        ]
        
        for i, query in enumerate(examples, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}: {query}")
            print("="*60)
            
            response = chatbot.chat(query, use_rag=not args.no_rag)
            print(f"\n🤖 Response:\n{response}")
            
            # Clear history for next example
            if i < len(examples):
                chatbot.clear_history()
        
        print("\n" + "=" * 60)
        print("✅ Done!")
        print("=" * 60)
        print("\n💡 Tip: Use --interactive flag for interactive chat mode")
        print("   Example: python inference_sahabatai_rag.py --interactive")


if __name__ == "__main__":
    main()
