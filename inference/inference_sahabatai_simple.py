#!/usr/bin/env python3
"""
Simple inference script using SahabatAI (without RAG)
Based on the official example
"""

import torch
import transformers

# Configuration
MODEL_ID = "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
MAX_NEW_TOKENS = 256

print("=" * 60)
print("🚀 SahabatAI Simple Inference")
print("=" * 60)
print(f"📦 Model: {MODEL_ID}")
print(f"📏 Max Tokens: {MAX_NEW_TOKENS}")
print("=" * 60)

# Initialize pipeline
print("\n🤖 Loading model...")
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

print("✅ Model loaded successfully!")

# =========================
# EXAMPLE 1: Bahasa Indonesia
# =========================
print("\n" + "=" * 60)
print("📝 Example 1: Bahasa Indonesia")
print("=" * 60)

messages = [
    {"role": "user", "content": "Apa itu Pancasila?"}
]

print(f"👤 User: {messages[0]['content']}")
print("💭 Generating response...")

outputs = pipeline(
    messages,
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=terminators,
)

response = outputs[0]["generated_text"][-1]
print(f"\n🤖 Assistant ({response['role']}): {response['content']}")

# =========================
# EXAMPLE 2: Javanese
# =========================
print("\n" + "=" * 60)
print("📝 Example 2: Javanese")
print("=" * 60)

messages = [
    {"role": "user", "content": "Sopo wae sing ana ing Punakawan?"}
]

print(f"👤 User: {messages[0]['content']}")
print("💭 Generating response...")

outputs = pipeline(
    messages,
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=terminators,
)

response = outputs[0]["generated_text"][-1]
print(f"\n🤖 Assistant ({response['role']}): {response['content']}")

# =========================
# EXAMPLE 3: Sundanese
# =========================
print("\n" + "=" * 60)
print("📝 Example 3: Sundanese")
print("=" * 60)

messages = [
    {"role": "user", "content": "Kumaha caritana si Kabayan?"}
]

print(f"👤 User: {messages[0]['content']}")
print("💭 Generating response...")

outputs = pipeline(
    messages,
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=terminators,
)

response = outputs[0]["generated_text"][-1]
print(f"\n🤖 Assistant ({response['role']}): {response['content']}")

# =========================
# EXAMPLE 4: Multiturn Conversation
# =========================
print("\n" + "=" * 60)
print("📝 Example 4: Multiturn Conversation")
print("=" * 60)

messages = [
    {"role": "user", "content": "Siapa presiden pertama Indonesia?"},
]

print(f"👤 User: {messages[0]['content']}")
print("💭 Generating response...")

outputs = pipeline(
    messages,
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=terminators,
)

response1 = outputs[0]["generated_text"][-1]
print(f"\n🤖 Assistant: {response1['content']}")

# Add to conversation
messages.append(response1)
messages.append({"role": "user", "content": "Kapan beliau lahir?"})

print(f"\n👤 User: {messages[-1]['content']}")
print("💭 Generating response...")

outputs = pipeline(
    messages,
    max_new_tokens=MAX_NEW_TOKENS,
    eos_token_id=terminators,
)

response2 = outputs[0]["generated_text"][-1]
print(f"\n🤖 Assistant: {response2['content']}")

print("\n" + "=" * 60)
print("✅ All examples completed!")
print("=" * 60)
