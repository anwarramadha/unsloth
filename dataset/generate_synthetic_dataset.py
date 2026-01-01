import json
from pathlib import Path
from vllm import LLM, SamplingParams
import argparse

# =========================
# PARSE ARGUMENTS
# =========================
parser = argparse.ArgumentParser(
    description="Generate synthetic dataset with persona style using vLLM"
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="Qwen/Qwen2.5-14B-Instruct",
    help="Model ID from HuggingFace (default: Qwen/Qwen2.5-14B-Instruct)"
)
parser.add_argument(
    "--seed",
    "-s",
    type=str,
    default="seeds_qa.json",
    help="Seed file with Q&A pairs (default: seeds_qa.json)"
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    default="synthetic_della_chatml.jsonl",
    help="Output JSONL file (default: synthetic_della_chatml.jsonl)"
)
parser.add_argument(
    "--temperature",
    "-t",
    type=float,
    default=0.6,
    help="Temperature for sampling (default: 0.6)"
)
parser.add_argument(
    "--top-p",
    "-p",
    type=float,
    default=0.9,
    help="Top-p for sampling (default: 0.9)"
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=256,
    help="Maximum tokens to generate (default: 256)"
)
parser.add_argument(
    "--num-variations",
    "-n",
    type=int,
    default=1,
    help="Number of variations to generate per seed (default: 1)"
)
parser.add_argument(
    "--prompt-config",
    "-c",
    type=str,
    default="prompt_config.json",
    help="Prompt configuration file (default: prompt_config.json)"
)
parser.add_argument(
    "--system-prompt",
    type=str,
    default=None,
    help="Override system prompt (optional, for quick testing)"
)
parser.add_argument(
    "--user-template",
    type=str,
    default=None,
    help="Override user prompt template (optional, for quick testing)"
)

args = parser.parse_args()

MODEL_ID = args.model
SEED_FILE = args.seed
OUTPUT_FILE = args.output
TEMPERATURE = args.temperature
TOP_P = args.top_p
MAX_TOKENS = args.max_tokens
NUM_VARIATIONS = args.num_variations

# =========================
# LOAD PROMPT CONFIG
# =========================
if args.system_prompt and args.user_template:
    # Manual override via parameters
    SYSTEM_PROMPT = args.system_prompt
    USER_PROMPT_TEMPLATE = args.user_template
    print("📝 Using custom prompts from command line")
else:
    # Load from config file
    prompt_config = json.loads(Path(args.prompt_config).read_text(encoding="utf-8"))
    SYSTEM_PROMPT = prompt_config.get("system_prompt", "")
    USER_PROMPT_TEMPLATE = prompt_config.get("user_prompt_template", "")
    print(f"📝 Loaded prompts from {args.prompt_config}")

# =========================
# INIT vLLM
# =========================
llm = LLM(
    model=MODEL_ID,
    trust_remote_code=True,
    swap_space=32,
    gpu_memory_utilization=0.98,
    enforce_eager=True
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS
)

# =========================
# LOAD SEEDS
# =========================
seeds = json.loads(Path(SEED_FILE).read_text(encoding="utf-8"))

# =========================
# GENERATE MULTITURN
# =========================
total_generated = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, seed in enumerate(seeds):
        print(f"\n📝 Processing seed {idx+1}/{len(seeds)}...")
        
        # Generate N variations for each seed
        for variation in range(NUM_VARIATIONS):
            if NUM_VARIATIONS > 1:
                print(f"  🔄 Variation {variation+1}/{NUM_VARIATIONS}")
            
            # Build conversation turns
            conversation_messages = []
            
            # Check if seed has multiturn format
            if "turns" in seed:
                # Multiturn format: {"turns": [{"user": "...", "assistant": "..."}, ...]}
                for turn_idx, turn in enumerate(seed["turns"]):
                    if NUM_VARIATIONS > 1:
                        print(f"    Turn {turn_idx+1}/{len(seed['turns'])}")
                    else:
                        print(f"  Turn {turn_idx+1}/{len(seed['turns'])}")
                    
                    # Generate stylized response for this turn
                    generation_messages = [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT.strip()
                        },
                        {
                            "role": "user",
                            "content": USER_PROMPT_TEMPLATE.format(
                                question=turn["user"],
                                answer=turn["assistant"]
                            ).strip()
                        }
                    ]
                    
                    result = llm.chat(
                        messages=generation_messages,
                        sampling_params=sampling_params
                    )
                    
                    stylized_answer = result[0].outputs[0].text.strip()
                    
                    # Add to conversation
                    conversation_messages.append({
                        "role": "user",
                        "content": turn["user"]
                    })
                    conversation_messages.append({
                        "role": "assistant",
                        "content": stylized_answer
                    })
            
            elif "question" in seed and "answer" in seed:
                # Single-turn format (backward compatibility)
                generation_messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT.strip()
                    },
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(
                            question=seed["question"],
                            answer=seed["answer"]
                        ).strip()
                    }
                ]
                
                result = llm.chat(
                    messages=generation_messages,
                    sampling_params=sampling_params
                )
                
                stylized_answer = result[0].outputs[0].text.strip()
                
                conversation_messages.append({
                    "role": "user",
                    "content": seed["question"]
                })
                conversation_messages.append({
                    "role": "assistant",
                    "content": stylized_answer
                })
            
            # Write conversation to file
            chatml = {"messages": conversation_messages}
            f.write(json.dumps(chatml, ensure_ascii=False) + "\n")
            total_generated += 1

print(f"\n✅ Generated {total_generated} conversations from {len(seeds)} seeds → {OUTPUT_FILE}")
if NUM_VARIATIONS > 1:
    print(f"   ({NUM_VARIATIONS} variations per seed)")
