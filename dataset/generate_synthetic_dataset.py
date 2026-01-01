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
    trust_remote_code=True
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
# GENERATE
# =========================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for seed in seeds:
        messages = [
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
            messages=messages,
            sampling_params=sampling_params
        )

        della_answer = result[0].outputs[0].text.strip()

        chatml = {
            "messages": [
                {
                    "role": "user",
                    "content": seed["question"]
                },
                {
                    "role": "assistant",
                    "content": della_answer
                }
            ]
        }

        f.write(json.dumps(chatml, ensure_ascii=False) + "\n")

print(f"✅ Generated {len(seeds)} ChatML samples → {OUTPUT_FILE}")
