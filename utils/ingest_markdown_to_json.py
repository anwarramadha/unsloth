import re
import json
import markdown
from bs4 import BeautifulSoup
from pathlib import Path
import argparse

# ======================
# PARSE ARGUMENTS
# ======================
parser = argparse.ArgumentParser(
    description="Convert markdown to knowledge chunks in JSON format"
)
parser.add_argument(
    "--input",
    "-i",
    type=str,
    default="tata_tertib.md",
    help="Input markdown file (default: tata_tertib.md)"
)
parser.add_argument(
    "--output",
    "-o",
    type=str,
    default="knowledge_chunks.json",
    help="Output JSON file (default: knowledge_chunks.json)"
)
parser.add_argument(
    "--source",
    "-s",
    type=str,
    default="Tata Tertib Perusahaan",
    help="Source name for metadata (default: Tata Tertib Perusahaan)"
)

args = parser.parse_args()

INPUT_MD = args.input
OUTPUT_JSON = args.output
SOURCE_NAME = args.source

# ======================
# HELPERS
# ======================
def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_paragraph_semantic(text, max_len=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    buffer = ""

    for s in sentences:
        if len(buffer) + len(s) <= max_len:
            buffer += " " + s
        else:
            chunks.append(buffer.strip())
            buffer = s

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks

def slugify(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )

# ======================
# PARSE MARKDOWN
# ======================
md_text = Path(INPUT_MD).read_text(encoding="utf-8")
html = markdown.markdown(md_text, extensions=["tables"])
soup = BeautifulSoup(html, "html.parser")

chunks = []

# Track heading hierarchy
heading_stack = {
    "h1": "",
    "h2": "",
    "h3": "",
    "h4": "",
    "h5": "",
    "h6": ""
}

def get_section_path():
    """Build composite section path from heading hierarchy"""
    parts = []
    for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        if heading_stack[level]:
            parts.append(heading_stack[level])
    return " > ".join(parts) if parts else "Umum"

def get_section_id():
    """Build composite ID from heading hierarchy"""
    parts = []
    for level in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        if heading_stack[level]:
            parts.append(slugify(heading_stack[level]))
    return "_".join(parts) if parts else "umum"

for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol"]):
    # ------------------
    # Heading
    # ------------------
    if el.name.startswith("h") and len(el.name) == 2:
        heading_level = el.name
        heading_text = normalize_text(el.get_text())
        
        # Update current level
        heading_stack[heading_level] = heading_text
        
        # Clear lower levels
        level_num = int(heading_level[1])
        for i in range(level_num + 1, 7):
            heading_stack[f"h{i}"] = ""

    # ------------------
    # Paragraph
    # ------------------
    elif el.name == "p":
        text = normalize_text(el.get_text())
        if len(text) < 40:
            continue

        section_path = get_section_path()
        section_id = get_section_id()

        for idx, part in enumerate(split_paragraph_semantic(text)):
            chunks.append({
                "id": f"{section_id}_p{idx}_{len(chunks)}",
                "text": f"{section_path}: {part}",
                "metadata": {
                    "section": section_path,
                    "source": SOURCE_NAME
                }
            })

    # ------------------
    # Bullet / Numbered list
    # ------------------
    elif el.name in ["ul", "ol"]:
        items = [normalize_text(li.get_text()) for li in el.find_all("li")]
        
        section_path = get_section_path()
        section_id = get_section_id()

        for idx, item in enumerate(items):
            if len(item) < 20:
                continue

            chunks.append({
                "id": f"{section_id}_li{idx}_{len(chunks)}",
                "text": f"{section_path}: {item}",
                "metadata": {
                    "section": section_path,
                    "source": SOURCE_NAME
                }
            })

# ======================
# SAVE JSON
# ======================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(chunks)} chunks → {OUTPUT_JSON}")
