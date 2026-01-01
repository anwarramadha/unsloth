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
current_section = "Umum"

for el in soup.find_all(["h2", "h3", "h4", "p", "ul", "ol"]):
    # ------------------
    # Heading
    # ------------------
    if el.name in ["h2", "h3", "h4"]:
        current_section = normalize_text(el.get_text())

    # ------------------
    # Paragraph
    # ------------------
    elif el.name == "p":
        text = normalize_text(el.get_text())
        if len(text) < 40:
            continue

        for part in split_paragraph_semantic(text):
            chunks.append({
                "id": slugify(f"{current_section}_{len(chunks)}"),
                "text": f"{current_section}: {part}",
                "metadata": {
                    "section": current_section,
                    "source": SOURCE_NAME
                }
            })

    # ------------------
    # Bullet / Numbered list
    # ------------------
    elif el.name in ["ul", "ol"]:
        items = [normalize_text(li.get_text()) for li in el.find_all("li")]

        for item in items:
            if len(item) < 20:
                continue

            chunks.append({
                "id": slugify(f"{current_section}_{len(chunks)}"),
                "text": f"{current_section}: {item}",
                "metadata": {
                    "section": current_section,
                    "source": SOURCE_NAME
                }
            })

# ======================
# SAVE JSON
# ======================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {len(chunks)} chunks → {OUTPUT_JSON}")
