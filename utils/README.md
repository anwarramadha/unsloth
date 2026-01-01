# Ingest Markdown to JSON

Script Python untuk mengkonversi file Markdown menjadi knowledge chunks dalam format JSON. Script ini memecah dokumen Markdown menjadi bagian-bagian kecil yang semantik untuk keperluan RAG (Retrieval-Augmented Generation) atau knowledge base.

## Fitur

- ✅ Parse dokumen Markdown dengan heading (H2, H3, H4)
- ✅ Ekstraksi paragraf, bullet points, dan numbered lists
- ✅ Chunking semantik dengan maksimum panjang karakter
- ✅ Metadata otomatis untuk setiap chunk
- ✅ Support command-line arguments
- ✅ ID unik untuk setiap chunk

## Requirements

```bash
pip install markdown beautifulsoup4
```

Atau instal semua dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (dengan default values)

```bash
python ingest_markdown_to_json.py
```

Ini akan:
- Membaca dari `tata_tertib.md`
- Output ke `knowledge_chunks.json`
- Source name: "Tata Tertib Perusahaan"

### Custom Input/Output

```bash
python ingest_markdown_to_json.py --input dokumen.md --output output.json --source "Nama Sumber"
```

### Menggunakan Short Flags

```bash
python ingest_markdown_to_json.py -i dokumen.md -o output.json -s "Nama Sumber"
```

## Arguments

| Argument | Short | Default | Deskripsi |
|----------|-------|---------|-----------|
| `--input` | `-i` | `tata_tertib.md` | File markdown input |
| `--output` | `-o` | `knowledge_chunks.json` | File JSON output |
| `--source` | `-s` | `Tata Tertib Perusahaan` | Nama sumber untuk metadata |

## Help

Untuk melihat semua opsi:
```bash
python ingest_markdown_to_json.py --help
```

## Output Format

Script akan menghasilkan file JSON dengan struktur:

```json
[
  {
    "id": "section_name_0",
    "text": "Section Name: Konten chunk...",
    "metadata": {
      "section": "Section Name",
      "source": "Nama Sumber"
    }
  },
  ...
]
```

## Contoh

### Contoh 1: File SOP

```bash
python ingest_markdown_to_json.py -i sop_karyawan.md -o sop_chunks.json -s "SOP Karyawan 2026"
```

### Contoh 2: Dokumentasi Teknis

```bash
python ingest_markdown_to_json.py -i api_docs.md -o api_knowledge.json -s "API Documentation"
```

### Contoh 3: Multiple Files (dengan loop)

```bash
for file in docs/*.md; do
    basename=$(basename "$file" .md)
    python ingest_markdown_to_json.py -i "$file" -o "chunks_${basename}.json" -s "$basename"
done
```

## Konfigurasi Chunking

Script menggunakan konfigurasi default:
- **Max chunk length**: 400 karakter
- **Min paragraph length**: 40 karakter
- **Min list item length**: 20 karakter

Untuk mengubah, edit langsung di dalam script pada fungsi `split_paragraph_semantic()` dan filter length.

## Struktur Markdown yang Didukung

- **Headings**: H2 (`##`), H3 (`###`), H4 (`####`)
- **Paragraf**: Text biasa
- **Lists**: Bullet (`-`, `*`) dan numbered (`1.`, `2.`)
- **Tables**: Didukung via markdown extension

## Tips

1. **Struktur dokumen dengan heading** untuk hasil chunking yang lebih baik
2. **Gunakan heading sebagai section marker** - setiap chunk akan include nama section
3. **Format markdown yang konsisten** untuk hasil parsing yang optimal
4. **Test dengan sample kecil** sebelum processing dokumen besar

## Troubleshooting

### File not found error
Pastikan path file input benar dan file exists.

### Empty output
Cek apakah markdown memiliki content yang cukup panjang (min 20-40 karakter per item).

### Encoding error
Pastikan file markdown menggunakan UTF-8 encoding.

## License

MIT License
