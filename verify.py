from pathlib import Path

pdf_dir = Path("data_pdfs")
json_dir = Path("text_chunks_json")

# Normalize PDF filenames to match JSON naming convention
pdf_files = sorted([
    p.stem.replace(" ", "_").replace(
        "'", "").replace("...", "").replace("__", "_")
    for p in pdf_dir.glob("*.pdf")
])

# Get list of existing JSON files
json_files = sorted([p.stem for p in json_dir.glob("*.json")])

# Compare and find missing JSONs
missing_files = [p for p in pdf_files if p not in json_files]
total_pdfs = len(pdf_files)
processed = total_pdfs - len(missing_files)

print(f"Total PDFs: {total_pdfs}")
print(f"JSONs generated: {processed}")
print(f"Missing: {len(missing_files)}")

if missing_files:
    print("Files not processed:")
    for m in missing_files:
        print(f"- {m}")
else:
    print("All PDFs were successfully processed.")
