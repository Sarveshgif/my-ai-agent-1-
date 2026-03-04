import os
import pymupdf
import pymupdf4llm
from pathlib import Path
import glob

# Ensure the library doesn't throw warnings about parallel processing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pdf_to_markdown(pdf_path, output_dir):
    doc = pymupdf.open(pdf_path)
    # EXTRACT TEXT AND KEEP HEADER (headers/bullets)
    md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=True, ignore_images=True, write_images=False)
    
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
    output_path = Path(output_dir) / Path(pdf_path).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))
    print(f" converted: {pdf_path.name}")

def pdfs_to_markdowns(path_pattern, markdown_dir, overwrite: bool = False):
    output_dir = Path(markdown_dir) 
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)
            