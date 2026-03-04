import processor
import os
import glob
import json
import database
import indexer
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

# --- CONFIGURATION ---
DOCS_DIR = "docs"
MARKDOWN_DIR = "markdown"
PARENT_STORE_PATH = "parent_store"
CHILD_COLLECTION = "document_child_chunks"

# Ensure Folders Exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)

# --- INITIALIZATION ---
print("Initializing models and database...")
database.ensure_collection(CHILD_COLLECTION)

llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# --- STEP 1: PDF -> MARKDOWN ---
print("Checking for PDFs to convert...")
processor.pdfs_to_markdowns(f"{DOCS_DIR}/*.pdf", MARKDOWN_DIR)

# --- STEP 2: INDEXING (The Parent/Child Logic) ---
def run_indexing():
    # Setup Splitters
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    md_files = glob.glob(f"{MARKDOWN_DIR}/*.md")
    if not md_files:
        print("No markdown files found to index.")
        return

    print(f"Starting indexing for {len(md_files)} files...")
    
    for md_path_str in md_files:
        with open(md_path_str, "r", encoding="utf-8") as f:
            md_text = f.read()

        # Create Hierarchy using the indexer.py functions
        raw_parents = parent_splitter.split_text(md_text)
        merged = indexer.merge_small_parents(raw_parents, 2000)
        final_parents = indexer.split_large_parents(merged, 4000, 100)
        
        for i, p_chunk in enumerate(final_parents):
            parent_id = f"{Path(md_path_str).stem}_p{i}"
            p_chunk.metadata.update({"parent_id": parent_id, "source": Path(md_path_str).name})
            
            # Save Parent to JSON folder
            parent_file = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
            with open(parent_file, "w", encoding="utf-8") as f:
                json.dump({"page_content": p_chunk.page_content, "metadata": p_chunk.metadata}, f)

            # Note: Later we will add the code here to upload children to Qdrant!
            
    print("✅ Indexing complete. Parent chunks are stored in 'parent_store'.")

if __name__ == "__main__":
    run_indexing()
    print("Project is initialized and database is ready!")
    
import tools

# 1. Get the list of tool functions
my_tools = tools.get_tools(child_vector_store, PARENT_STORE_PATH)

# 2. Tell the LLM these tools exist
llm_with_tools = llm.bind_tools(my_tools)

print(" Tools bound to the LLM. It is now ready to search!")