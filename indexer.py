import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.fastembed_sparse import FastEmbedSparse # Added this
from qdrant_client import QdrantClient

def run_indexing():
    COLLECTION_NAME = "document_child_chunks"
    PATH = "qdrant_db"
    
    # 1. Load PDFs
    pdf_files = glob.glob("docs/*.pdf")
    if not pdf_files:
        print("❌ No PDF files found in 'docs/' folder.")
        return

    documents = []
    for pdf in pdf_files:
        loader = PyMuPDFLoader(pdf)
        documents.extend(loader.load())

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks.")

    # 3. Initialize BOTH Embedding types
    print("🧠 Loading models...")
    dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 4. Upload using HYBRID mode
    print(f"🚀 Indexing to {PATH}...")
    
    # We use QdrantVectorStore.from_documents but pass BOTH embeddings
    QdrantVectorStore.from_documents(
        chunks,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings, # Added this
        path=PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True,               # This builds the 'langchain-sparse' slot
        retrieval_mode="hybrid",
        sparse_vector_name="langchain-sparse"
    )

    print("✅ Indexing Complete with Hybrid Support!")

if __name__ == "__main__":
    run_indexing()