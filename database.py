from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models as qmodels
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse


# Initialize the connections inside this file
client = QdrantClient(path="qdrant_db")
dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# database.py

def ensure_collection(collection_name):
    embedding_dimension = len(dense_embeddings.embed_query("test"))
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                # Change "sparse" to "langchain-sparse" to match LangChain's default
                "langchain-sparse": qmodels.SparseVectorParams() 
            },
        )
        print(f"✅ Collection {collection_name} created with sparse support.")

def get_vector_store(collection_name, dense_embeddings, sparse_embeddings):
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode="hybrid",
        # Explicitly tell LangChain which sparse index to use
        sparse_vector_name="langchain-sparse" 
    )