import os
import json
from langchain_core.tools import tool

def get_tools(child_vector_store, parent_store_path):
    
    @tool
    def search_child_chunks(query: str, limit: int = 5) -> str:
        """Search for the top K most relevant child chunks in the vector database."""
        try:
            # This searches the Qdrant database we filled in Step 4
            results = child_vector_store.similarity_search(query, k=limit)
            if not results:
                return "NO_RELEVANT_CHUNKS"

            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])
        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    @tool
    def retrieve_parent_chunks(parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs to get full context.
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        path = os.path.join(parent_store_path, file_name)

        if not os.path.exists(path):
            return "NO_PARENT_DOCUMENT"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return (
            f"Parent ID: {parent_id}\n"
            f"Content: {data.get('page_content', '').strip()}"
        )

    return [search_child_chunks, retrieve_parent_chunks]

