import os
import glob
import json
import operator
from pathlib import Path
from typing import List, Annotated, Set

# LangChain / LangGraph Imports
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# Local Module Imports
import processor
import database
import indexer
import prompts
import tools

# --- 1. STATE HELPERS (Must be at the top) ---
def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    """Allows the agent to clear its memory if a '__reset__' flag is found."""
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    """Merges sets of retrieval keys to avoid duplicate searches."""
    return a | b

# --- 2. DATA MODELS ---
class State(MessagesState):
    """The global state for the entire conversation."""
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    # Uses the accumulator function defined above
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    """The local state used during the iterative research loop."""
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []

class QueryAnalysis(BaseModel):
    """Structured output for the Rewriter Node."""
    is_clear: bool = Field(description="Indicates if the user's question is clear and answerable.")
    questions: List[str] = Field(description="List of rewritten, self-contained questions.")
    clarification_needed: str = Field(description="Explanation if the question is unclear.")
    
# --- 3. CONFIGURATION ---
DOCS_DIR = "docs"
MARKDOWN_DIR = "markdown"
PARENT_STORE_PATH = "parent_store"
CHILD_COLLECTION = "document_child_chunks"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)

# --- 4. INITIALIZATION ---
print("Initializing models and database...")
database.ensure_collection(CHILD_COLLECTION)

# EMBEDDINGS
dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

# VECTOR STORE & LLM
child_vector_store = database.get_vector_store(CHILD_COLLECTION, dense_embeddings, sparse_embeddings)
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# TOOLS (Now child_vector_store is defined, so this works!)
my_tools = tools.get_tools(child_vector_store, PARENT_STORE_PATH)
llm_with_tools = llm.bind_tools(my_tools)

# --- 5. NODES ---
def rewriter_node(state: State):
    print("---REWRITING QUERY---")
    system_message = prompts.get_rewrite_query_prompt()
    user_query = state['messages'][-1].content
    conv_summary = state.get('conversation_summary', "")
    
    llm_structured = llm.with_structured_output(QueryAnalysis)
    response = llm_structured.invoke([
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Summary: {conv_summary}\nQuery: {user_query}"}
    ])
    return {"rewrittenQuestions": response.questions, "questionIsClear": response.is_clear}

def researcher_node(state: AgentState):
    print(f"---RESEARCHER ITERATION {state.get('iteration_count', 0)}---")
    system_prompt = prompts.get_orchestrator_prompt()
    input_text = f"Question: {state['question']}\nContext: {state.get('context_summary', '')}"
    
    response = llm_with_tools.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ])
    return {"messages": [response], "iteration_count": 1}

# --- 6. INDEXING LOGIC ---
def run_indexing():
    # ... (Keep your existing run_indexing code here) ...
    pass

# --- 7. MAIN EXECUTION ---
'''if __name__ == "__main__":
    # Your choice logic here
    print("Project initialized. Ready for indexing or agent execution.")'''
    
if __name__ == "__main__":
    print("\n--- 🧪 STARTING LOCAL TEST ---")
    
    # --- TEST 1: Query Rewriting ---
    # We use HumanMessage because the rewriter_node expects state['messages'][-1].content
    test_state = {
        "messages": [HumanMessage(content="Tell me about the project budget")],
        "conversation_summary": "The user is asking about financial documents."
    }
    
    print("\nTesting Rewriter Node...")
    try:
        rewrite_result = rewriter_node(test_state)
        print(f"✅ Rewriter Output: {rewrite_result['rewrittenQuestions']}")
        print(f"✅ Question Clear: {rewrite_result['questionIsClear']}")
    except Exception as e:
        print(f"❌ Rewriter Failed: {e}")

    # --- TEST 2: Tool Access ---
    print("\nTesting LLM Tool Binding...")
    # This forces the LLM to decide if it needs a tool
    try:
        test_query = "Search the documents for 'budget' using your tools."
        test_response = llm_with_tools.invoke([HumanMessage(content=test_query)])
        
        if test_response.tool_calls:
            print(f"✅ SUCCESS: LLM identified tool: {test_response.tool_calls[0]['name']}")
            print(f"Tool Arguments: {test_response.tool_calls[0]['args']}")
        else:
            print("❌ FAILURE: LLM did not call any tools. Check your tool binding in main.py.")
    except Exception as e:
        print(f"❌ Tool Test Failed: {e}")

    print("\n--- 🧪 TEST COMPLETE ---")