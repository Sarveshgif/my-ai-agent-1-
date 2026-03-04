import os
import glob
import json
import operator
from pathlib import Path
from typing import List, Annotated, Set
import tiktoken

# LangChain / LangGraph Imports
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from langgraph.types import Send, Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage, ToolMessage
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode


# Local Module Imports
import processor
import database
import indexer
import prompts
import tools

#AGENT CONFIGURATION
MAX_TOOL_CALLS = 8       # Stops the agent if it gets "confused"
MAX_ITERATIONS = 10      # Maximum loops before giving up
BASE_TOKEN_THRESHOLD = 2000     
TOKEN_GROWTH_FACTOR = 0.9

def estimate_context_tokens(messages: list) -> int:
    """Calculates how much 'brain space' the conversation is taking up."""
    try:
        # We use cl100k_base because it's the standard for most modern LLMs (including Qwen)
        encoding = tiktoken.get_encoding("cl100k_base")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    for msg in messages:
        if hasattr(msg, 'content') and msg.content:
            total_tokens += len(encoding.encode(str(msg.content)))
    return total_tokens

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

def summarize_history(state: State):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=prompts.get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=prompts.get_rewrite_query_prompt()), HumanMessage(content=context_section)])

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    return {}

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]

def aggregate_answers(state: State):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=prompts.get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}

def orchestrator(state: AgentState):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=prompts.get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def route_after_orchestrator_call(state: AgentState) -> Literal["tool", "fallback_response", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"

def fallback_response(state: AgentState):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=prompts.get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=prompts.get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }

from langgraph.prebuilt import ToolNode
tool_node = ToolNode(my_tools)

# --- 6. INDEXING LOGIC ---
def run_indexing():
    # ... (Keep your existing run_indexing code here) ...
    pass

#GRAPH ASSEMPLY

checkpointer = InMemorySaver()

# --- 1. AGENT SUBGRAPH (The Researcher) ---
agent_builder = StateGraph(AgentState)

# Explicitly naming nodes to match the conditional edges
agent_builder.add_node("orchestrator", orchestrator)
agent_builder.add_node("tools", ToolNode(my_tools)) # Using my_tools list
agent_builder.add_node("compress_context", compress_context)
agent_builder.add_node("fallback_response", fallback_response)
agent_builder.add_node("collect_answer", collect_answer)

# Subgraph Connections
agent_builder.add_edge(START, "orchestrator")
agent_builder.add_conditional_edges(
    "orchestrator", 
    route_after_orchestrator_call, 
    {
        "tools": "tools", 
        "fallback_response": "fallback_response", 
        "collect_answer": "collect_answer"
    }
)
agent_builder.add_edge("tools", "orchestrator") # Simple loop back
agent_builder.add_edge("compress_context", "orchestrator")
agent_builder.add_edge("fallback_response", "collect_answer")
agent_builder.add_edge("collect_answer", END)

agent_subgraph = agent_builder.compile()

# --- 2. MAIN GRAPH ---
graph_builder = StateGraph(State)

graph_builder.add_node("summarize_history", summarize_history)
graph_builder.add_node("rewrite_query", rewrite_query)
graph_builder.add_node("request_clarification", request_clarification)
graph_builder.add_node("agent", agent_subgraph)
graph_builder.add_node("aggregate_answers", aggregate_answers)

graph_builder.add_edge(START, "summarize_history")
graph_builder.add_edge("summarize_history", "rewrite_query")
graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
graph_builder.add_edge("request_clarification", "rewrite_query")
graph_builder.add_edge("agent", "aggregate_answers")
graph_builder.add_edge("aggregate_answers", END)

# Final Compile
agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])

# --- 7. MAIN EXECUTION ---
'''if __name__ == "__main__":
    # Your choice logic here
    print("Project initialized. Ready for indexing or agent execution.")'''
    
if __name__ == "__main__":
    # Initialize the checkpointer for memory
    checkpointer = InMemorySaver()
    # (Ensure your graph assembly code is above this and uses 'agent_graph')
    
    print("\n🚀 Agent Online! Type 'exit' to stop.")
    config = {"configurable": {"thread_id": "session_1"}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break

        for event in agent_graph.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config, 
            stream_mode="values"
        ):
            # This will print the latest message from the graph
            if "messages" in event:
                event["messages"][-1].pretty_print()