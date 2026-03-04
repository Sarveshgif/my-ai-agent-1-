import gradio as gr
import uuid
from main import agent_graph
from langchain_core.messages import HumanMessage

# --- UI LOGIC ---

def create_config():
    """Generates a unique session ID for LangGraph memory."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}

# Global config variable for the current session
current_config = create_config()

def chat_with_agent(message, history):
    """
    Function called by Gradio on every message.
    'message' is the new string from the user.
    'history' is the list of previous messages in OpenAI format.
    """
    global current_config
    inputs = {"messages": [HumanMessage(content=message.strip())]}
    
    try:
        result = agent_graph.invoke(inputs, current_config)
        final_answer = result['messages'][-1].content
        
        # OLD GRADIO FORMAT: Just return the string. 
        # ChatInterface handles appending it to the history list automatically.
        return final_answer
        
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def reset_chat():
    """Resets the thread ID so the agent 'forgets' the conversation."""
    global current_config
    current_config = create_config()
    return [] # Returns an empty list to clear the chatbot component

# --- GRADIO LAYOUT ---

# --- Layout Fix for Older Gradio ---

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 RAG Research Agent")
    
    # 1. REMOVED type="messages" (Old Gradio defaults to tuples)
    chatbot = gr.Chatbot(label="Agent Conversation", height=500)
    
    msg = gr.Textbox(label="Ask a question...", placeholder="What is in my documents?")
    clear = gr.Button("New Conversation (Reset Memory)")

    # 2. REMOVED type="messages" here as well
    gr.ChatInterface(
        fn=chat_with_agent, 
        chatbot=chatbot
    )

    clear.click(reset_chat, None, chatbot)
if __name__ == "__main__":
    # Move theme here as per the console warning you saw earlier
    demo.launch(theme=gr.themes.Citrus())