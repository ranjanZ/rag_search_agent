import streamlit as st
import sys
import os

st.config.set_option("server.address", "::")

# Add parent directory to path to import the agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.chat_agent_service.chat_agent import run_agent
#from src.chat_agent_service.chat_agent_v1 import run_agent

# --- Streamlit UI Setup ---
st.set_page_config(page_title="MBZUAI Agent Chat", layout="wide")
st.title("🎓 MBZUAI Intelligent Agent Assistant")

# Initialize session state for chat history and metrics tracking
if "messages" not in st.session_state:
    st.session_state.messages = []

if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_questions": 0,
        "off_topic_count": 0,
        "relevant_count": 0,
        "confidence_scores": []  # Tracks confidence scores for evaluated queries
    }

# Sidebar for session management
with st.sidebar:
    st.header("⚙️ Session Controls")
    st.caption("Track your session analytics and clear history.")
    if st.button("🗑️ Clear Chat & Metrics", use_container_width=True):
        st.session_state.messages = []
        st.session_state.metrics = {
            "total_questions": 0,
            "off_topic_count": 0,
            "relevant_count": 0,
            "confidence_scores": []
        }
        st.rerun()

# Create the Two Tabs
tab1, tab2 = st.tabs(["💬 Chat Interface", "📊 Analytics & Retrieval"])

# ==========================================
# TAB 1: CHAT INTERFACE
# ==========================================
with tab1:
    st.markdown("This agent uses **LangGraph** to orchestrate retrieval, evaluate context confidence, and resolve ambiguity.")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about MBZUAI research..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the LangGraph Agent
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent is Searching & Evaluating..."):
                try:
                    # run_agent returns the full state dictionary
                    agent_state = run_agent(prompt)
                    
                    # Extract data from the new state keys
                    response = agent_state.get("final_answer", "Sorry, I couldn't process that.")
                    is_off_topic = agent_state.get("is_off_topic", False)
                    is_relevant = agent_state.get("is_relevant", False)
                    confidence_score = agent_state.get("confidence_score", 0.0)
                    reason = agent_state.get("reason", "")
                    retrieved_docs = agent_state.get("retrieved_docs", [])
                    
                    # Update session metrics
                    st.session_state.metrics["total_questions"] += 1
                    if is_off_topic:
                        st.session_state.metrics["off_topic_count"] += 1
                    if is_relevant:
                        st.session_state.metrics["relevant_count"] += 1
                        # Only track confidence if it actually reached the evaluation phase
                        st.session_state.metrics["confidence_scores"].append(confidence_score)
                        
                except Exception as e:
                    response = f"An error occurred while processing your request: {str(e)}"
                    retrieved_docs = []
                    confidence_score = 0.0
                    reason = ""
                    
            st.markdown(response)
            
        # Add assistant response to chat history (including retrieved docs and metrics for Tab 2)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "retrieved_docs": retrieved_docs,
            "confidence_score": confidence_score,
            "reason": reason
        })

# ==========================================
# TAB 2: ANALYTICS & RETRIEVAL
# ==========================================
with tab2:
    st.header("📊 Session Analytics")
    
    # Display Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Questions", st.session_state.metrics["total_questions"])
    col2.metric("Off-Topic Queries", st.session_state.metrics["off_topic_count"])
    col3.metric("Relevant Retrievals", st.session_state.metrics["relevant_count"])
    
    # Calculate Confidence Metrics
    scores = st.session_state.metrics["confidence_scores"]
    avg_conf = sum(scores) / len(scores) if scores else 0.0
    last_conf = scores[-1] if scores else 0.0
    
    col4.metric("Avg Confidence", f"{avg_conf:.2f}", delta=f"Last: {last_conf:.2f}")
    
    # Highlight the system threshold
    st.caption("🎯 **System Threshold:** The agent requires a confidence score of **≥ 0.6** to generate a final answer. Scores below 0.6 trigger the abstention and clarification workflow.")
    
    st.divider()
    st.header("🔍 Retrieved Chunks History")
    st.caption("View the exact chunks retrieved by the hybrid engine, along with the LLM's confidence evaluation for each query.")
    
    # Extract pairs of user questions and assistant answers
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    asst_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
    
    if not asst_msgs:
        st.info("No retrieval data available yet. Ask a question in the Chat tab!")
    else:
        # Iterate through the conversation history
        for i, (user_msg, asst_msg) in enumerate(zip(user_msgs, asst_msgs)):
            query_text = user_msg["content"]
            docs = asst_msg.get("retrieved_docs", [])
            conf = asst_msg.get("confidence_score", 0.0)
            reason = asst_msg.get("reason", "")
            
            # Expand the latest query by default, collapse older ones to save space
            is_latest = (i == len(user_msgs) - 1)
            
            with st.expander(f"Query {i+1}: {query_text}", expanded=is_latest):
                # Display the confidence score and reasoning for this specific query
                if conf > 0 or reason:
                    st.info(f"**Confidence Score:** `{conf:.2f}` | **Reasoning:** {reason}")
                
                if not docs:
                    st.warning("No chunks were retrieved for this query (Triggered Off-Topic or Not Relevant).")
                else:
                    for j, doc in enumerate(docs):
                        meta = doc.get("metadata", {})
                        title = meta.get("title") or meta.get("name", "Unknown Source")
                        doc_type = meta.get("doc_type", "N/A")
                        section = meta.get("section", "N/A")
                        
                        st.markdown(f"**Chunk {j+1}** | 📁 Source: `{title}` | 🏷️ Type: `{doc_type}` | 📂 Section: `{section}`")
                        st.caption(f"Chunk ID: {doc.get('id', 'N/A')}")
                        
                        # Use a disabled text area for clean, readable text display
                        st.text_area(
                            "Content", 
                            value=doc.get("enriched_text", ""), 
                            height=150, 
                            disabled=True, 
                            key=f"doc_{i}_{j}"
                        )
                        st.divider()