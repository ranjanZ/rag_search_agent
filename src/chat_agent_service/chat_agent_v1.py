import sys
import os
import json
import re
from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Kept for the commented-out Ollama small_llm
from langchain_google_genai import ChatGoogleGenerativeAI # NEW: For Gemini
from langgraph.graph import StateGraph, END

# Add parent directory to path to import src modules and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.retrieval_service import engine 

# from src.config import DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, LLM_MODEL # REMOVED

# ==========================================================
# 1. Initialize Main LLM (Google Gemini 1.5 Flash)
# ==========================================================
# Note: "Gemini 3.5" is likely a typo for "Gemini 1.5". 
# LangChain expects the model name without the "models/" prefix.
GEMINI_API_KEY = "AIzaSyCZoL7wOmm3U13lsnRAIvjOOE3lW5MDqqo"
GEMINI_MODEL = "models/gemini-2.5-flash" # Use "gemini-1.5-flash" or your specific model alias

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ==========================================================
# 2. Initialize Small LLM (Local Ollama/LM Studio) for routing
# ==========================================================
# Note: Ensure Ollama is running locally with the model pulled.
# small_llm = ChatOpenAI(
#     model="qwen2.5:1.5b-instruct-q4_K_M",
#     base_url="http://localhost:11434/v1", 
#     api_key="ollama", # Required by langchain_openai even if local server ignores it
#     temperature=0
# )

# Define the threshold for confidence
CONFIDENCE_THRESHOLD = 0.6

# --- Helper Function for Robust JSON Parsing ---
def extract_json_from_llm(text: str) -> str:
    """Robustly extracts JSON from LLM output, ignoring conversational filler and markdown."""
    text = text.strip()
    # 1. Look for markdown code blocks (handles "```json" or "```")
    match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2. Look for raw JSON objects or arrays if not wrapped in markdown
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # 3. Final fallback
    return text.strip('`').strip()

# --- Define the Agent State ---
class AgentState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]
    scores: Dict[str, Any]
    is_off_topic: bool
    is_relevant: bool
    confidence_score: float      # Replaces is_ambiguous (0.0 to 1.0)
    reason: str                  # Replaces ambiguity_reason
    suggested_questions: List[str]
    final_answer: str
    extracted_answer: Optional[str]

# --- Node Functions (The Steps) ---

def check_off_topic(state: AgentState):
    """Checks if the query is related to MBZUAI/Academic topics using the SMALL model."""
    query = state["query"]
    prompt = ChatPromptTemplate.from_template(
        "You are a strict classifier. Determine if the user query is related to MBZUAI, "
        "artificial intelligence, computer science, research projects, or academic topics.\n"
        "If it is completely unrelated (e.g., weather, cooking, sports, general trivia), respond ONLY with 'OFF_TOPIC'.\n"
        "If it is related, respond ONLY with 'ON_TOPIC'.\n\n"
        "Query: {query}\n\n"
        "Response:"
    )
    # USE THE MAIN LLM HERE (or switch to small_llm if uncommented)
    chain = prompt | llm
    response = chain.invoke({"query": query})
    is_off_topic = "OFF_TOPIC" in response.content.upper()
    return {"is_off_topic": is_off_topic}

def hybrid_retrieval(state: AgentState):
    """Calls the 3-index hybrid retrieval engine."""
    query = state["query"]
    results_dict = engine.hybrid_search(query)
    
    return {
        "retrieved_docs": results_dict["fused"],
        "scores": {
            "semantic": results_dict["semantic"]["scores"],
            "lexical": results_dict["lexical"]["scores"],
            "name_lexical": results_dict["name_lexical"]["scores"]
        }
    }

def check_relevance(state: AgentState):
    """Checks if the retrieved documents have high enough scores to be considered relevant."""
    docs = state["retrieved_docs"]
    scores = state["scores"]
    
    max_sem = max(scores.get("semantic", [0.0])) if scores.get("semantic") else 0.0
    max_lex = max(scores.get("lexical", [0.0])) if scores.get("lexical") else 0.0
    max_name = max(scores.get("name_lexical", [0.0])) if scores.get("name_lexical") else 0.0
    
    is_relevant = len(docs) > 0 and (max_sem > 0.35 or max_lex > 2.0 or max_name > 2.0)
    return {"is_relevant": is_relevant}

def evaluate_confidence(state: AgentState):
    """Uses LLM to evaluate context confidence and attempts to extract the answer."""
    query = state["query"]
    docs = state["retrieved_docs"]
    
    # Combine top 10 chunks for context
    context_text = "\n\n".join([doc.get("enriched_text", "") for doc in docs[:10]])
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert evaluator. Given a user query and a set of retrieved documents, determine how well the "
        "context answers the user's question. You might be able to answer the question fully or partially.\n\n"
        "Evaluate the context and respond ONLY with a valid JSON object in the following format:\n"
        "{{\n"
        "  \"confidence\": <float between 0.0 and 1.0 indicating your confidence that the context contains the answer. 0.0 = no info, 1.0 = perfect answer>,\n"
        "  \"reason\": \"<Briefly explain why you assigned this confidence score>\",\n"
        "  \"answer\": \"<If confidence >= 0.6, provide the extracted answer here. If confidence < 0.6 or context lacks the answer, set this to null>\"\n"
        "}}\n\n"
        "Query: {query}\n\n"
        "Retrieved Documents:\n{context}\n\n"
        "JSON Response:"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context_text})
    
    # Set defaults
    confidence = 0.0
    reason = ""
    extracted_answer = None
    
    try:
        # UPDATED: Use the robust JSON extractor for Gemini
        clean_json = extract_json_from_llm(response.content)
        result = json.loads(clean_json)
        
        # Extract confidence and ensure it's a float safely clamped between 0.0 and 1.0
        try:
            confidence = float(result.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.0
            
        reason = str(result.get("reason", ""))
        extracted_answer = result.get("answer")
        
        # Clean up extracted answer if it's the string "null"
        if isinstance(extracted_answer, str) and extracted_answer.lower() == "null":
            extracted_answer = None
            
    except Exception as e:
        print(f"⚠️ Error parsing confidence JSON: {e}")
        confidence = 0.0
        reason = "Failed to parse LLM evaluation response."
        
    suggested = []
    # If confidence is below the threshold, generate suggested questions
    if confidence < CONFIDENCE_THRESHOLD:
        suggest_prompt = ChatPromptTemplate.from_template(
            "The user asked a question, but the provided context doesn't fully answer it (Confidence is low). "
            "Suggest 3 alternative or follow-up questions that are closely related to the user's original question, "
            "but CAN be answered using the provided context.\n\n"
            "Original User Question: {query}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Instructions:\n"
            "- Generate exactly 3 questions.\n"
            "- The questions should be refinements, specific aspects, or related topics of the original question that the context actually covers.\n"
            "- The questions be independent of each other to the point where they can be answered using the context.\n"
            "- Provide the output strictly as a JSON list of strings.\n\n"
            "Examples:\n"
            "Original Question: \"Tell me about Le Song's publications on quantum computing.\"\n"
            "Context: (Contains info about Le Song's work in reinforcement learning, but no quantum computing).\n"
            "Output: [\"What are Le Song's main research areas in machine learning?\", \"Can you list some of Le Song's recent publications in reinforcement learning?\", \"What projects is Le Song currently leading at MBZUAI?\"]\n\n"
            "Original Question: \"What is the budget for the Climate AI project?\"\n"
            "Context: (Contains details about the Climate AI project's goals and team, but no financial info).\n"
            "Output: [\"Who are the lead researchers for the Climate AI project?\", \"What are the main objectives of the Climate AI project?\", \"Which category does the Climate AI project fall under?\"]\n\n"
            "Now, generate the 3 questions for the following:\n"
            "Original Question: {query}\n"
            "Context:\n{context}\n\n"
            "JSON Output:"
        )
        suggest_chain = suggest_prompt | llm
        suggest_response = suggest_chain.invoke({"query": query, "context": context_text})
        try:
            # UPDATED: Use the robust JSON extractor for Gemini
            clean_json = extract_json_from_llm(suggest_response.content)
            suggested = json.loads(clean_json)
            if not isinstance(suggested, list):
                suggested = []
        except Exception:
            suggested = [
                f"Who are the main researchers or faculty members in Machine Learning?",
                f"what is the mail id of Prof Le Song?",
                f"What is the url of Prof Le Song?"
            ]

    return {
        "confidence_score": confidence, 
        "reason": reason,
        "suggested_questions": suggested,
        "extracted_answer": extracted_answer
    }

def generate_answer(state: AgentState):
    """Generates the final answer. Reuses extracted_answer if already available to save an LLM call!"""
    extracted_answer = state.get("extracted_answer")
    
    # OPTIMIZATION: If the confidence checker already extracted a valid answer, just use it!
    if extracted_answer and extracted_answer.lower() != "null":
        return {"final_answer": extracted_answer}
        
    # Fallback: Generate it from scratch if extraction failed
    query = state["query"]
    docs = state["retrieved_docs"]
    
    context_text = ""
    for i, doc in enumerate(docs):
        meta = doc.get("metadata", {})
        title = meta.get("title") or meta.get("name", "Unknown")
        context_text += f"\n--- Document {i+1} ---\nSource: {title}\nContent: {doc.get('enriched_text', '')}\n"
        
    prompt = ChatPromptTemplate.from_template(
        "You are an expert AI assistant for MBZUAI. Answer the user's question based ONLY on the provided context. "
        "If the answer is not in the context, state that you don't know.\n\nContext: {context}\n\nUser Question: {query}\n\nAnswer:"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "query": query})
    
    return {"final_answer": response.content}

# --- Terminal Nodes ---

def return_abstain(state: AgentState):
    suggested = state.get("suggested_questions", [])
    reason = state.get("reason", "The retrieved information is insufficient to answer confidently.")
    confidence = state.get("confidence_score", 0.0)
    
    # Show the user the exact confidence score and the LLM's reasoning
    msg = f"I'm sorry, but I cannot answer your question with high confidence (Confidence: {confidence:.2f}). Reason: {reason}\n\n"
    if suggested:
        msg += "Here are some relevant questions you might want to ask instead:\n"
        for i, q in enumerate(suggested, 1):
            msg += f"{i}. {q}\n"
    return {"final_answer": msg}

def return_generic_questions(state: AgentState):
    msg = ("I couldn't find specific information related to your query in the MBZUAI database. "
           "Here are some generic questions you can ask:\n"
           "1. What are the main research projects at MBZUAI?\n"
           "2. Who are the faculty members in the Machine Learning department?\n"
           "3. Tell me about the research projet related to CyberAI")
    return {"final_answer": msg}

def return_off_topic(state: AgentState):
    return {"final_answer": "I am designed to answer questions specifically about MBZUAI, its research projects, and faculty members. Please ask a question related to these topics."}

# --- Graph Construction ---

def route_after_off_topic(state):
    return "return_off_topic" if state.get("is_off_topic") else "hybrid_retrieval"

def route_after_relevance(state):
    return "return_generic_questions" if not state.get("is_relevant") else "evaluate_confidence"

def route_after_confidence(state):
    # Route to abstain if confidence is below the dynamic threshold
    return "return_abstain" if state.get("confidence_score", 0.0) < CONFIDENCE_THRESHOLD else "generate_answer"

workflow = StateGraph(AgentState)

# 1. Add Nodes
workflow.add_node("check_off_topic", check_off_topic)
workflow.add_node("hybrid_retrieval", hybrid_retrieval)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("evaluate_confidence", evaluate_confidence)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("return_abstain", return_abstain)
workflow.add_node("return_generic_questions", return_generic_questions)
workflow.add_node("return_off_topic", return_off_topic)

# 2. Set Entry Point
workflow.set_entry_point("check_off_topic")

# 3. Add Edges
workflow.add_conditional_edges("check_off_topic", route_after_off_topic, {
    "hybrid_retrieval": "hybrid_retrieval", "return_off_topic": "return_off_topic"
})
workflow.add_edge("hybrid_retrieval", "check_relevance")
workflow.add_conditional_edges("check_relevance", route_after_relevance, {
    "evaluate_confidence": "evaluate_confidence", "return_generic_questions": "return_generic_questions"
})
workflow.add_conditional_edges("evaluate_confidence", route_after_confidence, {
    "generate_answer": "generate_answer", "return_abstain": "return_abstain"
})

workflow.add_edge("generate_answer", END)
workflow.add_edge("return_abstain", END)
workflow.add_edge("return_generic_questions", END)
workflow.add_edge("return_off_topic", END)

agent_graph = workflow.compile()

def run_agent(query: str):
    """Entry point for the Streamlit app to run the agent."""
    initial_state = {
        "query": query, "retrieved_docs": [], "scores": {},
        "is_off_topic": False, "is_relevant": False, 
        "confidence_score": 0.0, "reason": "",
        "suggested_questions": [], "final_answer": "",
        "extracted_answer": None
    }
    final_state = agent_graph.invoke(initial_state)
    return final_state

  
if __name__ == "__main__":
    print("Testing Agent...")
    print(run_agent("What is the capital of France?")) 
    final_state = run_agent("Who is Prof Le Song")
    print(final_state.get("final_answer"))
    
    final_state = run_agent("what is the mail id of Prof Le Song")
    print(final_state.get("final_answer"))