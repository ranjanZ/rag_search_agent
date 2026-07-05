import sys
import os
import json
from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Add parent directory to path to import src modules and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.retrieval_service import engine
from config import DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, LLM_MODEL

# Initialize LLM (DeepInfra)
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=DEEPINFRA_API_KEY,
    base_url=DEEPINFRA_BASE_URL,
    temperature=0
)


# --- Define the Agent State ---
class AgentState(TypedDict):
    query: str
    retrieved_docs: List[Dict[str, Any]]
    scores: Dict[str, Any]
    is_off_topic: bool
    is_relevant: bool
    is_ambiguous: bool
    suggested_questions: List[str]
    final_answer: str
    extracted_answer: Optional[str]  # NEW: Stores the answer if extracted during ambiguity check
    ambiguity_reason: str            # NEW: Stores the reason for the ambiguity decision




# --- Node Functions (The Steps) ---
def check_off_topic(state: AgentState):
    """Checks if the query is related to MBZUAI/Academic topics."""
    query = state["query"]
    prompt = ChatPromptTemplate.from_template(
        "You are a router. Determine if the following user query is related to MBZUAI, "
        "its research projects, faculty members, computer science, AI, or academic topics. "
        "If it is completely off-topic (e.g., weather, cooking, general trivia), respond with 'OFF_TOPIC'. "
        "Otherwise, respond with 'ON_TOPIC'.\n\nQuery: {query}\n\nResponse:"
    )
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



def check_ambiguity(state: AgentState):
    """Uses LLM to check if the retrieved context is sufficient and attempts to extract the answer."""
    query = state["query"]
    docs = state["retrieved_docs"]
    
    # Combine top 3 chunks for context
    context_text = "\n\n".join([doc.get("enriched_text", "") for doc in docs[:10]])
    
    # NOTE: We use {{ and }} to escape braces in LangChain's ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(
        "You are an expert evaluator. Given a user query and a set of retrieved documents, determine if you can "
        "answer the user's question using the provided context. You might be able to answer the question fully or "
        "partially using only some of the context.\n\n"
        "Evaluate the context and respond ONLY with a valid JSON object in the following format:\n"
        "{{\n"
        "  \"reason\": \"Briefly explain if the context provides enough information to answer the query, or why it is insufficient/ambiguous.\",\n"
        "  \"answer\": \"If the question is answerable from the context, provide the answer here. It does not need to be a perfectly clean or full answer; just extract and provide whatever relevant information is available. If the context absolutely does not contain the answer, set this to null.\",\n"
        "  \"status\": \"Use 'CLEAR' if the context contains the answer (even partially). Use 'AMBIGUOUS' ONLY if the context is completely insufficient, irrelevant, or contradictory and cannot answer the question at all.\"\n"
        "}}\n\n"
        "Query: {query}\n\n"
        "Retrieved Documents:\n{context}\n\n"
        "JSON Response:"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context_text})
    
    # Set default to False
    is_ambiguous = False
    reason = ""
    extracted_answer = None
    
    try:
        # Clean up markdown formatting if the LLM adds it
        clean_json = response.content.strip().strip('```json').strip('```').strip()
        result = json.loads(clean_json)
        
        # Default status to CLEAR if missing, and only set to True if AMBIGUOUS is explicitly stated
        status = str(result.get("status", "CLEAR")).upper()
        is_ambiguous = "AMBIGUOUS" in status
        reason = result.get("reason", "")
        extracted_answer = result.get("answer")
        
        # Clean up extracted answer if it's the string "null"
        if isinstance(extracted_answer, str) and extracted_answer.lower() == "null":
            extracted_answer = None
            
    except Exception as e:
        print(f"⚠️ Error parsing ambiguity JSON: {e}")
        # Fallback if JSON parsing fails: only set to True if AMBIGUOUS is in the raw text
        is_ambiguous = "AMBIGUOUS" in response.content.upper()
        
    suggested = []
    if is_ambiguous:
        # Generate suggested questions based on the user's query and the retrieved chunks
        suggest_prompt = ChatPromptTemplate.from_template(
            "The user asked a question, but the provided context doesn't fully answer it. "
            "Suggest 3 alternative or follow-up questions that are closely related to the user's original question, "
            "but CAN be answered using the provided context.\n\n"
            "Original User Question: {query}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Instructions:\n"
            "- Generate exactly 3 questions.\n"
            "- The questions should be refinements, specific aspects, or related topics of the original question that the context actually covers.\n"
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
            clean_json = suggest_response.content.strip().strip('```json').strip('```').strip()
            suggested = json.loads(clean_json)
            # Ensure it's a list of strings
            if not isinstance(suggested, list):
                suggested = []
        except Exception:
            # Fallback generic suggestions if JSON parsing fails
            suggested = [
                f"What are the main details about the topic in the user's question?",
                f"Who are the key people involved in the research related to the user's question?",
                f"What are the specific objectives of the projects related to the user's question?"
            ]

    return {
        "is_ambiguous": is_ambiguous, 
        "suggested_questions": suggested,
        "extracted_answer": extracted_answer,
        "ambiguity_reason": reason
    }


def generate_answer(state: AgentState):
    """Generates the final answer. Reuses extracted_answer if already available to save an LLM call!"""
    extracted_answer = state.get("extracted_answer")
    
    # OPTIMIZATION: If the ambiguity checker already extracted a valid answer, just use it!
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
    reason = state.get("ambiguity_reason", "The retrieved information is ambiguous or insufficient.")
    
    msg = f"I'm sorry, but I cannot answer your question accurately. Reason: {reason}\n\n"
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
           "3. Tell me about the Computer Vision research.")
    return {"final_answer": msg}

def return_off_topic(state: AgentState):
    return {"final_answer": "I am designed to answer questions specifically about MBZUAI, its research projects, and faculty members. Please ask a question related to these topics."}

# --- Graph Construction ---

def route_after_off_topic(state):
    return "return_off_topic" if state.get("is_off_topic") else "hybrid_retrieval"

def route_after_relevance(state):
    return "return_generic_questions" if not state.get("is_relevant") else "check_ambiguity"

def route_after_ambiguity(state):
    return "return_abstain" if state.get("is_ambiguous") else "generate_answer"

workflow = StateGraph(AgentState)

# 1. Add Nodes
workflow.add_node("check_off_topic", check_off_topic)
workflow.add_node("hybrid_retrieval", hybrid_retrieval)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("check_ambiguity", check_ambiguity)
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
    "check_ambiguity": "check_ambiguity", "return_generic_questions": "return_generic_questions"
})
workflow.add_conditional_edges("check_ambiguity", route_after_ambiguity, {
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
        "is_off_topic": False, "is_relevant": False, "is_ambiguous": False,
        "suggested_questions": [], "final_answer": "",
        "extracted_answer": None, "ambiguity_reason": ""
    }
    final_state = agent_graph.invoke(initial_state)
    return final_state

  
if __name__ == "__main__":
    print("Testing Agent...")
    print(run_agent("What is the capital of France?")) 
    print(run_agent("Who is Prof Le Song"))          