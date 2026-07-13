import sys
import os
import json
from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


# Add parent directory to path to import src modules and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.retrieval_service import engine
from src.config import DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL, LLM_MODEL

# ==========================================================
# 1. Initialize Main LLM (DeepInfra) for complex reasoning
# ==========================================================
# llm = ChatOpenAI(
#     model=LLM_MODEL,
#     api_key=DEEPINFRA_API_KEY,
#     base_url=DEEPINFRA_BASE_URL,
#     temperature=0
# )



# llm = ChatOllama(
#     model="qwen2.5:1.5b-instruct-q4_K_M", 
#     temperature=0,
#     num_ctx=2048
# )


llm = ChatOllama(
    model="llama3.2",  
    temperature=0,
    num_ctx=2048,        
    num_thread=4
)


# Define the confidence threshold for confidence (configurable via environment or runtime)
# Default value, can be overridden at runtime
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))

# Ablation study configuration flags (configurable at runtime)
USE_HISTORICAL_CONTEXT = False  # Whether to use historical questions as context

# --- Define the Agent State ---
class AgentState(TypedDict):
    query: str
    original_query: str  # Store the original query before rewriting
    retrieved_docs: List[Dict[str, Any]]
    scores: Dict[str, Any]
    is_off_topic: bool
    is_relevant: bool
    confidence_score: float      # Replaces is_ambiguous (0.0 to 1.0)
    reason: str                  # Replaces ambiguity_reason
    suggested_questions: List[str]
    final_answer: str
    extracted_answer: Optional[str]
    chat_history: List[Dict[str, str]]  # List of {"role": "user"/"assistant", "content": "..."}
    use_historical_context: bool  # Flag to enable/disable historical context
    confidence_threshold: float   # Per-session confidence threshold
    rewritten_query: Optional[str]  # Stores the rewritten query if historical context was used
    query_was_rewritten: bool     # Flag indicating if query rewriting occurred
    historical_queries: List[str]  # List of previous user queries only (for ablation study)

# --- Node Functions (The Steps) ---

def rewrite_query_with_context(state: AgentState):
    """Rewrites the current query using historical queries if USE_HISTORICAL_CONTEXT is enabled.
    
    This function resolves ambiguous references (e.g., pronouns like "he", "it", "they") 
    by incorporating context from previous user queries only. The immediate query is treated as 
    high priority, with historical context used only for disambiguation.
    
    If USE_HISTORICAL_CONTEXT is False, no rewriting occurs and the original query is used.
    """
    query = state["query"]
    chat_history = state.get("chat_history", [])
    use_historical_context = state.get("use_historical_context", USE_HISTORICAL_CONTEXT)
    
    # Store original query
    original_query = query
    
    # If historical context is disabled or no chat history, skip rewriting
    if not use_historical_context or not chat_history:
        return {
            "original_query": original_query,
            "rewritten_query": None,
            "query_was_rewritten": False,
            "historical_queries": []
        }
    
    # Extract only user queries from history (last 5 for context)
    historical_queries = []
    for msg in chat_history[-5:]:
        if msg["role"] == "user":
            historical_queries.append(msg["content"])
    
    # If no previous user queries, skip rewriting
    if not historical_queries:
        return {
            "original_query": original_query,
            "rewritten_query": None,
            "query_was_rewritten": False,
            "historical_queries": []
        }
    
    # Build historical context string from user queries only
    history_context = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(historical_queries)])
    
    # Prompt to rewrite the query with context resolution
    rewrite_prompt = ChatPromptTemplate.from_template(
        "You are a query rewriting assistant. Your task is to rewrite the current user query "
        "by resolving any ambiguous references using the conversation history.\n\n"
        "**Important Guidelines:**\n"
        "1. The CURRENT query is the HIGHEST PRIORITY - do not change its intent or focus.\n"
        "2. Only add context from history if the current query contains ambiguous references "
        "   (pronouns like 'he', 'she', 'it', 'they', 'this', 'that', or incomplete references).\n"
        "3. If the current query is already complete and clear, return it unchanged.\n"
        "4. Keep the rewritten query concise and natural.\n"
        "5. Do NOT add unnecessary historical information if not needed.\n\n"
        "Conversation History (User Queries Only):\n"
        "{history_context}\n\n"
        "Current Query: {query}\n\n"
        "Respond ONLY with the rewritten query (or the original query if no changes needed). "
        "Do not include any explanations or additional text.\n\n"
        "Examples:\n"
        "History: Q1: What is Le Song's research area?\n"
        "         Q2: Tell me about his publications\n"
        "Current: What about his students?\n"
        "Rewritten: What are Le Song's students?\n\n"
        "History: Q1: What are the ML projects at MBZUAI?\n"
        "Current: Who leads the climate AI project?\n"
        "Rewritten: Who leads the climate AI project? (unchanged - already clear)\n\n"
        "Now rewrite this query:\n"
        "History:\n{history_context}\n"
        "Current Query: {query}\n"
        "Rewritten Query:"
    )
    
    try:
        chain = rewrite_prompt | llm
        response = chain.invoke({
            "history_context": history_context,
            "query": query
        })
        
        rewritten_query = response.content.strip()
        
        # Check if the query actually changed
        query_was_rewritten = rewritten_query.lower() != query.lower()
        
        # If no meaningful change, treat as not rewritten
        if len(rewritten_query) < len(query) * 0.8 or len(rewritten_query) > len(query) * 1.5:
            # Significant length difference might indicate a problem, use original
            rewritten_query = query
            query_was_rewritten = False
            
    except Exception as e:
        print(f"⚠️ Error rewriting query: {e}")
        rewritten_query = query
        query_was_rewritten = False
    
    return {
        "original_query": original_query,
        "rewritten_query": rewritten_query if query_was_rewritten else None,
        "query_was_rewritten": query_was_rewritten,
        "historical_queries": historical_queries,
        "query": rewritten_query  # Update the query field for downstream nodes
    }


def check_off_topic(state: AgentState):
    """Checks if the query is related to the configured domain topic using the SMALL model."""
    query = state["query"]
    original_query = state.get("original_query", query)
    
    # Import DOMAIN_TOPIC from config
    try:
        from src.config import DOMAIN_TOPIC
    except ImportError:
        from config import DOMAIN_TOPIC
    
    # If DOMAIN_TOPIC is None, skip off-topic checking (allow all queries)
    if DOMAIN_TOPIC is None:
        return {"is_off_topic": False}
    
    prompt = ChatPromptTemplate.from_template(
        f"You are a strict classifier. Determine if the user query is related to {DOMAIN_TOPIC}.\n"
        "If it is completely unrelated (e.g., weather, cooking, sports, general trivia), respond ONLY with 'OFF_TOPIC'.\n"
        "If it is related, respond ONLY with 'ON_TOPIC'.\n\n"
        "Query: {query}\n\n"
        "Response:"
    )
    # USE THE SMALL LLM HERE
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
    
    is_relevant = len(docs) > 0 and (max_sem > 0.3 or max_lex > 4.0 or max_name > 4.0)
    return {"is_relevant": is_relevant}


def evaluate_confidence(state: AgentState):
    """Uses LLM to evaluate if context contains the answer and extract it if possible.
    
    This function separates two distinct concepts:
    1. answerability_confidence: How confident we are that the context CONTAINS the answer (0-1)
    2. answer_confidence: If answerable, how confident we are in the extracted answer (0-1)
    
    The routing decision uses answerability_confidence to decide whether to answer or abstain.
    """
    query = state["query"]
    docs = state["retrieved_docs"]
    chat_history = state.get("chat_history", []) 


    # Combine top 10 chunks for context
    context_text = "\n\n".join([doc.get("enriched_text", "") for doc in docs[:10]])
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert evaluator. Given a user query and retrieved documents, determine if the context contains enough information to answer the question.\n\n"
        "**CRITICAL**: You must distinguish between:\n"
        "1. Whether the context CONTAINS the answer (answerability)\n"
        "2. Whether you can provide a correct answer (only if answerable)\n\n"
        "Evaluate the context and respond ONLY with a valid JSON object in the following format:\n"
        "{{\n"
        "  \"answerability_confidence\": <float 0.0-1.0: confidence that the context CONTAINS the answer. 0.0 = definitely NOT in context, 1.0 = definitely IN context>,\n"
        "  \"answer_confidence\": <float 0.0-1.0: IF answerable, confidence in the correctness of your extracted answer. Set to 0.0 if not answerable>,\n"
        "  \"is_answerable\": <boolean: true if context contains the answer, false otherwise>,\n"
        "  \"reason\": \"<Brief explanation: what information is present/missing in the context>\",\n"
        "  \"extracted_answer\": \"<If is_answerable=true, provide the direct answer here. If false, set to null>\",\n"
        "  \"answer_type\": \"<one of: 'explicit' (directly stated), 'inferable' (can be deduced), 'missing' (not in context)>\"\n"
        "}}\n\n"
        "**Examples:**\n\n"
        "Query: \"What is John's gym membership?\"\n"
        "Context: \"John is a professor at MBZUAI. He published 50 papers.\"\n"
        "Response: {{\"answerability_confidence\": 0.05, \"answer_confidence\": 0.0, \"is_answerable\": false, \"reason\": \"Context mentions John's profession and publications but contains no information about gym membership\", \"extracted_answer\": null, \"answer_type\": \"missing\"}}\n\n"
        "Query: \"What is John's title?\"\n"
        "Context: \"John is a professor at MBZUAI. He published 50 papers.\"\n"
        "Response: {{\"answerability_confidence\": 0.95, \"answer_confidence\": 0.95, \"is_answerable\": true, \"reason\": \"Context explicitly states John's title\", \"extracted_answer\": \"Professor at MBZUAI\", \"answer_type\": \"explicit\"}}\n\n"
        "Query: \"What is John's favorite city?\"\n"
        "Context: \"John visited Paris and Tokyo for conferences.\"\n"
        "Response: {{\"answerability_confidence\": 0.1, \"answer_confidence\": 0.0, \"is_answerable\": false, \"reason\": \"Context mentions cities John visited but does not state his favorite\", \"extracted_answer\": null, \"answer_type\": \"missing\"}}\n\n"
        "Now evaluate:\n\n"
        "Query: {query}\n\n"
        "Retrieved Documents:\n{context}\n\n"
        "JSON Response:"
    )


    chain = prompt | llm
    response = chain.invoke({
        "query": query, 
        "context": context_text
    })
    
    # Set defaults
    answerability_confidence = 0.0
    answer_confidence = 0.0
    is_answerable = False
    reason = ""
    extracted_answer = None
    
    try:
        # Clean up markdown formatting if the LLM adds it
        clean_json = response.content.strip().strip('```json').strip('```').strip()
        result = json.loads(clean_json)
        
        # Extract answerability confidence (this drives the abstain decision)
        try:
            answerability_confidence = float(result.get("answerability_confidence", 0.0))
            answerability_confidence = max(0.0, min(1.0, answerability_confidence))
        except (ValueError, TypeError):
            answerability_confidence = 0.0
        
        # Extract answer confidence (only meaningful if answerable)
        try:
            answer_confidence = float(result.get("answer_confidence", 0.0))
            answer_confidence = max(0.0, min(1.0, answer_confidence))
        except (ValueError, TypeError):
            answer_confidence = 0.0
            
        is_answerable = result.get("is_answerable", False)
        reason = str(result.get("reason", ""))
        extracted_answer = result.get("extracted_answer")
        
        # Clean up extracted answer if it's the string "null" or empty
        if isinstance(extracted_answer, str):
            if extracted_answer.lower() == "null" or extracted_answer.strip() == "":
                extracted_answer = None
        
        # Enforce consistency: if not answerable, extracted_answer should be None
        if not is_answerable:
            extracted_answer = None
            answer_confidence = 0.0
            
    except Exception as e:
        print(f"⚠️ Error parsing confidence JSON: {e}")
        print(f"Raw response: {response.content}")
        answerability_confidence = 0.0
        answer_confidence = 0.0
        is_answerable = False
        reason = "Failed to parse LLM evaluation response."
        
    suggested = []
    # Get the threshold from state or use default
    threshold = state.get("confidence_threshold", CONFIDENCE_THRESHOLD)
    
    # If answerability confidence is below threshold, generate suggested questions
    if answerability_confidence < threshold:
        # Build historical context string if available
        historical_context = ""
        if chat_history:
            user_queries = [msg["content"] for msg in chat_history[-5:] if msg["role"] == "user"]
            if user_queries:
                historical_context = "Previous questions in conversation:\n" + "\n".join([f"- {q}" for q in user_queries]) + "\n\n"
        
        suggest_prompt = ChatPromptTemplate.from_template(
            "The user asked a question, but the provided context does NOT contain the answer (low answerability confidence). "
            "Suggest 3 alternative or follow-up questions that are closely related to the user's original question, "
            "but CAN be answered using the provided context.\n\n"
            "{historical_context}"
            "Original User Question: {query}\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Instructions:\n"
            "- Generate exactly 3 questions.\n"
            "- The questions should be refinements, specific aspects, or related topics of the original question that the context actually covers.\n"
            "- They must be independent and answerable using only the provided context.\n"
            "- Do NOT mention any entities or facts that are not present in the context.\n"
            "- The questions must be phrased as natural, standalone questions.\n"
            "- Provide the output strictly as a JSON list of strings, e.g., [\"Question 1?\", \"Question 2?\", \"Question 3?\"]\n"
            "- Do not include any extra text, explanations, or markdown formatting.\n\n"
            "Now, generate the 3 questions for the following:\n"
            "Original Question: {query}\n"
            "Context:\n{context}\n\n"
            "JSON Output:"
        )

        suggest_chain = suggest_prompt | llm
        suggest_response = suggest_chain.invoke({
            "historical_context": historical_context,
            "query": query, 
            "context": context_text
        })
        try:
            clean_json = suggest_response.content.strip().strip('```json').strip('```').strip()
            suggested = json.loads(clean_json)
            if not isinstance(suggested, list):
                suggested = []
        except Exception:
            suggested = [
                "Who are the main researchers or faculty members in Machine Learning?",
                "What is the email address of Prof Le Song?",
                "What is the website URL of Prof Le Song?"
            ]

    return {
        "confidence_score": answerability_confidence,  # This is now clearly answerability confidence
        "answer_confidence": answer_confidence,         # Additional field for answer quality
        "is_answerable": is_answerable,                 # Explicit boolean flag
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
    
    # # Fallback: Generate it from scratch if extraction failed
    # query = state["query"]
    # docs = state["retrieved_docs"]
    
    # context_text = ""
    # for i, doc in enumerate(docs):
    #     meta = doc.get("metadata", {})
    #     title = meta.get("title") or meta.get("name", "Unknown")
    #     context_text += f"\n--- Document {i+1} ---\nSource: {title}\nContent: {doc.get('enriched_text', '')}\n"
        
    # prompt = ChatPromptTemplate.from_template(
    #     "You are an expert AI assistant for MBZUAI. Answer the user's question based ONLY on the provided context. "
    #     "If the answer is not in the context, state that you don't know.\n\nContext: {context}\n\nUser Question: {query}\n\nAnswer:"
    # )
    # chain = prompt | llm
    # response = chain.invoke({"context": context_text, "query": query})
    
    return {"final_answer": "null"}

# --- Terminal Nodes ---

def return_abstain(state: AgentState):
    suggested = state.get("suggested_questions", [])
    reason = state.get("reason", "The retrieved information is insufficient to answer confidently.")
    confidence = state.get("confidence_score", 0.0)
    is_answerable = state.get("is_answerable", False)

    # Show the user the exact confidence score and the LLM's reasoning
    # Make it clear that the context does not contain the answer (not just low confidence in the answer)
    msg = f"I'm sorry, but the provided context does not contain enough information to answer your question (Answerability Confidence: {confidence:.2f}). Reason: {reason}\n\n"
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
workflow.add_node("rewrite_query_with_context", rewrite_query_with_context)
workflow.add_node("check_off_topic", check_off_topic)
workflow.add_node("hybrid_retrieval", hybrid_retrieval)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("evaluate_confidence", evaluate_confidence)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("return_abstain", return_abstain)
workflow.add_node("return_generic_questions", return_generic_questions)
workflow.add_node("return_off_topic", return_off_topic)

# 2. Set Entry Point - First rewrite query with context, then proceed
workflow.set_entry_point("rewrite_query_with_context")

# 3. Add Edges
workflow.add_edge("rewrite_query_with_context", "check_off_topic")
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

def run_agent(query: str, chat_history: Optional[List[Dict[str, str]]] = None, 
              use_historical_context: bool = False, confidence_threshold: float = 0.6):
    """Entry point for the Streamlit app to run the agent.
    
    Args:
        query: The user's current question
        chat_history: List of previous messages [{"role": "user"/"assistant", "content": "..."}]
        use_historical_context: Whether to use historical questions as context (for ablation study)
        confidence_threshold: The threshold for confidence score (configurable per session)
    
    Returns:
        Dictionary containing the final state with all intermediate results for analysis
    """
    initial_state = {
        "query": query, 
        "original_query": query,  # Will be updated by rewrite node
        "retrieved_docs": [], 
        "scores": {},
        "is_off_topic": False, 
        "is_relevant": False, 
        "confidence_score": 0.0, 
        "reason": "",
        "suggested_questions": [], 
        "final_answer": "",
        "extracted_answer": None,
        "chat_history": chat_history or [],
        "use_historical_context": use_historical_context,
        "confidence_threshold": confidence_threshold,
        "rewritten_query": None,
        "query_was_rewritten": False,
        "historical_queries": []
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