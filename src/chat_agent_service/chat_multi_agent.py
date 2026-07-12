import sys
import os
import json
from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.retrieval_service import engine

try:
    from src.config import DOMAIN_TOPIC
except ImportError:
    DOMAIN_TOPIC = None

# ==========================================================
# 1. Initialize Main LLM (Optimized for CPU)
# ==========================================================
llm = ChatOllama(
    model="qwen2.5:1.5b-instruct-q4_K_M", 
    temperature=0,
    num_ctx=1024,  # Reduced to 1024 to save RAM and speed up CPU inference
    num_thread=4   # Set this to the number of physical cores on your CPU
)

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))
USE_HISTORICAL_CONTEXT = False

# --- Define the Agent State ---
class AgentState(TypedDict):
    query: str
    original_query: str
    retrieved_docs: List[Dict[str, Any]]
    scores: Dict[str, Any]
    is_off_topic: bool
    is_relevant: bool
    chat_history: List[Dict[str, str]]
    use_historical_context: bool
    confidence_threshold: float
    rewritten_query: Optional[str]
    query_was_rewritten: bool
    historical_queries: List[str]

    # A2MAC-specific fields
    fast_response: Dict[str, Any]
    checker_decisions: List[Dict[str, Any]]
    fusion_decision: bool
    abstention_type: Optional[str]
    final_rationale: str
    final_answer: str
    extracted_answer: Optional[str]

# ==========================================================
# 2. Original Nodes (Restored)
# ==========================================================
def rewrite_query_with_context(state: AgentState):
    query = state["query"]
    chat_history = state.get("chat_history", [])
    use_historical_context = state.get("use_historical_context", USE_HISTORICAL_CONTEXT)
    original_query = query
    
    if not use_historical_context or not chat_history:
        return {"original_query": original_query, "rewritten_query": None, "query_was_rewritten": False, "historical_queries": []}
    
    historical_queries = [msg["content"] for msg in chat_history[-5:] if msg["role"] == "user"]
    if not historical_queries:
        return {"original_query": original_query, "rewritten_query": None, "query_was_rewritten": False, "historical_queries": []}
    
    history_context = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(historical_queries)])
    rewrite_prompt = ChatPromptTemplate.from_template(
        "Rewrite the current user query by resolving ambiguous references using the conversation history.\n"
        "History:\n{history_context}\nCurrent Query: {query}\nRespond ONLY with the rewritten query:"
    )
    try:
        chain = rewrite_prompt | llm
        response = chain.invoke({"history_context": history_context, "query": query})
        rewritten_query = response.content.strip()
        query_was_rewritten = rewritten_query.lower() != query.lower()
        if len(rewritten_query) < len(query) * 0.8 or len(rewritten_query) > len(query) * 1.5:
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
        "query": rewritten_query
    }

def check_off_topic(state: AgentState):
    query = state["query"]
    if DOMAIN_TOPIC is None:
        return {"is_off_topic": False}
    prompt = ChatPromptTemplate.from_template(
        f"Determine if the user query is related to {DOMAIN_TOPIC}. "
        "If unrelated, respond ONLY with 'OFF_TOPIC'. If related, respond ONLY with 'ON_TOPIC'.\n"
        "Query: {query}\nResponse:"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    is_off_topic = "OFF_TOPIC" in response.content.upper()
    return {"is_off_topic": is_off_topic}

def hybrid_retrieval(state: AgentState):
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
    docs = state["retrieved_docs"]
    scores = state["scores"]
    max_sem = max(scores.get("semantic", [0.0])) if scores.get("semantic") else 0.0
    max_lex = max(scores.get("lexical", [0.0])) if scores.get("lexical") else 0.0
    max_name = max(scores.get("name_lexical", [0.0])) if scores.get("name_lexical") else 0.0
    
    is_relevant = len(docs) > 0 and (max_sem > 0.3 or max_lex > 4.0 or max_name > 4.0)
    return {"is_relevant": is_relevant}

# ==========================================================
# 3. New A2MAC Nodes (Optimized for CPU: using top 3 chunks)
# ==========================================================
def fast_response_node(state: AgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    context = "\n\n".join([doc.get("enriched_text", "") for doc in docs[:3]])

    # FIX: Escaped the JSON curly braces using {{ and }}
    prompt = ChatPromptTemplate.from_template(
        "Given the query and documents, decide if it can be answered. "
        "Format as JSON: {{'answerable': boolean, 'aspects': list of strings}}.\n"
        "Query: {query}\nDocuments:\n{context}\nJSON Response:"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})

    try:
        clean_json = response.content.strip().strip('```json').strip('```').strip()
        result = json.loads(clean_json)
        answerable = result.get("answerable", False)
        aspects = result.get("aspects", [])
    except:
        answerable = False
        aspects = ["contradiction", "relevance", "sufficiency"]

    return {"fast_response": {"decision": answerable, "aspects": aspects, "reasoning": response.content[:200]}}

def contradiction_checker(state: AgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    context = "\n\n".join([doc.get("enriched_text", "") for doc in docs[:3]])
    prompt = ChatPromptTemplate.from_template(
        "Detect if documents contain contradictory statements making a consistent answer impossible. "
        "Respond ONLY with 'Yes' (contradictions exist) or 'No'.\n"
        "Question: {query}\nDocuments:\n{context}\nDecision (Yes/No):"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    decision = response.content.strip().upper().startswith("YES")
    return {"checker_decisions": state.get("checker_decisions", []) + [{"checker": "contradiction", "decision": decision, "reason": response.content.strip(), "explanation": response.content.strip()}]}

def relevance_checker(state: AgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    context = "\n\n".join([doc.get("enriched_text", "") for doc in docs[:3]])
    prompt = ChatPromptTemplate.from_template(
        "Assess if documents are relevant to answering the question. "
        "Respond ONLY with 'Yes' (irrelevant, should abstain) or 'No' (relevant).\n"
        "Question: {query}\nDocuments:\n{context}\nDecision (Yes/No):"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    decision = response.content.strip().upper().startswith("YES")
    return {"checker_decisions": state.get("checker_decisions", []) + [{"checker": "relevance", "decision": decision, "reason": response.content.strip(), "explanation": response.content.strip()}]}

def sufficiency_checker(state: AgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    context = "\n\n".join([doc.get("enriched_text", "") for doc in docs[:3]])
    prompt = ChatPromptTemplate.from_template(
        "Determine if documents provide enough information to fully answer the question. "
        "Respond ONLY with 'Yes' (insufficient, should abstain) or 'No' (sufficient).\n"
        "Question: {query}\nDocuments:\n{context}\nDecision (Yes/No):"
    )
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    decision = response.content.strip().upper().startswith("YES")
    return {"checker_decisions": state.get("checker_decisions", []) + [{"checker": "sufficiency", "decision": decision, "reason": response.content.strip(), "explanation": response.content.strip()}]}

def fusion_and_summary_node(state: AgentState):
    query = state["query"]
    fast = state.get("fast_response", {})
    checker_decisions = state.get("checker_decisions", [])
    
    checker_summary = "\n".join([
        f"- {c['checker']} checker: {'Abstain' if c['decision'] else 'Answer'} (reason: {c['reason']})"
        for c in checker_decisions
    ])

    # FIX: Escaped the JSON curly braces using {{ and }}
    prompt = ChatPromptTemplate.from_template(
        "Combine assessments to decide whether to abstain.\n"
        "Query: {query}\nFast response: {fast_resp}\nChecker decisions:\n{checker_summary}\n\n"
        "Output JSON: {{'abstain': boolean, 'abstention_type': 'contradictory'/'irrelevant'/'insufficient'/null, 'rationale': string}}\n"
        "JSON Response:"
    )
    chain = prompt | llm
    response = chain.invoke({
        "query": query,
        "fast_resp": json.dumps(fast),
        "checker_summary": checker_summary
    })

    try:
        clean_json = response.content.strip().strip('```json').strip('```').strip()
        result = json.loads(clean_json)
        abstain = result.get("abstain", True)
        abstention_type = result.get("abstention_type")
        rationale = result.get("rationale", "No detailed rationale provided.")
    except:
        abstain = any(c["decision"] for c in checker_decisions)
        abstention_type = None
        if abstain:
            for c in checker_decisions:
                if c["checker"] == "contradiction" and c["decision"]:
                    abstention_type = "contradictory"; break
            if not abstention_type:
                for c in checker_decisions:
                    if c["checker"] == "relevance" and c["decision"]:
                        abstention_type = "irrelevant"; break
            if not abstention_type:
                for c in checker_decisions:
                    if c["checker"] == "sufficiency" and c["decision"]:
                        abstention_type = "insufficient"; break
        rationale = "Based on checker assessments, the decision was made."

    return {
        "fusion_decision": abstain,
        "abstention_type": abstention_type,
        "final_rationale": rationale,
        "final_answer": ""
    }

# ==========================================================
# 4. Answer Generation & Terminal Nodes
# ==========================================================
def generate_answer(state: AgentState):
    query = state["query"]
    docs = state["retrieved_docs"]
    context_text = ""
    for i, doc in enumerate(docs[:3]): 
        meta = doc.get("metadata", {})
        title = meta.get("title") or meta.get("name", "Unknown")
        context_text += f"\n--- Document {i+1} ---\nSource: {title}\nContent: {doc.get('enriched_text', '')}\n"
    prompt = ChatPromptTemplate.from_template(
        "Answer the user's question based ONLY on the provided context. If not in context, state you don't know.\n"
        "Context: {context}\nUser Question: {query}\nAnswer:"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "query": query})
    return {"final_answer": response.content}

def return_abstain(state: AgentState):
    abstention_type = state.get("abstention_type", "unknown")
    rationale = state.get("final_rationale", "No specific reason given.")
    msg = f"I'm sorry, but I cannot answer this question. The system has determined that abstention is necessary due to {abstention_type} evidence. Rationale: {rationale}"
    return {"final_answer": msg}

def return_generic_questions(state: AgentState):
    msg = ("I couldn't find specific information related to your query in the database. "
           "Here are some generic questions you can ask:\n"
           "1. What are the main research projects at MBZUAI?\n"
           "2. Who are the faculty members in the Machine Learning department?\n"
           "3. Tell me about the research project related to CyberAI")
    return {"final_answer": msg}

def return_off_topic(state: AgentState):
    return {"final_answer": "I am designed to answer questions specifically about MBZUAI, its research projects, and faculty members. Please ask a question related to these topics."}

# ==========================================================
# 5. Graph Construction
# ==========================================================
def route_after_off_topic(state):
    return "return_off_topic" if state.get("is_off_topic") else "hybrid_retrieval"

def route_after_relevance(state):
    is_relevant = state.get("is_relevant", False)
    if not is_relevant:
        return "return_generic_questions"
    else:
        return "fast_response"

def route_after_fusion(state):
    if state.get("fusion_decision", True):
        return "return_abstain"
    else:
        return "generate_answer"

workflow = StateGraph(AgentState)

workflow.add_node("rewrite_query_with_context", rewrite_query_with_context)
workflow.add_node("check_off_topic", check_off_topic)
workflow.add_node("hybrid_retrieval", hybrid_retrieval)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("return_generic_questions", return_generic_questions)
workflow.add_node("return_off_topic", return_off_topic)

workflow.add_node("fast_response", fast_response_node)
workflow.add_node("contradiction_checker", contradiction_checker)
workflow.add_node("relevance_checker", relevance_checker)
workflow.add_node("sufficiency_checker", sufficiency_checker)
workflow.add_node("fusion_and_summary", fusion_and_summary_node)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("return_abstain", return_abstain)

workflow.set_entry_point("rewrite_query_with_context")

workflow.add_edge("rewrite_query_with_context", "check_off_topic")
workflow.add_conditional_edges("check_off_topic", route_after_off_topic, {
    "hybrid_retrieval": "hybrid_retrieval",
    "return_off_topic": "return_off_topic"
})
workflow.add_edge("hybrid_retrieval", "check_relevance")
workflow.add_conditional_edges("check_relevance", route_after_relevance, {
    "return_generic_questions": "return_generic_questions",
    "fast_response": "fast_response"
})

workflow.add_edge("fast_response", "contradiction_checker")
workflow.add_edge("contradiction_checker", "relevance_checker")
workflow.add_edge("relevance_checker", "sufficiency_checker")
workflow.add_edge("sufficiency_checker", "fusion_and_summary")

workflow.add_conditional_edges("fusion_and_summary", route_after_fusion, {
    "return_abstain": "return_abstain",
    "generate_answer": "generate_answer"
})

workflow.add_edge("generate_answer", END)
workflow.add_edge("return_abstain", END)
workflow.add_edge("return_generic_questions", END)
workflow.add_edge("return_off_topic", END)

agent_graph = workflow.compile()

# ==========================================================
# 6. Runner Function
# ==========================================================
def run_agent(query: str, chat_history: Optional[List[Dict[str, str]]] = None,
              use_historical_context: bool = False, confidence_threshold: float = 0.6):
    initial_state = {
        "query": query,
        "original_query": query,
        "retrieved_docs": [],
        "scores": {},
        "is_off_topic": False,
        "is_relevant": False,
        "chat_history": chat_history or [],
        "use_historical_context": use_historical_context,
        "confidence_threshold": confidence_threshold,
        "rewritten_query": None,
        "query_was_rewritten": False,
        "historical_queries": [],
        "fast_response": {},
        "checker_decisions": [],
        "fusion_decision": True,
        "abstention_type": None,
        "final_rationale": "",
        "final_answer": "",
        "extracted_answer": None,
    }
    final_state = agent_graph.invoke(initial_state)
    return final_state

# ==========================================================
# 7. Test
# ==========================================================
if __name__ == "__main__":
    print("Testing A2MAC agent...")
    print(run_agent("What is the capital of France?")) 
    final_state = run_agent("Who is Prof Le Song")
    print(final_state.get("final_answer"))
    final_state = run_agent("what is the mail id of Prof Le Song")
    print(final_state.get("final_answer"))