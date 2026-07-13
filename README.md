# MBZUAI RAG System – Faculty & Research Projects

A hybrid RAG (Retrieval-Augmented Generation) system built for  profiles and research projects at MBZUAI. The system uses **LangGraph** orchestration with explicit gates for domain filtering, relevance scoring, ambiguity detection based on confidence, and configurable abstention by Threshold.

Please note that I am only using 

1. Faculty data of Machine Learfacultyning department: https://mbzuai.ac.ae/research-department/machine-learning-department/


2. esearch Projects: https://research.mbzuai.ac.ae/research-projects

---

## 📋 Step-by-Step Setup & Execution Guide

Follow these steps in order to get the application running on your local machine.

### Step 1: Clone the Repository
Open your terminal and clone the project repository.

git clone https://github.com/ranjanZ/rag_search_agent.git
cd rag_search_agent


### Step 2: Create virtual env and install pacges 
python -m venv .venv

source .venv/bin/activate  

pip install -r requirements.txt

### Step 3: Configure Your DeepInfra API Key
open src/config.py   and update DEEPINFRA_API_KEY


### Step 4: Launch the Streamlit Application
ollama pull llama3.2:3b
streamlit run app.py



###  Step 5: Testing
Once the UI is open, you can test the agent's behavior. The system is designed to Answer, Ask Clarifying Questions, or Abstain depending on the query.
 For every query, click on the "📊 Analytics & Retrieval" tab inside the app. You will see the exact text chunks retrieved, the LLM's confidence score, and other information

✅ Test 1: Answerable Questions (High Confidence)

The agent should answer these correctly using the indexed data.

"What awards did Fakhri Karray win?"

"Where did Professor Karray get his PhD?"

"Tell me about the background of Fakhri Karray."
(Note: Check the confidence score in the Analytics tab to see how the system handles broad queries).

"What is the email id of Prof. Martin?"

"Who is the Associate Dean for Academic Affairs?"

🛑 Test 2: Abstained Questions (Low Confidence)

The agent should refuse to answer these and provide helpful suggestions or reasons.

"What is the budget for the traffic prediction project?"
(The agent retrieves the project but realizes the budget field is missing → Low Confidence → Abstains).

"What is the capital of France?"
(Caught by the local router model → Off-Topic → Abstains immediately).

Currently Abstain threshod is 0.6 




### Other Commands 
#### For  indexing 
python src/ingestion_service.py

#### Retreival
python src/retrieval_service.py

### For evalaution 
Update the src/cong.py 

#### To evalaute retrieval 
run python evaluation/evaluation_retrieval.py
# threshold finding 
run python utils/find_threshold.py 

#### To evalaute abastain behabior 
run python evaluation/evaluation_abstain.py




##MAZUAI dataset with our chat agent llama3.2 model

--- Answerable Questions ---
  Count:              50
  Abstain Rate:       0.0800
  Coverage:           0.9200

--- Unanswerable Questions ---
  Count:              50
  Correct Abstain:    1.0000
  False Answer Rate:  0.0000

--- Overall ---
  Total Queries:      100
  Overall Abstain:    0.5400
======================================================================


======================================================================
🎯 ANSWER QUALITY EVALUATION RESULTS [MULTI AGENT] lama3.2 model
======================================================================

--- Answerable Questions ---
  Count:              50
  Abstain Rate:       1.0000
  Coverage:           0.0000

--- Unanswerable Questions ---
  Count:              50
  Correct Abstain:    1.0000
  False Answer Rate:  0.0000

--- Overall ---
  Total Queries:      100
  Overall Abstain:    1.0000
======================================================================







