# MBZUAI RAG System – Faculty & Research Projects

A hybrid RAG (Retrieval-Augmented Generation) system built for faculty profiles and research projects at MBZUAI. The system uses **LangGraph** orchestration with explicit gates for domain filtering, relevance scoring, ambiguity detection based on confidence, and configurable abstention by Threshold.

Please note that I am only using 
Faculty data of Machine Learning department: https://mbzuai.ac.ae/research-department/machine-learning-department/
Research Projects: https://research.mbzuai.ac.ae/research-projects

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
open src/config.py

export DEEPINFRA_API_KEY="your-actual-api-key-here"

### Step 4: Launch the Streamlit Application
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
