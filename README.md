# Loan Portfolio Monitoring AI Agent 🛡️

A high-precision AI agent designed for **Post-Disbursement Credit Risk Monitoring**. This agent continuously monitors a portfolio of loans by scraping live financial news and comparing it against original loan memos and legal covenants to generate real-time risk alerts.

### ✨ Key Features
- **Agentic Live Search**: Uses DuckDuckGo to scrape the latest business news for any company in the world.
- **RAG (Retrieval-Augmented Generation)**: Ingests unstructured loan memos into a FAISS vector store.
- **Advanced Reasoning**: Powered by **Llama 3 (8B)** with Chain-of-Thought prompting and Few-Shot examples for "Human-in-the-loop" quality reasoning.
- **Interactive Dashboard**: Built with Streamlit to visualize portfolio health (GREEN / RED statuses).
- **Built-in Evals**: A validation framework to measure agent accuracy against a "Golden Dataset."

### 🚀 Getting Started
1. **Prerequisites**: [Ollama](https://ollama.com/) must be installed.
2. **Pull the Model**:
   ```bash
   ollama pull llama3:8b
   ```
3. **Install Dependencies**:
   ```bash
   pip install langchain langchain-community langchain-openai streamlit pandas python-dotenv faiss-cpu duckduckgo-search
   ```
4. **Index the Data**:
   ```bash
   python vector_store.py
   ```
5. **Run the App**:
   ```bash
   streamlit run app.py
   ```

---
*Built for the Blue Sherpa Job Application - Demonstrating Agentic Tool Use and LLM Evaluation.*
