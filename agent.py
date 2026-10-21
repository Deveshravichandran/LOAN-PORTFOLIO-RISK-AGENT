"""
Main Orchestration Module for the Risk Monitoring Agent.
Handles the end-to-end loop: Retrieval (RAG), Search (Tools), and Analysis (LLM).
"""
import os
from dotenv import load_dotenv
import json
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from tools import search_company_news
from vector_store import get_vector_store

load_dotenv()

def analyze_portfolio(portfolio_path="data/portfolio.csv", index_path="faiss_index"):
    """
    Analyzes a loan portfolio by comparing internal credit memos with real-time news news.
    
    Args:
        portfolio_path (str): The path to the CSV file containing portfolio data.
        index_path (str): The path where the FAISS vector index is stored.
        
    Returns:
        list: A collection of analysis results for each company.
    """
    df = pd.read_csv(portfolio_path, encoding="utf-8")
    vector_store = get_vector_store(index_path)

    # Use a low-memory model (gemma2:tiny, ~1.8GB RAM) to prevent memory allocation errors
    llama_model = os.getenv("LLAMA_MODEL", "gemma2:tiny")
    llm = ChatOllama(model=llama_model, temperature=0.0)
    
    analysis_results = []
    
    prompt_template = """
    You are an expert Credit Risk Analyst. 
    Analyze the company based on their Original Credit Memo and the Latest News.
    Determine if their Post-Disbursement Risk status should be GREEN (Safe), YELLOW (Monitor), or RED (High Risk of Breach/Default).

    Company: {company}
    Industry: {industry}
    Key Covenants: {covenants}

    --- ORIGINAL LOAN MEMO CONTEXT ---
    {memo_context}

    --- LATEST WEB NEWS ---
    {news_context}

    Respond ONLY with valid JSON in the following format. Ensure there is no additional text outside the JSON.
    {{
      "status": "GREEN" | "YELLOW" | "RED",
      "reasoning": "1-2 sentences explaining why based on the news vs the memo."
    }}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["company", "industry", "covenants", "memo_context", "news_context"])
    chain = prompt | llm
    
    for _, row in df.iterrows():
        company = row['Company']
        
        memo_docs = vector_store.similarity_search(f"Credit Memo {company}", k=1)
        memo_context = memo_docs[0].page_content if memo_docs else "No memo found."
        
        news_context = search_company_news.invoke({"company_name": company})
        
        result = chain.invoke({
            "company": company,
            "industry": row['Industry'],
            "covenants": row['Key Covenants'],
            "memo_context": memo_context,
            "news_context": news_context
        })
        
        try:
            clean_output = result.content.strip().strip('```json').strip('```').strip()
            parsed = json.loads(clean_output)
            status = parsed.get("status", "UNKNOWN")
            reasoning = parsed.get("reasoning", "No reasoning provided.")
        except Exception as e:
            status = "PARSE_ERROR"
            reasoning = f"Failed to parse model output: {result.content}"
            
        analysis_results.append({
            "Company": company,
            "Status": status,
            "Reasoning": reasoning
        })
        
    return analysis_results

if __name__ == "__main__":
    analyze_portfolio()
