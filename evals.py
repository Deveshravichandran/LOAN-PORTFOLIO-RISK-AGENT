"""
Automated Evaluation Framework for the Portfolio Monitoring Agent.
This suite ensures the AI model maintains high reliability and prevents regressions.
"""
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
import json

EVAL_DATASET = [
    {
        "company": "Acme Corp",
        "industry": "Manufacturing",
        "covenants": "Debt/EBITDA < 3.0, Operating Margin > 10%",
        "memo_context": "Acme relies heavily on steel imports from Taiwan. High debt load, needs >10% margin.",
        "mock_news": "Taiwan steel exports halted due to strike. Acme Corp announces they expect a 5% drop in operating margins this quarter.",
        "expected_status": "RED"
    },
    {
        "company": "Stark Industries",
        "industry": "Defense & Technology",
        "covenants": "Must maintain $5M cash on hand, no new lawsuits",
        "memo_context": "Prone to lawsuits. Must keep $5M cash.",
        "mock_news": "Stark Industries wins new $200M government contract. CEO Tony Stark donates $1M to charity.",
        "expected_status": "GREEN"
    }
]

def run_evals():
    """
    Executes a series of mock scenarios against the agent's analytical chain 
    to measure accuracy and decision-making quality.
    """
    import os
    llama_model = os.getenv("LLAMA_MODEL", "gemma2:2b")
    llm = ChatOllama(model=llama_model, temperature=0.0)
    
    prompt_template = """    You are an expert Credit Risk Analyst. 
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
    
    correct = 0
    total = len(EVAL_DATASET)
    
    print("\n--- Running Eval Suite against Golden Dataset ---")
    
    for item in EVAL_DATASET:
        print(f"\nEvaluating: {item['company']} Scenario")
        result = chain.invoke({
            "company": item['company'],
            "industry": item['industry'],
            "covenants": item['covenants'],
            "memo_context": item['memo_context'],
            "news_context": item['mock_news']
        })
        
        try:
            clean_output = result.content.strip().strip('```json').strip('```').strip()
            parsed = json.loads(clean_output)
            predicted_status = parsed.get("status", "UNKNOWN")
            reasoning = parsed.get("reasoning", "")
        except Exception as e:
            predicted_status = "PARSE_ERROR"
            reasoning = str(e)
            
        is_match = predicted_status == item['expected_status']
        if is_match:
            correct += 1
            print(f"✅ PASS: Expected {item['expected_status']}, Got {predicted_status}")
        else:
            print(f"❌ FAIL: Expected {item['expected_status']}, Got {predicted_status}")
            
        print(f"   Reasoning: {reasoning}")
            
    accuracy = (correct / total) * 100
    print(f"\n=== Eval Summary: {accuracy:.1f}% Accuracy ({correct}/{total}) ===")

if __name__ == "__main__":
    run_evals()
