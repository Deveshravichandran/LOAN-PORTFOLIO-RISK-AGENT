import streamlit as st
import pandas as pd
from agent import analyze_portfolio
import os

st.set_page_config(page_title="Risk Portfolio Monitor", page_icon="📈", layout="wide")

st.title("📈 Loan Portfolio Monitoring Agent")
st.markdown("""
This AI agent monitors a loan portfolio by comparing **Live Web News** against the **Original Credit Memo**. 
It uses Embeddings, Web Search Tooling, and Structured LLM Reasoning to flag post-disbursement risk.
""")

# Load Portflio
@st.cache_data
def load_data():
    if os.path.exists("data/portfolio.csv"):
        return pd.read_csv("data/portfolio.csv")
    return pd.DataFrame()

df = load_data()

if not df.empty:
    st.subheader("Current Loan Portfolio")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No portfolio data found in data/portfolio.csv")

if st.button("Run Live Risk Analysis (Agent Web Search)"):
    with st.spinner("Agent is fetching news & analyzing credit memos..."):
        try:
            results = analyze_portfolio(portfolio_path="data/portfolio.csv", index_path="faiss_index")
            
            st.subheader("Agent Risk Assessment")
            
            for res in results:
                company = res["Company"]
                status = res["Status"]
                reasoning = res["Reasoning"]
                
                if status == "GREEN":
                    st.success(f"**{company}** - {status}\n\n{reasoning}")
                elif status == "YELLOW":
                    st.warning(f"**{company}** - {status}\n\n{reasoning}")
                elif status == "RED" or status == "HIGH RISK":
                    st.error(f"**{company}** - {status}\n\n{reasoning}")
                else:
                    st.info(f"**{company}** - {status}\n\n{reasoning}")
                    
        except Exception as e:
            st.error(f"Error running analysis: {e}")
            
st.divider()
st.caption("Evals Framework: To see the rigorous accuracy testing of this agent, run `python evals.py` in the terminal.")
