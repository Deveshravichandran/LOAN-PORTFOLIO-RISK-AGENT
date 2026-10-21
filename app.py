"""
The Streamlit Frontend for the Blue Sherpa Risk Portfolio Monitor.
Provides UI for viewing the portfolio and triggering AI risk assessments.
"""
import streamlit as st
import pandas as pd
from agent import analyze_portfolio
import os

st.set_page_config(page_title="Risk Portfolio Monitor", page_icon="📈", layout="wide")

def load_data():
    """
    Loads company portfolio data from a CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the portfolio, or an empty DataFrame if file is not found.
    """
    if os.path.exists("data/portfolio.csv"):
        return pd.read_csv("data/portfolio.csv", encoding="utf-8")
    return pd.DataFrame()

def render_ui():
    """
    Handles the layout, logic, and rendering of all Streamlit components.
    """
    st.title("📈 Loan Portfolio Monitoring Agent")
    st.markdown("""
    This AI agent monitors a loan portfolio by comparing **Live Web News** against the **Original Credit Memo**. 
    It leverages Vector Embeddings, Web Search Tools, and LLM Reasoning to proactively flag post-disbursement risks.
    """)

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
                
                st.subheader("Agent Risk Assessment Results")
                
                for res in results:
                    company = res["Company"]
                    status = res["Status"]
                    reasoning = res["Reasoning"]
                    
                    if status == "GREEN":
                        st.success(f"**{company}** - {status}\n\n{reasoning}")
                    elif status == "YELLOW":
                        st.warning(f"**{company}** - {status}\n\n{reasoning}")
                    elif status == "RED":
                        st.error(f"**{company}** - {status}\n\n{reasoning}")
                    else:
                        st.info(f"**{company}** - {status}\n\n{reasoning}")
                        
            except Exception as e:
                st.error(f"Error executing analysis: {e}")
                
    st.divider()
    st.caption("Evals Framework Ready: Robustly test agent accuracy by running `python evals.py`.")

if __name__ == "__main__":
    render_ui()
