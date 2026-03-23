from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

@tool
def search_company_news(company_name: str) -> str:
    """Searches the web for the latest financial news and updates about a company."""
    search = DuckDuckGoSearchResults(num_results=3, backend="news")
    try:
        results = search.run(f"{company_name} financial news or corporate events latest")
        return results
    except Exception as e:
        return f"Error searching news for {company_name}: {str(e)}"
