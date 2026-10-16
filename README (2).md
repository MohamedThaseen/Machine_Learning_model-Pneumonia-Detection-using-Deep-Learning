# NewsFindr — Agentic AI News Retrieval System

## Problem Statement

NewsFindr is redefining news discovery by delivering real-time news updates tailored to user interests. Traditional search methods and generic news feeds often lead to information overload and inefficiencies, making it challenging for users to access relevant and trustworthy content efficiently.

To address this, NewsFindr leverages **Agentic AI** to build an AI-powered news retrieval agent that ensures accuracy and credibility. Using a structured, multi-step approach, the system provides secure, fair, and explainable recommendations — enhancing user engagement, optimizing content discovery, and improving access to timely and relevant news.

## Objective

- **Provide real-time, personalized news retrieval** to help users discover relevant content effortlessly.
- **Ensure accuracy and credibility** by sourcing news from trusted platforms and minimizing misinformation.
- **Improve user engagement** through seamless content discovery, reducing information overload.
- **Streamline the news consumption process** by eliminating outdated and irrelevant content, providing a refined reading experience.

## How It Works

NewsFindr is built as a multi-agent pipeline that combines structured data retrieval, LLM-driven query expansion, live web search, credibility filtering, and summarization:

1. **Personalized Retrieval**
   An SQL agent queries a customer database (`customer.db`) to identify a specific user's interests based on their email ID (e.g., a user's interest in Technology and Automobile).

2. **Query Expansion**
   The expansion tool converts these broad interests, combined with the user's specific query, into precise, time-sensitive search terms optimized for current news discovery.

3. **Real-Time Search**
   A DuckDuckGo search tool executes the expanded queries and fetches the most relevant, live links from the web.

4. **Credibility Filtering**
   A credibility filter acts as a fact-checking layer, auditing the retrieved URLs to prioritize authoritative sources (e.g., Reuters, BBC, AP) while excluding clickbait, bias, or misinformation.

5. **Summarization**
   A summarization tool synthesizes the information from the trusted sources into a concise, bulleted, easy-to-digest report for the end user.

## Architecture

```
User Email + Query
       │
       ▼
┌─────────────────┐
│   SQL Agent      │  → Fetches user interests from customer.db
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ ExpandSearchQueries │ → Converts interests + query into time-sensitive search terms
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ DuckDuckGoSearch │ → Retrieves live news results
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ CredibilityFilter │ → Filters for trustworthy, relevant sources
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  SummarizeNews   │ → Produces final bulleted summary report
└─────────────────┘
       │
       ▼
   Final Response
```

## Tech Stack

- **LLM Provider:** [Groq](https://groq.com/) (`meta-llama/llama-4-scout-17b-16e-instruct`)
- **Orchestration:** [LangChain](https://www.langchain.com/) (`langchain`, `langchain-groq`, `langchain-experimental`)
- **Database:** SQLite (via `langchain.sql_database.SQLDatabase` + `SQLDatabaseToolkit`)
- **Web Search:** [DDGS](https://pypi.org/project/ddgs/) (DuckDuckGo Search)
- **Agent Framework:** LangChain's `initialize_agent` (Zero-Shot ReAct) and `create_sql_agent`

## Requirements

```bash
pip install -qU langchain-groq
pip install -qU \
    langchain==0.3.27 \
    langchainhub==0.1.21 \
    langchain-experimental==0.3.4 \
    langchain-groq==0.3.8 \
    ddgs==9.6.1 \
    numpy==2.0.2
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Groq API key**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

4. **Provide a customer database**
   The notebook expects a SQLite database (`customer.db`) containing customer records with at least `email_id` and `interests` columns.

5. **Run the notebook**
   Open `Advanced_GenAI_for_NewsFindr.ipynb` in Jupyter / Google Colab and run the cells sequentially.

## Usage

```python
email = "emma.a88fec03-c@gmail.com"
user_query = "Find the latest news on self-driving car technology advancements and how AI is being integrated into the next generation of electric vehicles."

response = query_response(email, user_query)
print(response)
```

The pipeline will:
1. Fetch the user's interests from the database.
2. Expand those interests + query into search-optimized queries.
3. Search the web for relevant, recent news.
4. Filter the results for credibility and relevance.
5. Return a concise, summarized report with source links.

## Key Components

| Component | Description |
|---|---|
| `sql_agent` | Retrieves customer records and interests from the SQLite database |
| `expand_search_queries` / `ExpandSearchQueries` | Generates time-sensitive, precise news search queries from user interests |
| `ddg_search` / `DuckDuckGoSearch` | Searches DuckDuckGo for the latest news articles |
| `filter_with_llm` / `CredibilityFilter` | Evaluates and filters search results based on source reliability, neutrality, and relevance |
| `summarize_news` / `SummarizeNews` | Synthesizes filtered news sources into a readable summary |
| `query_response` | End-to-end orchestration function tying all the above together |

## Notes

- A deterministic LLM (temperature `0`) is used for factual, consistent outputs (SQL querying, filtering, summarization).
- A high-temperature LLM (temperature `1.5`) is used elsewhere for more creative/exploratory generation and comparison.
- The credibility filter favors established news agencies (e.g., AP, Reuters, BBC), academic journals, and official reports, while excluding sensationalist or unverified sources.

## Disclaimer

This project is a prototype/academic exercise demonstrating an agentic AI pipeline for personalized, credibility-checked news retrieval. It is not intended for production use without further security, validation, and rate-limiting enhancements.
