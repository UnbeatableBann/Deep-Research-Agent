# 🧠 Advanced Research Agent

An intelligent multi-agent system that automates the entire research process — from planning and gathering information to fact-checking and synthesizing a final detailed report.

## ✨ Features

- **Planning Agent**: Breaks down the research query into a structured research plan.
- **Research Agent**: Conducts advanced web research using the **Tavily** API.
- **Fact-Checking Agent**: Validates sources and ensures information accuracy.
- **Answer Drafting Agent**: Synthesises validated research into a well-organised, comprehensive answer.

Built using:
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://openai.com/)  (can be used instead of Gemini)
- [Google Generative AI (Gemini)](https://developers.google.com/generative-ai)
- [Tavily Search API](https://docs.tavily.com/)

---

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/UnbeatableBann/advanced-research-agent.git
   cd advanced-research-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file and add your API keys:
   ```
   TAVILY_API_KEY=your_tavily_api_key
   GEMINI_API_KEY=your_google_gemini_api_key
   ```

---

## 🚀 Usage

```bash
python advanced_research_agent.py
```

The script will:
- Plan the research.
- Gather information via Tavily.
- Validate the information.
- Generate a detailed, structured answer.

**Example Query:**
```python
query = "How ChatGPT DeepSearch works?"
result = run_research(query)
print(result)
```

---

## 🧩 How it Works

1. **Planning Phase**  
   Creates a research plan by breaking the query into sub-questions, suggesting sources, and outlining the answer structure.

2. **Research Phase**  
   Performs a deep web search using **Tavily** to gather diverse information.

3. **Fact-Checking Phase**  
   Filters the gathered data to ensure source credibility and factual accuracy.

4. **Answer Drafting Phase**  
   Compiles a detailed, citation-rich, well-organised final report.

---

## 📚 Technologies Used

- `LangChain` for agent orchestration and prompt templates
- `Tavily Client` for advanced research
- `ChatGoogleGenerativeAI` (Gemini 2.0 Flash) for LLM interactions
- `dotenv` for secure API key management

---

## 📄 License

This project is licensed under the MIT License.

---

## 💬 Contributions

Contributions, issues, and feature requests are welcome!  
Feel free to fork the project and submit a pull request. 🚀
