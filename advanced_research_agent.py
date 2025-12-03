from typing import Dict, List, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from load_dotenv import load_dotenv
import os

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    research_results: Annotated[List[Dict], "The research results from Tavily"]
    validated_sources: Annotated[List[Dict], "Sources that have been fact-checked"]
    final_answer: Annotated[str, "The final answer drafted by the answer agent"]
    research_plan: Annotated[str, "The research plan created by the planning agent"]

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"), temperature=0)

def planning_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research planner. Your task is to break down the research query into specific sub-questions and create a research plan.\n\nFor the given query, create a detailed research plan that:\n1. Identifies key areas to investigate\n2. Specifies what information needs to be gathered\n3. Suggests potential sources to consult\n4. Outlines the structure of the final answer"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = planning_prompt | llm | StrOutputParser()
    research_plan = chain.invoke({"messages": messages})
    return {
        **state,
        "research_plan": research_plan
    }

def research_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    research_plan = state["research_plan"]
    last_message = messages[-1]
    research_query = last_message.content
    research_results = tavily_client.search(
        query=research_query,
        search_depth="advanced",
        max_results=10,
        include_answer=True,
        include_raw_content=True
    )
    return {
        **state,
        "research_results": research_results
    }

def fact_checking_agent(state: AgentState) -> AgentState:
    research_results = state["research_results"]
    fact_checking_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert fact-checker. Your task is to validate the research results and identify reliable sources.\n\nFor each source in the research results:\n1. Evaluate the credibility of the source\n2. Check for consistency with other sources\n3. Identify any potential biases\n4. Flag any unverified claims\n\nReturn only the sources that pass your validation criteria."""),
        ("human", "{research_results}")
    ])
    chain = fact_checking_prompt | llm | StrOutputParser()
    validated_sources = chain.invoke({"research_results": research_results})
    return {
        **state,
        "validated_sources": validated_sources
    }

def answer_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    research_plan = state["research_plan"]
    validated_sources = state["validated_sources"]
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research assistant. Your task is to synthesize the research findings into a comprehensive, well-structured answer.\n\nResearch Plan:\n{research_plan}\n\nValidated Sources:\n{validated_sources}\n\nPlease provide a detailed answer that:\n1. Follows the research plan structure\n2. Incorporates only verified information\n3. Provides proper citations\n4. Is well-organized and easy to read\n5. Includes a summary of key findings\n6. Notes any limitations or uncertainties"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = answer_prompt | llm | StrOutputParser()
    final_answer = chain.invoke({
        "research_plan": research_plan,
        "validated_sources": validated_sources,
        "messages": messages
    })
    return {
        **state,
        "final_answer": final_answer
    }

def create_research_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("planning", planning_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("fact_checking", fact_checking_agent)
    workflow.add_node("answer", answer_agent)
    workflow.add_edge("planning", "research")
    workflow.add_edge("research", "fact_checking")
    workflow.add_edge("fact_checking", "answer")
    workflow.add_edge("answer", END)
    workflow.set_entry_point("planning")
    return workflow.compile()


query = "How chatgpt deepsearch works?"

initial_state = {
    "messages": [HumanMessage(content=query)],
    "research_results": [],
    "validated_sources": [],
    "final_answer": "",
    "research_plan": ""
}
workflow = create_research_workflow()
final_state = workflow.invoke(initial_state)

result = final_state["final_answer"]
print(result)