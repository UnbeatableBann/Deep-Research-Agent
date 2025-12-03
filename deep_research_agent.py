from typing import Dict, List, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
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
    final_answer: Annotated[str, "The final answer drafted by the answer agent"]

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

def research_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    
    research_query = last_message.content
    
    research_results = tavily_client.search(
        query=research_query,
        search_depth="advanced",
        max_results=5
    )
    
    return {
        "messages": messages,
        "research_results": research_results,
        "final_answer": state["final_answer"]
    }

def answer_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    research_results = state["research_results"]
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research assistant. Your task is to synthesize the research findings into a comprehensive, well-structured answer.
        
        Research Results:
        {research_results}
        
        Please provide a detailed answer that:
        1. Addresses the original question
        2. Incorporates key findings from the research
        3. Provides citations where appropriate
        4. Is well-structured and easy to read"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = answer_prompt | llm | StrOutputParser()
    final_answer = chain.invoke({
        "research_results": research_results,
        "messages": messages
    })
    
    return {
        "messages": messages,
        "research_results": research_results,
        "final_answer": final_answer
    }

def create_research_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("research", research_agent)
    workflow.add_node("answer", answer_agent)
    workflow.add_edge("research", "answer")
    workflow.add_edge("answer", END)
    workflow.set_entry_point("research")
    return workflow.compile()


query = "What are the latest developments in quantum computing?" #write any query I want

initial_state = {
    "messages": [HumanMessage(content=query)],
    "research_results": [],
    "final_answer": ""
}
workflow = create_research_workflow()
final_state = workflow.invoke(initial_state)
result =  final_state["final_answer"]
print(result)