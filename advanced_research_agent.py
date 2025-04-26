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

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    research_results: Annotated[List[Dict], "The research results from Tavily"]
    validated_sources: Annotated[List[Dict], "Sources that have been fact-checked"]
    final_answer: Annotated[str, "The final answer drafted by the answer agent"]
    research_plan: Annotated[str, "The research plan created by the planning agent"]

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key= os.getenv("GEMINI_API_KEY") , temperature=0)


# Planning Agent
def planning_agent(state: AgentState) -> AgentState:
    """Agent responsible for creating a research plan."""
    messages = state["messages"]
    last_message = messages[-1]
    
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research planner. Your task is to break down the research query into specific sub-questions and create a research plan.
        
        For the given query, create a detailed research plan that:
        1. Identifies key areas to investigate
        2. Specifies what information needs to be gathered
        3. Suggests potential sources to consult
        4. Outlines the structure of the final answer"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = planning_prompt | llm | StrOutputParser()
    research_plan = chain.invoke({"messages": messages})
    
    return {
        **state,
        "research_plan": research_plan
    }

# Research Agent
def research_agent(state: AgentState) -> AgentState:
    """Agent responsible for conducting research using Tavily."""
    messages = state["messages"]
    research_plan = state["research_plan"]
    last_message = messages[-1]
    
    # Extract the research query from the last message
    research_query = last_message.content
    
    # Perform research using Tavily with advanced settings
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

# Fact-Checking Agent
def fact_checking_agent(state: AgentState) -> AgentState:
    """Agent responsible for validating sources and facts."""
    research_results = state["research_results"]
    
    fact_checking_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert fact-checker. Your task is to validate the research results and identify reliable sources.
        
        For each source in the research results:
        1. Evaluate the credibility of the source
        2. Check for consistency with other sources
        3. Identify any potential biases
        4. Flag any unverified claims
        
        Return only the sources that pass your validation criteria."""),
        ("human", "{research_results}")
    ])
    
    chain = fact_checking_prompt | llm | StrOutputParser()
    validated_sources = chain.invoke({"research_results": research_results})
    
    return {
        **state,
        "validated_sources": validated_sources
    }

# Answer Drafting Agent
def answer_agent(state: AgentState) -> AgentState:
    """Agent responsible for drafting the final answer based on research."""
    messages = state["messages"]
    research_plan = state["research_plan"]
    validated_sources = state["validated_sources"]
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research assistant. Your task is to synthesize the research findings into a comprehensive, well-structured answer.
        
        Research Plan:
        {research_plan}
        
        Validated Sources:
        {validated_sources}
        
        Please provide a detailed answer that:
        1. Follows the research plan structure
        2. Incorporates only verified information
        3. Provides proper citations
        4. Is well-organized and easy to read
        5. Includes a summary of key findings
        6. Notes any limitations or uncertainties"""),
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

# Create the workflow graph
def create_research_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning", planning_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("fact_checking", fact_checking_agent)
    workflow.add_node("answer", answer_agent)
    
    # Add edges
    workflow.add_edge("planning", "research")
    workflow.add_edge("research", "fact_checking")
    workflow.add_edge("fact_checking", "answer")
    workflow.add_edge("answer", END)
    
    # Set the entry point
    workflow.set_entry_point("planning")
    
    return workflow.compile()

# Main function to run the research workflow
def run_research(query: str) -> str:
    # Initialize the state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "research_results": [],
        "validated_sources": [],
        "final_answer": "",
        "research_plan": ""
    }
    
    # Create and run the workflow
    workflow = create_research_workflow()
    final_state = workflow.invoke(initial_state)
    
    return final_state["final_answer"]

if __name__ == "__main__":
    """
    Contain four agent performing different task.
    1. Planning Agent: Creates a detailed research plan
    2. Research Agent: Conducts research using Tavily
    3. Fact-Checking Agent: Validates sources and information
    4. Answer Drafting Agent: Synthesizes the research into a comprehensive answer

    Used Tavily Client from Tavily docs. Also can use langchain tavily.
    """
    query = "How chatgpt deepsearch works?"
    result = run_research(query)
    print(result) 