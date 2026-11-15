from dotenv import load_dotenv

load_dotenv()

from typing import List
from pydantic import BaseModel, Field


from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """schema for source used by the agent"""

    url: str = Field(description="the URl of this Source")


class AgentResponce(BaseModel):
    """schema for agent responce with answer and sources"""
    answer: str = Field(description="the agent answer to the query ")
    source: List[Source] = Field(
        default_factory=list, description="List of sorces generated the answer"
    )


llm = ChatOllama(
    model="gpt-oss:20b", reasoning=False, temperature=0, num_ctx=8192, num_predict=1024
)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponce)


def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="I want to search for three job posting for an ai engineer in tamilnadu in the linkedin and list their details "
            )
        }
    )
    print(result)


main() 
