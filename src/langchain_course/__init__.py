from dotenv import load_dotenv
load_dotenv()


from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

llm = ChatOllama(model="qwen3.5:4b")
tools=[TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="I want to search for three job posting for an ai engineer in tamilnadu in the linkedin and list their details ")})
    print(result)

main()
