from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
load_dotenv()


def main() -> None:
    information="""
    Self-Sabotage and Hubris: Commentators, business figures like Mark Cuban, 
    and analysis in outlets like Bloomberg point out that Musk frequently harms his own business interests and public standing
    through erratic social media posts,
    controversial political commentary, and impulsive decision-making
    """
    summary_template= """
    give the information {information} about a person i want to create :
    1.a short summary
    2.two interesting fact about them."""

    prompt = PromptTemplate(
        input_variables=["information"] ,template=summary_template
    )

    llm = ChatOllama( 
        model="deepseek-r1:8b",
        temperature=0,
        validate_model_on_init=True
        
        )
    chain = prompt | llm
    response = chain.invoke(input={"information":information})
    print(response.content)

main()
