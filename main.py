from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    openai_llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
    )

    chain = prompt | openai_llm

    print(chain.invoke({"input": "Hello from langchain-course!"}).content)


if __name__ == "__main__":
    main()
