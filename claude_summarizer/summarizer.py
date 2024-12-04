import os

from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_anthropic import ChatAnthropic


load_dotenv()

def summarize_pdf(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    llm = ChatAnthropic(temperature=0,model_name="claude-3-5-haiku-latest")
    chain = load_summarize_chain(llm,chain_type="map_reduce")
    summary = chain.invoke(docs)

    return summary


if __name__ == "__main__":
    summary = summarize_pdf("math.pdf")

    print(summary)

