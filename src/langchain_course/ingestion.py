import os
import ssl 
from typing import Any,Dict,List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap


from langchain_course.logger import (Colors,log_error,log_header,log_info,log_success,log_warning)

load_dotenv()
#ssl certificate context

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()


embedding = OllamaEmbeddings(model="qwen3-embedding:4b")

# its used for local database to store embbeddings
# chroma = Chroma(persist_directory="chroma_db",embedding_function=embedding)
print(os.environ["INDEX_NAME"])
# its used for save the  embedding into cloud
vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embedding)
tavily_exttract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5,max_breadth=20)
tavily_crawl = TavilyCrawl()

def main():
    log_header("documentation ingestion pipeline")

    log_info("tawily crawl",Colors.PURPLE)

    res = tavily_crawl.invoke({
        "url":"https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth":5,
        "extract_depth":"advanced",
        "instructions":"content on ai agents"
        })
    all_docs =[Document(page_content=i["raw_content"],metadata={"source":i["url"]} ) for i in res["results"]]

    log_success(all_docs[0])

    log_info(all_docs)
main()