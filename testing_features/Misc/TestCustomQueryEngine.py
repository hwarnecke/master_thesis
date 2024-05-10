from CustomQueryEngine import ModifiedQueryEngine
from RAGQueryEngine import RAGQueryEngine
from dotenv import load_dotenv
import os
import re

from llama_index.core.query_engine import RetrieverQueryEngine
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer, SimpleDirectoryReader
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)

def create_index():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    documents = SimpleDirectoryReader("../data").load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index


def base(query: str):

    index = create_index()
    retriever = index.as_retriever()
    synthesizer = get_response_synthesizer()

    query_engine = ModifiedQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
    #query_engine = RAGQueryEngine(retriever=retriever, response_synthesizer=synthesizer)

    response = query_engine.custom_query(query)
    print(response)

def rerank(query: str):

    index = create_index()
    reranker = SentenceTransformerRerank()
    retriever = index.as_retriever()
    synthesizer = get_response_synthesizer()
    query_engine = ModifiedQueryEngine(retriever= retriever, response_synthesizer=synthesizer, reranker=reranker)
    response = query_engine.custom_query(query)
    print(response)

def hyde(query: str):
    index = create_index()
    hyde = HyDEQueryTransform(include_original=True)
    retriever = index.as_retriever()
    synthesizer = get_response_synthesizer()
    query_engine = ModifiedQueryEngine(retriever=retriever, response_synthesizer=synthesizer, hyde=hyde)
    response = query_engine.custom_query(query)
    print(response)

    print("hyde: " + query_engine.hyde_object)

if __name__ == "__main__":
    query = "Where did the author apply?"
    hyde(query)