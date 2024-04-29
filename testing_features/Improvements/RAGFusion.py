from dotenv import load_dotenv
import os
import re

from llama_index.core.query_engine import RetrieverQueryEngine
from openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.postprocessor import SentenceTransformerRerank

from FusionRetriever import FusionRetriever

def remove_leading_numbers(s):
    return re.sub(r'^\d+\.\s*', '', s)

# load api key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

es_vector_store = ElasticsearchStore(
    index_name="city_service_store",
    es_url="http://localhost:9200",
)

storage_context = StorageContext.from_defaults(vector_store=es_vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=es_vector_store, storage_context=storage_context, show_progress=True)

base_retriever = index.as_retriever(similarity_top_k=5)
retriever = FusionRetriever(retriever=base_retriever)

reranker = SentenceTransformerRerank(top_n=3)

response_synthesizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[reranker],
)

question = "Wie beantrage ich einen Personalausweis"
response = query_engine.query(question)

print(response)
