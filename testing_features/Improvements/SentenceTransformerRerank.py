# testing the SentenceTransformerRerank
# using the local HF Embedding as a CrossEncoder

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

documents = SimpleDirectoryReader(
    "../data"
).load_data()

index = VectorStoreIndex.from_documents(documents, show_progress=True)

local_model = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
reranker = SentenceTransformerRerank(
    model=local_model, top_n=3
)

query_rerank = index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker], verbose=True)

query_normal = index.as_query_engine(similarity_top_k=10)

response_rerank = query_rerank.query("Which grad schools did the author apply for and why?")
response_normal = query_normal.query("Which grad schools did the author apply for and why?")

"""
print("Rerank Answer:")
print(response_rerank.response)
print("Rerank Sources:")
print(response_rerank.get_formatted_sources(length=200))

print("Normal:")
print(response_normal.response)
print("Normal Sources:")
print(response_normal.get_formatted_sources(length=200))
"""

"""
this is a good way of getting the source nodes from the response object while actually using the query engines.
"""

print(type(response_rerank))

#print(response_rerank.get_formatted_sources(length=2000))
print(len(response_rerank.source_nodes))

for sources in response_rerank.source_nodes:
    print(sources.node.get_text())