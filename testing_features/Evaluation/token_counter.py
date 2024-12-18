"""
for tests that need some experiment files imported
"""
import os

import tiktoken
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex, get_response_synthesizer
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

from CreateQueryEngines import create_query_engines
from ModifiedQueryEngine import ModifiedQueryEngine

"""
Update:
It does work when the token counter is initialized in the same function as the query engine
But not if they are made in separate functions. Maybe it is also the order of operation.

so the token counter does not work with the CustomQueryEngine
or at least not when created by the CreateQueryEngine method

it does work when I just initialize and run the retriever and synthesizer in file
when initializing an index in file the token counter increases even for the customQE
although that particular index is not used by them...
"""

def main():
    query_engines = create_query_engines()
    llm = "gpt-3.5-turbo"

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(llm).encode,
        verbose=False,
    )
    Settings.callback_manager = CallbackManager([token_counter])

    es_vector_store = ElasticsearchStore(
        index_name="city_service_store",
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=True)

    retriever = index.as_retriever()
    response_synth = get_response_synthesizer()

    token_counter.reset_counts()

    query = "Wie viel Perso?"
    for name, qe in query_engines.items():
        answer = qe.query(query)
        token_embeddings = token_counter.total_embedding_token_count
        token_completion = token_counter.completion_llm_token_count
        print("Token Embeddings: " + str(token_embeddings))
        print("Token Completion: " + str(token_completion))

    print("in file:")
    answer = response_synth.synthesize(query, retriever.retrieve(query))
    token_embeddings = token_counter.total_embedding_token_count
    token_completion = token_counter.completion_llm_token_count
    print("Token Embeddings: " + str(token_embeddings))
    print("Token Completion: " + str(token_completion))

def CustomEngine():
    llm = "gpt-3.5-turbo"
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    es_vector_store = ElasticsearchStore(
        index_name="city_service_store",
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=True)

    retriever = index.as_retriever()
    response_synth = get_response_synthesizer()

    query = "Wie viel Perso?"
    qe = ModifiedQueryEngine(retriever=retriever, response_synthesizer=response_synth)

    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(llm).encode,
        verbose=False,
    )
    Settings.callback_manager = CallbackManager([token_counter])
    qe.query(query)
    token_embeddings = token_counter.total_embedding_token_count
    token_completion = token_counter.completion_llm_token_count
    print("Token Embeddings: " + str(token_embeddings))
    print("Token Completion: " + str(token_completion))

    qe.query(query)
    token_embeddings = token_counter.total_embedding_token_count
    token_completion = token_counter.completion_llm_token_count
    print("Token Embeddings: " + str(token_embeddings))
    print("Token Completion: " + str(token_completion))

def getQE():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    es_vector_store = ElasticsearchStore(
        index_name="city_service_store",
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=True)

    retriever = index.as_retriever()
    response_synth = get_response_synthesizer()
    qe = ModifiedQueryEngine(retriever=retriever, response_synthesizer=response_synth)
    return qe

def useQE():
    query = "Was Personalausweis?"
    llm = "gpt-3.5-turbo"
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(llm).encode,
        verbose=False,
    )
    Settings.callback_manager = CallbackManager([token_counter])
    qe = getQE()
    r = qe.query(query)
    qe.query(query)
    token_embeddings = token_counter.total_embedding_token_count
    token_completion = token_counter.completion_llm_token_count
    print("Token Embeddings: " + str(token_embeddings))
    print("Token Completion: " + str(token_completion))


def printing():
    print("normal")
    print("\twith indent")
    print("\twith indent")

if __name__ == "__main__":
    printing()