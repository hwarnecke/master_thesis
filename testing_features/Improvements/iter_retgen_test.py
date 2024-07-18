from ITER_RETGEN import ITER_RETGEN
import os


from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings, get_response_synthesizer, StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    llm = "gpt-3.5-turbo"
    Settings.llm = OpenAI(model=llm, api_key=api_key)
    vector_store_name = "service_" + "text-embedding-ada-002"

    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url="http://localhost:9200",
    )
    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)

    response_synth = get_response_synthesizer()
    basic_retriever = index.as_retriever(similarity_top_k=2)

    iter_qe = ITER_RETGEN(retriever=basic_retriever, generator=response_synth)

    query = "Was kostet ein Personalausweis?"
    response = iter_qe.query(query)
    print(response)
    print(type(response))
    print(iter_qe.get_log())

if __name__ == "__main__":
    main()