from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    vector_store_name = "city_service_store"
    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)

    retriever_top_k = 5
    docstore_nodes = index.docstore.docs.values()    # returns empty dict
    #vector_store_nodes = es_vector_store._data.embeddingt_dict.keys()  # _data doesn't exist
    #bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=retriever_top_k)

    # one proposed solution is to just set the top_k so high it will return every node
    # 10000 seems to be maximum but even then it fails
    # 1000 seems to be enough, as 386 are found
    retriever = index.as_retriever(similarity_top_k = 1000)
    source_nodes = retriever.retrieve("und")
    nodes = [x.node for x in source_nodes]

    bm25 = BM25Retriever.from_defaults(nodes = nodes, similarity_top_k=retriever_top_k)

    print(len(nodes))
    #print(nodes[0])

    bm25_nodes = bm25.retrieve("Personalausweis")

    print(bm25_nodes)
