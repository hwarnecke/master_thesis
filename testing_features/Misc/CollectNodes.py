from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import json

def main():
    # some settings
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    # load documents and build index and query engine
    documents = SimpleDirectoryReader("../data").load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    query_engine = index.as_query_engine()

    query_str = "Which grad schools did the author apply for and why?"
    response = query_engine.query(query_str)

    # get the retrieved nodes
    # Assume `response` is the response object from a query
    retrieved_nodes = response.source_nodes

    # Now `retrieved_nodes` is a list of dictionaries. You can iterate over it to access each retrieved document.
    # for node in retrieved_nodes:
    #     print(f"Document ID: {node['document_id']}")
    #     print(f"Score: {node['score']}")
    #     print(f"Document Text: {node['document_text']}")
    for node in retrieved_nodes:
        #print(node.__dict__)
        print(node.get_content())


def es_main():
    embedding_name = "intfloat/multilingual-e5-large-instruct"
    vector_store_name = "service_" + embedding_name.split("/")[1]
    #load_dotenv()
    #api_key = os.getenv("OPENAI_API_KEY")
    #Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_name)

    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)

    #query_engine = index.as_query_engine()
    retriever = index.as_retriever()

    query = "Wie teuer ist eine Umweltplakette?"

    #response = query_engine.query(query)
    retrieved_nodes = retriever.retrieve(query)


    print(retrieved_nodes)

    for node in retrieved_nodes:
        print(node.id_)

def create_nodes():
    chunk_size = 512
    sample_texts = ["Lorem ipsum dolor sit amet", "consectetur adipisici elit"]
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
    all_nodes = []
    for sample_text in sample_texts:
        text_nodes = splitter.get_nodes_from_documents([Document(text=sample_text)])
        for node in text_nodes:
            node.metadata = {
                'Name': "Wild Lorem"
            }
            all_nodes.extend(node)

    print(all_nodes)

    path = "text_nodes.json"
    with open(path, 'w') as file:
        json.dump(all_nodes, file)

    return all_nodes

def read_nodes():
    path = "text_nodes.json"
    with open(path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    es_main()
