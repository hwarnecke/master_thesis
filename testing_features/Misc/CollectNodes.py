from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, QueryBundle
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core import Settings, get_response_synthesizer
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

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


# what is all the weird reference answer shit there??


if __name__ == "__main__":
    main()