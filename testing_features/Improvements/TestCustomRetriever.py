from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings, StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter

from CustomRetriever import CustomRetriever
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_handling.webcrawl_code.ServiceScraper import ServiceScraper

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    documents = SimpleDirectoryReader("../data").load_data()

    # the BM25Retriever does not operate on the VectorStoreIndex, so we need to create nodes instead
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    # we still need a normal index for the base retriever
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)


    # define the retrievers that will go into the custom retriever
    retrievers = [
        BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5),
        index.as_retriever(similarity_top_k=5),
    ]

    # create the custom retriever
    retriever = CustomRetriever(retrievers, mode="OR")

    retrieved_nodes = retriever.retrieve("Which grad schools did the author apply for and why?")

    print(len(retrieved_nodes))

def hybrid():
    scraper = ServiceScraper()
    nodes = scraper.ScrapeServicePage(chunk_size=512,
                                      load_from_disc=True,
                                      service_path="/media/singularity/Dokumente/Studentenfutter/CogntiveScience/MasterThesis/Code/master_thesis/data_handling/DataStores/service.json",
                                      contact_path="/media/singularity/Dokumente/Studentenfutter/CogntiveScience/MasterThesis/Code/master_thesis/data_handling/DataStores/contact.json")
    print(nodes)

if __name__ == '__main__':
    hybrid()
