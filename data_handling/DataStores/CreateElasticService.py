import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from data_handling.webcrawl_code.ServiceScraper import ServiceScraper

"""
Create a ElasticSearch vector store and fill it with the City Service information scraped from the website.

Remember to start the specific ElasticSearch docker container

If the docker daemon is not running, start it with:

        sudo systemctl start docker
        
Start the container with:

        docker compose up city_services -d

To stop the container, use:

        docker compose down city_services
"""


def create_index(models: list,
                 es_url: str = "http://localhost:9200",
                 load_from_disc: bool = False):
    load_dotenv()
    open_ai_key = os.getenv("OPENAI_API_KEY")
    hugging_face_key = os.getenv("HF_API_KEY")

    # load the City Service data
    scraper = ServiceScraper()
    nodes = scraper.ScrapeServicePage(chunk_size=512, load_from_disc=load_from_disc)

    for model in models:

        name = "service_" + model.split("/")[1]
        print(f"starting: {name}")
        if not model == "OpenAI/text-embedding-ada-002":
            embedding_model = HuggingFaceEmbedding(model_name=model, max_length=512)
            Settings.embed_model = embedding_model

        es_vector_store = ElasticsearchStore(
            index_name=name,
            es_url=es_url,
        )

        storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

        try:
            index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
            print(f"Succeded: {name}")
        except Exception as e:
            print(f"failed: {name} \n with the following exception: {e}")


def main():
    models = ["OpenAI/text-embedding-ada-002",
              "jinaai/jina-embeddings-v2-base-de",
              "intfloat/multilingual-e5-large-instruct",
              "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
              ]

    create_index(models)

def single():
    models = [
        "intfloat/multilingual-e5-large-instruct"
    ]
    create_index(models, load_from_disc=True)

if __name__ == "__main__":
    single()
