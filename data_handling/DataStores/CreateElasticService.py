"""
Create a ElasticSearch vector store and fill it with the City Service information scraped from the website.

Remember to start the specific ElasticSearch docker container with the following command:

        docker compose up city_services -d

If the docker daemon is not running, start it with:

            sudo systemctl start docker

To stop the container, use:

                docker compose down city_services
"""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from data_handling.webcrawl_code.ServiceScraper import ServiceScraper



def old():
    # some settings
    # this currently uses the OpenAI embedding model,
    # I want to test if maybe a model specifically for german text would be better
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # create an Elasticsearch index
    es_vector_store = ElasticsearchStore(
        index_name="city_service_store",
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    # load the City Service data
    scraper = ServiceScraper()
    nodes = scraper.ScrapeServicePage()

    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)

def create_index(models: list,
                 es_url: str = "http://localhost:9200"):

    load_dotenv()
    open_ai_key = os.getenv("OPENAI_API_KEY")
    hugging_face_key = os.getenv("HF_API_KEY")

    # load the City Service data
    scraper = ServiceScraper()
    nodes = scraper.ScrapeServicePage(chunk_size=512)

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
        except:
            print(f"failed: {name}")


def main():
    models = ["OpenAI/text-embedding-ada-002",
              "jinaai/jina-embeddings-v2-base-de",
              "intfloat/multilingual-e5-large-instruct",
              #"T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
                ]

    create_index(models)

if __name__ == "__main__":
    main()