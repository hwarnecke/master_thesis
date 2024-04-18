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
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from data_handling.webcrawl_code.ServiceScraper import ServiceScraper


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