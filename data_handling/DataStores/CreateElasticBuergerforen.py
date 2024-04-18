"""
Create a ElasticSearch vector store and fill it with the extracted Bürgerforen data
Remember to start the specific ElasticSearch docker container with the following command:

    docker compose up buergerforen -d

If the docker daemon is not running, start it with:

    sudo systemctl start docker

To stop the container, use:

    docker compose down buergerforen
"""

from llama_index.core import Settings, get_response_synthesizer
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from ..metadata_extraction.BuergerforenLoader import BuergerforenPDFLoader

# some settings
# this currently uses the OpenAI embedding model,
# I want to test if maybe a model specifically for german text would be better
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

# create an Elasticsearch index
es_vector_store = ElasticsearchStore(
    index_name="buergerforen_store",
    es_url="http://localhost:9200",
)

storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

# load the Bürgerforen data
nodes = BuergerforenPDFLoader().load_data("../../data/Buergerforen")

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)