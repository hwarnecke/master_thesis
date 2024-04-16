from llama_index.core import Settings, get_response_synthesizer
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore


"""
This uses Elasticsearch in a docker container to store the index.
This allows me to leave the docker running in the background and starting different python scripts to interact with it.
Note that the Llamaindex documentation is rather sparse on how to use Elasticsearch.
Here is how to do it:
You need to start the docker container from a docker-compose.yml file with the following command:
docker-compose up
The docker-compose.yml file should contain a 'volumes:' section with a path to a folder where the data will be stored:
/path/to/persistent/volume:/usr/share/elasticsearch/data

This effectively tells the docker container to store the data in the folder on the host machine instead of keeping it
in-memory only and allows us to reload the data when we restart the container.
"""

# some settings
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

# create an Elasticsearch index
es_vector_store = ElasticsearchStore(
    index_name="thesis_test_store",
    es_url="http://localhost:9200",
)

storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

# when running it the first time we have to build the index from documents
# afterwards we can use the vector store directly
# now I only have to figure out how to actually save that elasticsearch instance to re-launch it later
#documents = SimpleDirectoryReader("./data").load_data()
#index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
index = VectorStoreIndex.from_vector_store(vector_store=es_vector_store, storage_context=storage_context, show_progress=True)

# create a query engine
query_engine = index.as_query_engine()
# query the index
response = query_engine.query("Which grad schools did the author apply for and why?")
print(response)
