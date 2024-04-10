"""
After some troubles with ElasticSearch and the Llamaindex documentation, I decided to try ChromaDB.
This is the only Vector DB that has an actual example in the documentation that describes how to make it persistent.

When I later need to transfer the previously stored index (the one created by web scraping) to the ChromaDB,
I can use this method: https://github.com/run-llama/llama_index/issues/6687
"""

import os
import chromadb
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# some settings
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-4", api_key=api_key)

# path to store the chromadb
persist_dir = "./chromadb_storage"

# this part stays the same, independent of whether the chromadb is already saved or not
db = chromadb.PersistentClient(path=persist_dir)
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# if the chromadb is already saved, we can load it, otherwise we need to load the data and create it
# EDIT: I think the folder is created when initializing the chromadb, so it will be true here even for the first time
# need a different automated check to see if I still need to index anything, for now I'll just use a boolean
already_indexed = True
if already_indexed:
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
else:
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

# create the query engine
query_engine = index.as_query_engine()
response = query_engine.query("Which grad schools did the author apply for and why?")
print(response)