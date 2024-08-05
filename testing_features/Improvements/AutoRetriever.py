"""
Auto-Retrieval or sometimes Self-Querying leverages the existence of metadata.
In short, you give an LLM the query and a list of what metadata is present and it will rearrange the query into
a mix of a query and a filter to apply.
The idea is to narrow down the possible documents that can be matched by semantic search in order to find better results
"""

import os
from dotenv import load_dotenv
from io import StringIO
import sys

from llama_index.core import Settings, get_response_synthesizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine


def main():

    # some settings
    #load_dotenv()
    #api_key = os.getenv("OPENAI_API_KEY")
    #Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
    Settings.llm = Ollama(model="sauerkraut_nemo", request_timeout=600)
    embedding_name = "jinaai/jina-embeddings-v2-base-de"
    embedding_model = HuggingFaceEmbedding(model_name=embedding_name)
    Settings.embed_model = embedding_model

    # create an Elasticsearch index
    # don't forget to start the docker container with the Elasticsearch instance

    es_vector_store = ElasticsearchStore(
        index_name="city_service_jina-embeddings-v2-base-de",
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=True)

    # create a vector store info that contains an overview over the metadata
    vector_store_info = VectorStoreInfo(
        content_info="Informationen über die Dienstleistungen der Stadt Osnabrück.",
        metadata_info=[
            MetadataInfo(
                name="Typ",
                description="Der Typ der Information, die in dem Dokument beschrieben ist.",
                type="String",
            ),
            MetadataInfo(
                name="Name",
                description="Der Name der Dienstleistung.",
                type="String",
            ),
            MetadataInfo(
                name="URL",
                description="Die URL, unter der die Dienstleistung zu finden ist.",
                type="String",
            ),
            MetadataInfo(
                name="Kategorie",
                description="Die Kategorie, zu der die Dienstleistung gehört.",
                type="String or list[String]",
            ),
            MetadataInfo(
                name="Anfangsbuchstabe",
                description="Der Anfangsbuchstabe des Namens der Dienstleistung.",
                type="String",
            ),
            MetadataInfo(
                name="Synonyme",
                description="Synonyme für den Namen der Dienstleistung.",
                type="list[String]",
            ),
            MetadataInfo(
                name="Fachbereich",
                description="Der Fachbereich, zu dem die Dienstleistung gehört.",
                type="String",
            ),
            MetadataInfo(
                name="Kontakt",
                description="Der Kontakt für die Dienstleistung.",
                type="String",
            ),
            MetadataInfo(
                name="Kontakt URL",
                description="Die URL des Kontakts für die Dienstleistung.",
                type="String",
            ),
        ],
    )

    # create an auto retriever
    auto_retriever = VectorIndexAutoRetriever(
        index=index,
        vector_store_info=vector_store_info,
        verbose=True,
    )

    # catch the verbose output of the auto retriever
    filter_info = StringIO()
    old_stdout = sys.stdout
    sys.stdout = filter_info

    query = "Wie viel kostet ein Personalausweis?"
    #documents = []
    try:
        documents = auto_retriever.retrieve(query)
    except Exception as e:
        documents = []
        print(f"retrieval failed with {type(e).__name__}")

    sys.stdout = old_stdout

    value = filter_info.getvalue()
    lines = value.split("\n")
    query_str = lines[0].split(": ")[1]
    filters = lines[1].split(": ")[1]

    print("Query: " + query_str)
    print("filter:" + filters)
    print(f"Retrieved documents: {len(documents)}")

if __name__ == "__main__":
    main()

"""
While it does seem to work in principle, there is still some work to do.
For once, the metadata is rarely used for simple questions. It seems like I have to actively ask in a way that names
the metadata category for the LLM to be able to match it.

Another problem is that the LLM does not know the possible values for the metadata categories. This means that it will
throw an error if I ask i.e. for a specific category that does not exist in the metadata.
This can be problematic as the user writing the initial query might not know the exact metadata categories.
In order to be truly useful the LLM should be capable of matching the metadata by intend and not by exact name.

In order to store the information about what filters where used I need to remap the stdout to a string.
"""


# the verbose output from the retriever can still be catched when used in a query engine
# response_synthesizer = get_response_synthesizer()
#
# query_engine = RetrieverQueryEngine(
#     retriever=auto_retriever,
#     response_synthesizer=response_synthesizer,
# )
#
# question = "Welche Dienstleistungen bietet die Stadt Osnabrück an?"
# response = query_engine.query(question)
