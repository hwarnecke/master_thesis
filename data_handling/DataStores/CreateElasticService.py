import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from data_handling.webcrawl_code.ServiceScraper import ServiceScraper
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding

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


def create_index(models: dict,
                 es_url: str = "http://localhost:9200",
                 load_from_disc: bool = False):
    """
    Creates an ElasticSearch index over the service page data.
    :param models: the embedding models to create an embedding for
    :param es_url: the ElasticSearch container
    :param load_from_disc: In case the service page was already scraped, the data can just be loaded again
    """
    load_dotenv()
    # load the City Service data
    scraper = ServiceScraper()
    nodes = scraper.ScrapeServicePage(chunk_size=512, load_from_disc=load_from_disc)

    # create an embedding for each embedding model
    for embedding_name, embedding_type in models.items():

        # Embedding models can come from different suppliers, load them accordingly
        # is mostly the same code as in CreateQueryEngines, because it does the same thing
        match embedding_type:
            case 'OpenAI':
                openAI_api_key = os.getenv("OPENAI_API_KEY")
                embedding_model = OpenAIEmbedding(model_name=embedding_name, api_key=openAI_api_key)
            case 'Cohere':
                cohere_api_key = os.getenv("COHERE_API_KEY")
                embedding_model = CohereEmbedding(api_key=cohere_api_key, model_name=embedding_name)
            case 'HuggingFace':
                text_instructions: str = "Repräsentiere das Dokument für eine Suche."
                query_instructions: str = "Finde relevante Dokumente, die die folgende Frage beantworten."
                # there have been OOM issues for some models, when running on cuda and I don't want to list
                # individual models, so I'll just run them all on cpu.
                # also, I wasn't sure what happens if I supply instructions to non instruct models, so I split it here
                if "instruct" in embedding_name:
                    embedding_model = HuggingFaceEmbedding(model_name=embedding_name,
                                                           device='cpu',
                                                           query_instruction=query_instructions,
                                                           text_instruction=text_instructions,
                                                           max_length=512)
                else:
                    embedding_model = HuggingFaceEmbedding(model_name=embedding_name, device='cpu')
                # HuggingFace models always comes like 'author/model-name', so I need to remove the author.
                # additionally, they can use both '-' and '_' to seperate words in the model_name,
                # so if I want to split it, I need to filter for both
            case _:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")

        name = "service_" + embedding_name
        Settings.embed_model = embedding_model
        print(f"starting: {name}")

        es_vector_store = ElasticsearchStore(
            index_name=name,
            es_url=es_url,
        )

        storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
        print(f"Succeded: {name}")


def main():

    embedding_models = {"aari1995/German_Semantic_V3b": "HuggingFace",
                        "T-Systems-onsite/cross-en-de-roberta-sentence-transformer": "HuggingFace",
                        "jinaai/jina-embeddings-v2-base-de": "HuggingFace",
                        "jinaai/jina-clip-v1": "HuggingFace",
                        "intfloat/multilingual-e5-large-instruct": "HuggingFace",
                        "Alibaba-NLP/gte-multilingual-base": "HuggingFace",
                        "dunzhang/stella_en_1.5B_v5": "HuggingFace",
                        "GritLM/GritLM-7B": "HuggingFace",
                        "embed-multilingual-v3.0": "Cohere",
                        "text-embedding-3-small": "OpenAI"}

    create_index(models=embedding_models, load_from_disc=True)

def single():
    models = [
        "intfloat/multilingual-e5-large-instruct"
    ]
    create_index(models, load_from_disc=True, device="cpu")

if __name__ == "__main__":
    main()
