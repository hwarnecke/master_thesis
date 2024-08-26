from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from dotenv import load_dotenv
import os
from llama_index.core import Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext

def loadCohere():
    load_dotenv()
    return os.getenv("COHERE_API_KEY")

def llm():
    api_key = loadCohere()
    Settings.llm = Cohere(api_key=api_key, model="command-r-plus")
    response = Settings.llm.complete("Wie teuer ist ein Personalausweis?")
    print(response)

def embedding():
    api_key = loadCohere()
    embed_model = CohereEmbedding(
        api_key=api_key,
        model_name="embed-multilingual-v3.0",
        input_type="search_query",
    )

    embeddings = embed_model.get_text_embedding("Hallo, wohin des Wegs die Dame?")

    print(len(embeddings))
    print(embeddings[:5])

def reranker():
    api_key = loadCohere()
    Settings.llm = Cohere(api_key=api_key, model="command-r-plus")
    cohere_rerank = CohereRerank(api_key=api_key, top_n=2, model="rerank-multilingual-v3.0")
    embedding_name = "intfloat/multilingual-e5-large-instruct"
    embedding_model = HuggingFaceEmbedding(model_name=embedding_name)
    Settings.embed_model = embedding_model

    # create an Elasticsearch index
    # don't forget to start the docker container with the Elasticsearch instance

    es_vector_store = ElasticsearchStore(
        index_name="service_multilingual-e5-large-instruct",
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=True)

    query_rerank = index.as_query_engine(similarity_top_k=10, node_postprocessors=[cohere_rerank], verbose=True)

    query = "Welche Fristen muss ich beim Anmelden von Osterfeuern beachten?"

    response = query_rerank.query(query)
    print(response)

if __name__ == "__main__":
    reranker()