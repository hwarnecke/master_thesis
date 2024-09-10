# testing the SentenceTransformerRerank
# using the local HF Embedding as a CrossEncoder

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, QueryBundle, StorageContext
from llama_index.core import Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.postprocessor.openvino_rerank import OpenVINORerank
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank


def old():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    documents = SimpleDirectoryReader(
        "../data"
    ).load_data()

    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    local_model = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    reranker = SentenceTransformerRerank(
        model=local_model, top_n=3
    )

    query_rerank = index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker], verbose=True)


    query_normal = index.as_query_engine(similarity_top_k=10)

    response_rerank = query_rerank.query("Which grad schools did the author apply for and why?")
    response_normal = query_normal.query("Which grad schools did the author apply for and why?")

    """
    print("Rerank Answer:")
    print(response_rerank.response)
    print("Rerank Sources:")
    print(response_rerank.get_formatted_sources(length=200))
    
    print("Normal:")
    print(response_normal.response)
    print("Normal Sources:")
    print(response_normal.get_formatted_sources(length=200))
    """

    """
    this is a good way of getting the source nodes from the response object while actually using the query engines.
    """

    print(type(response_rerank))

    #print(response_rerank.get_formatted_sources(length=2000))
    print(len(response_rerank.source_nodes))

    for sources in response_rerank.source_nodes:
        print(sources.node.get_text())


    """ appearantly it can also be used as standalone"""
    print("\n\n")
    query_str = QueryBundle("Which grad schools did the author apply for and why?")
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query_str)
    reranked_nodes = reranker.postprocess_nodes(nodes, query_str)

    print(len(nodes))
    print(len(reranked_nodes))

def reranker():
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    vector_store_name = "service_" + "embed-multilingual-v3.0"
    Settings.embed_model = CohereEmbedding(model_name="embed-multilingual-v3.0", api_key=api_key)
    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url="http://localhost:9200",
    )
    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)

    local_model = "jinaai/jina-reranker-v2-base-multilingual"


    # both work with sentence transformer and openvino
    success = ["BAAI/bge-reranker-v2-m3",
               "ml6team/cross-encoder-mmarco-german-distilbert-base"]

    creating_fail = ["Alibaba-NLP/gte-multilingual-reranker-base",  # export custom architecture but works with sentence transformer instead
                     "castorini/monot5-base-msmarco",               # unsupported task but works with sentence transformer instead
                     "jinaai/jina-reranker-v2-base-multilingual"    # export custom architecture | sentence transformer: data did not match any variant of untagged enum PyPreTokenizerTypeWrapper
                     ]

    executing_fail = [
        "BAAI/bge-reranker-v2-gemma",                     # sum() received an invalid combination of arguments but works with sentence transformer
        "cross-encoder/msmarco-MiniLM-L12-en-de-v1",      # shape infer input shapes dim index: 1 mismatch but works with sentence transformer
        "deepset/gbert-base-germandpr-reranking",         # works with Colbert | sum() received an invalid combination of arguments | The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    ]

    # all models that I could run with openvino where also possible with sentence transformer and that is less hassle
    models = {
        "BAAI/bge-reranker-v2-m3": "sentenceTransformer",
        "BAAI/bge-reranker-v2-gemma": "sentenceTransformer",
        "ml6team/cross-encoder-mmarco-german-distilbert-base": "sentenceTransformer",
        "Alibaba-NLP/gte-multilingual-reranker-base": "sentenceTransformer",
        "castorini/monot5-base-msmarco": "sentenceTransformer",
        "cross-encoder/msmarco-MiniLM-L12-en-de-v1": "sentenceTransformer",
        "deepset/gbert-base-germandpr-reranking": "colbert",
        "jina-reranker-v2-base-multilingual": "jina"
    }

    other = ["Alibaba-NLP/gte-multilingual-reranker-base"]
    error_messages = []
    for model in other:
        # reranker = sentenceTransformer(model)
        try:
            reranker = sentenceTransformer(model)
        except Exception as e:
            print(f"Creating Model: {model} failed.")
            error_messages.append(e)
            continue

        query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[reranker])

        query = "Wo Zoo?"

        try:
            answer = query_engine.query(query)
            print(f"Model {model} successful.")
        except Exception as e:
            print(f"Executing Model: {model} failed.")
            error_messages.append(e)

    for error in error_messages:
        print(error)

def openvino(model_name: str):
    OpenVINORerank.create_and_save_openvino_model(
        model_name, "./rerank_ov", export_kwargs={'trust_remote_code': True}
    )
    ov_rerank = OpenVINORerank(
        model_id_or_path="./rerank_ov", device="cpu", top_n=2
    )
    return ov_rerank

def flagEmbedding(model_name: str):
    rerank = FlagEmbeddingReranker(model=model_name, top_n=3)
    return rerank

def sentenceTransformer(model_name: str):
    reranker = SentenceTransformerRerank(
        model=model_name, top_n=3, device="cpu"
    )
    return reranker

def colbert(model_name: str):
    colbert_reranker = ColbertRerank(
        top_n=3,
        model=model_name,
        keep_retrieval_score=True,
    )
    return colbert_reranker

def jina(model_name: str):
    api_key = "jina_35f0c554616a4f3f857030d3f2d90b06rYfWQvbtCg0bw-tPhS9Wzp7reEQe"
    jina_rerank = JinaRerank(api_key=api_key, top_n=3, model=model_name)
    return jina_rerank

if __name__ == "__main__":
    reranker()