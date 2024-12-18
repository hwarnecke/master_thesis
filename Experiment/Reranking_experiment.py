from datetime import datetime

from dotenv import load_dotenv
import main
import CreateQueryEngines
import ModifiedQueryEngine
from llama_index.core import Settings, get_response_synthesizer, VectorStoreIndex, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
import json
import DataLogging

"""
The main experiment tests each query engine in isolation. 
Meaning that if I run multiple rerankers, it will retrieve a new set of documents for each.
Usually, the same documents are retrieved but differences are possible.
This spin-off of the main experiment retrieves a single set of documents 
and then tests all rerankers on the same set.
"""


def reranking_comparison():
    load_dotenv()
    # create reranker
    reranking_models = {
        "BAAI/bge-reranker-v2-m3": "sentenceTransformer",
        "BAAI/bge-reranker-v2-gemma": "sentenceTransformer",
        "ml6team/cross-encoder-mmarco-german-distilbert-base": "sentenceTransformer",
        "Alibaba-NLP/gte-multilingual-reranker-base": "sentenceTransformer",
        "castorini/monot5-base-msmarco": "sentenceTransformer",
        "cross-encoder/msmarco-MiniLM-L12-en-de-v1": "sentenceTransformer",
        "deepset/gbert-base-germandpr-reranking": "colbert",
        "jina-reranker-v2-base-multilingual": "jina",
        "rerank-multilingual-v3.0": "cohere"
    }

    reranking_shorts = [
        "no_reranking",
        "bge_m3",
        "bge_gemma",
        "cross",
        "gte",
        "monot5",
        "MiniLM",
        "gbert",
        "JinaAI",
        "Cohere"
    ]

    reranker = []
    for model, type in reranking_models.items():
        rerank = CreateQueryEngines.get_Reranker(type=type, rerank_model=model, rerank_top_n=3)
        reranker.append(rerank)

    # create Query Engine
    embedding_name = "text-embedding-3-small"
    CreateQueryEngines.get_LLM(type="OpenAI",llm="gpt-4o-mini")
    embedding_model, embedding_id = CreateQueryEngines.get_Embedding(type="OpenAI", embedding_name=embedding_name)
    Settings.embed_model = embedding_model

    vector_store_name = "service_" + embedding_name.lower()
    embedding_url = "http://localhost:9200"
    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url=embedding_url,
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    basic_response_synthesizer = get_response_synthesizer(response_mode="no_text")
    retriever_top_k = 20
    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)
    basic_retriever = index.as_retriever(similarity_top_k=retriever_top_k)

    query_engine = ModifiedQueryEngine.ModifiedQueryEngine(retriever=basic_retriever, response_synthesizer=basic_response_synthesizer)

    # load questions
    question_file = "questions_extended.json"
    questions = json.load(open(question_file))

    # prepare data logging
    current_time = datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    path = "logs/" + "rerank_experiment" + "_" + timestamp + ".csv"
    data_logger = DataLogging.DataLogging(file_path=path)

    # run experiment
    question_counter = 0
    for question in questions:
        question_counter += 1
        print(f"Running Question: {question_counter}")
        responses, times = query_engine.reranker_test(query_str=question["question"], reranker=reranker)

        info = {}
        # extract for logging
        for i in range(len(responses)):
            print(f"\tRunning reranker: {i+1} of {len(responses)}")
            node_info = main.create_context_log(responses[i], identifier=reranking_shorts[i])
            info.update(node_info)
            time_log = {f"{reranking_shorts[i]}_query_time": times[i]}
            info.update(time_log)
            print(time_log)

        data_logger.write_csv(info)


if __name__ == '__main__':
    reranking_comparison()