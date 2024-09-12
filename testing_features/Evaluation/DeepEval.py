"""
DeepEval is a test suite for evaluating the performance of various LLM applications.
It also includes a lot of metrics specific for RAG evaluation.
It would also support RAGAs if needed but maybe their native evaluation metrics are enough or even better.
"""
from deepeval.metrics import AnswerRelevancyMetric
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import Response
from llama_index.llms.cohere import Cohere

# there is a direct support for llamaindex in deepeval
# using these we can directly use the response objects that llamaindex returns
from deepeval.integrations.llama_index import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalContextualRelevancyEvaluator,
    DeepEvalSummarizationEvaluator,
    DeepEvalBiasEvaluator,
    DeepEvalToxicityEvaluator,
)
from DeepEvalCustomLLM import DeepEvalCustomLLM
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext



def old():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    documents = SimpleDirectoryReader("../data").load_data()

    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    application = index.as_query_engine()

    user_input = "Which grad schools did the author apply for and why?"

    response_object = application.query(user_input)

    # for a full list and explanations of the metrics look here: https://docs.confident-ai.com/docs/integrations-llamaindex
    answerRelevancyEvaluator = DeepEvalAnswerRelevancyEvaluator()
    faithfulnessEvaluator = DeepEvalFaithfulnessEvaluator()
    contextualRelevancyEvaluator = DeepEvalContextualRelevancyEvaluator()
    summarizationEvaluator = DeepEvalSummarizationEvaluator()
    biasEvaluator = DeepEvalBiasEvaluator()
    toxicityEvaluator = DeepEvalToxicityEvaluator()

    # summarization Evaluator is probably not needed for my use case
    # bias Evaluator is probably also irrelevant, but I am interested what it does to my use case
    # toxcicity Evaluator is  interesting but also not necessary for my use case
    # Note: when I tested the reranking approach I got two completely different answers to the question.
    #       but this one got perfect scores for the three main metrics. This shows that I should at best include a
    #       reference answer, so I can also check if the answer is correct.
    #       It can very well be that answers are very good, in terms of context-answer relation but not correct because
    #       the context is wrong.

    evaluators = {
        "answerRelevancyEvaluator": answerRelevancyEvaluator,
        "faithfulnessEvaluator": faithfulnessEvaluator,
        "contextualRelevancyEvaluator": contextualRelevancyEvaluator,
        # "summarizationEvaluator": summarizationEvaluator,
        # "biasEvaluator": biasEvaluator,
        # "toxicityEvaluator": toxicityEvaluator,
    }

    results = []
    for evaluator in evaluators:
        # print(evaluator)
        # evaluation_result = evaluators[evaluator].evaluate_response(query=user_input, response=response_object)
        # print(evaluation_result.score)
        # results.extend([evaluation_result.passing, evaluation_result.feedback, evaluation_result.score])
        # print("\n")
        name = str(evaluator)
        # print(type(name))
        print(name)

    # print(results)

def custom_response():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
    answerRelevancyEvaluator = DeepEvalAnswerRelevancyEvaluator()
    query = "Das ist eine Frage?"
    response = Response(response="Hallo", source_nodes=["strings"])
    evaluation_result = answerRelevancyEvaluator.evaluate_response(query=query, response=response)
    print(evaluation_result.score)

def custom_llm():
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    llm = Cohere(api_key=cohere_api_key, model="command-r-plus")
    Settings.llm = llm
    custom_llm = DeepEvalCustomLLM(llm=llm)

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

    query_engine = index.as_query_engine(similarity_top_k=3, verbose=True)

    answerRelevancyEvaluator = DeepEvalAnswerRelevancyEvaluator(model=custom_llm)

    query = "Wie teuer ist ein Personalausweis?"
    response = query_engine.query(query)

    evaluation = answerRelevancyEvaluator.evaluate_response(query=query, response=response)
    print(evaluation)



if __name__ == "__main__":
    custom_llm()

