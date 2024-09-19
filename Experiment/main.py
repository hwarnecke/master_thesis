import time
from typing import Dict

from CreateQueryEngines import create_query_engines
from DataLogging import DataLogging
import json, os
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from DeepEvalCustomLLM import DeepEvalCustomLLM
from dotenv import load_dotenv
from llama_index.llms.cohere import Cohere

"""
This is  the main file to run the experiment.
It will create the different kind of retrievers / query engines,
test them on a set of questions,
evaluate the results
and save them to disk.
"""


def run_experiment(questions: str = "questions_extended.json",
                   custom_qa_path: str = None,
                   custom_refine_path: str = None,
                   embedding: str = "text-embedding-3-small",
                   embedding_type: str = "OpenAI",
                   llm: str = "gpt-4o-mini",
                   llm_type: str = "OpenAI",
                   rerank_top_n: int = 3,
                   rerank_model: str = "rerank-multilingual-v3.0",
                   rerank_type: str = "cohere",
                   retrieval_top_k: int = 12,
                   use_query_engines: list[str] = None,
                   evaluate: bool = False,
                   response_mode: str = "refine"
                   ):
    """
    Set up and run the experiment. The following parameters can be changed:
    :param questions: path of the json file containing the questions
    :param custom_qa_path: path to the prompt template to use
    :param custom_refine_path: path to the prompt template to use
    :param embedding: name of the HF embedding to use (as default text-embedding-ada-002 is used)
    :param llm: name of the llm to use (currently only OpenAI models are supported)
    :param rerank_top_n: how many documents the reranker should choose (default is 3)
    :param rerank_model: which model to use, currently limited to some HuggingFace models
    :param retrieval_top_k: how many documents the retriever should fetch (default is 6)
    :param use_query_engines: list of names if only some specific query engines should be used instead of all of them
    :param response_mode: refine for normal behaviour or no_text for skipping the response synthesizer (only retrieval)
    :return: No return
    """

    """
    Initialize variables and stuff
    """

    # the token counter needs to be initialized first, before the query engines
    # otherwise it will log nothing
    # also, it seems like using the default tokenizer might work best for using different models, even if it isn't
    # perfect for all of them
    token_counter = TokenCountingHandler(
        #tokenizer=tiktoken.encoding_for_model(llm).encode,
        verbose=False,
    )
    Settings.callback_manager = CallbackManager([token_counter])

    # load custom prompt if needed
    custom_qa_content = None
    custom_refine_content = None
    if custom_qa_path:
        with open(custom_qa_path, "r") as file:
            custom_qa_content = file.read()
    if custom_refine_path:
        with open(custom_refine_path, "r") as file:
            custom_refine_content = file.read()

    # Create the query engines
    # for a more detailed description of the query engines and the possible variations that can be made,
    # see the CreateQueryEngines.py file
    print("Creating the Query Engines and setting up the experiment")

    if use_query_engines is None:
        use_query_engines = ["base", "rerank", "hybrid", "auto", "hyde", "fusion", "agent", "iter-retgen"]

    query_engines = create_query_engines(llm=llm,
                                         llm_type=llm_type,
                                         embedding_name=embedding,
                                         embedding_type=embedding_type,
                                         rerank_top_n=rerank_top_n,
                                         retriever_top_k=retrieval_top_k,
                                         rerank_model= rerank_model,
                                         rerank_type=rerank_type,
                                         custom_qa_prompt=custom_qa_content,
                                         custom_refine_prompt=custom_refine_content,
                                         use_query_engines=use_query_engines,
                                         response_mode=response_mode)

    print("The following query engines have been chosen:")
    count: int = 0
    for name, qe in query_engines.items():
        count += 1
        print(f"{count}: {name.split('/')[1].split('_2024')[0]}")

    # load the questions
    # currently the idea is to store them as a JSON in the format of a list of dictionaries
    questions = json.load(open(questions))

    # I ditched the Llamaindex native integration of deepeval in order to include the agent
    metrics = create_metrics()

    # the count is only reset manually, otherwise it would accumulate over multiple queries
    token_counter.reset_counts()

    """
    Running the experiment.
    """

    print("Starting Experiment")
    current_qe: int = 0
    total_amount = len(query_engines)
    total_amount_questions = len(questions)

    for qe_id, qe in query_engines.items():
        current_qe += 1
        # create the data logging object
        path = "logs/" + qe_id + ".csv"
        data_logger = DataLogging(file_path=path)

        # tqdm looked a bit ugly, so I'm creating my own outputs
        qe_name = qe_id.split("/")[1].split("_2024")[0]
        print(f"Now starting Query Engine {current_qe} of {total_amount}: {qe_name}.")
        current_question = 0

        for question in questions:
            current_question += 1
            print(f"\tAsking question {current_question} of {total_amount_questions}")

            query = question["question"]

            # cohere API key is limited to 10 calls per minute, so I need to add a delay here if I use it
            # which also means the time comparisons are off for cohere, but I guess I can just subtract that later
            if rerank_model == "rerank-multilingual-v3.0":
                time.sleep(7)

            query_attempt: int = 0
            max_query_attempts = 10
            while query_attempt < max_query_attempts:
                try:
                    response = qe.query(query)  # the actual query call
                    break
                except Exception as e:
                    query_attempt += 1
                    print(f"\t\tQuery failed with: {e}")
                    print("\t\tRetrying in 3 seconds")
                    time.sleep(3)

            if "agent" in qe_id:
                agent_nodes = qe.get_nodes()
                nodes: dict[str, str] = create_agent_log(agent_nodes, ret_top_k=rerank_top_n)
            else:
                nodes: dict[str, str] = create_context_log(response)

            correct_answer = question["answer"]

            # save the information in a dictionary
            info = {"ID": qe_id.split("/")[1],  # ignore the part of the ID that is used for the directory
                    "query": query,
                    "response": response,
                    "correct_answer": correct_answer
                    }
            times: dict[str, float] = qe.get_time()
            info.update(times)

            # collect additional data if necessary and log them in a separate file
            base_name, extension = os.path.splitext(path)
            add_path: str = f"{base_name}_additional_data{extension}"
            add_data: dict[str, any] = create_additional_log(qe_id=qe_id, qe=qe)
            if add_data:
                data_logger.write_csv(add_data, add_path)

            print("\t\tDone querying.")

            tokens = collect_tokens(token_counter)
            # collect information into a single dict
            data = {}
            data.update(info)
            data.update(nodes)
            data.update(tokens)

            # evaluation is toggleable, because it takes a lot of API tokens and is not necessary for testing
            if evaluate:
                print("\t\tStarting Evaluation.")
                context: list = extract_context(response)
                evaluation = evaluate_response(metrics=metrics,
                                               input=query,
                                               actual_output=response,
                                               retrieval_context=context)
                data.update(evaluation)
            else:
                print("\t\tSkipping Evaluation.")

            # save the information to disk
            data_logger.write_csv(data)


def create_additional_log(qe_id: str, qe) -> dict[str, any]:
    if "fusion" in qe_id:
        add_data: dict = qe.retriever.generated_questions
    elif "hyde" in qe_id:
        add_data: dict = qe.hyde_object
    elif "agent" in qe_id:
        add_data: dict = qe.verbose_output
    else:
        add_data = {}

    return add_data

def collect_tokens(token_counter) -> dict:
    """
    read the token counter and create a dictionary that is ready to be logged
    :param token_counter:
    :return:
    """
    token_embeddings = token_counter.total_embedding_token_count
    token_prompt = token_counter.prompt_llm_token_count
    token_completion = token_counter.completion_llm_token_count
    token_total = token_counter.total_llm_token_count
    tokens = {"embedding_tokens": token_embeddings,
              "prompt_tokens": token_prompt,
              "completion_tokens": token_completion,
              "total_tokens": token_total}

    token_counter.reset_counts()  # do not forget to reset the counts!
    return tokens

def create_metrics() -> list:
    """
    create a set of deepeval metrics for the evaluation.
    :return:
    """
    # set Command-R+ as eval model, maybe test a different one later on
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    llm = Cohere(api_key=cohere_api_key, model="command-r-plus")
    custom_llm = DeepEvalCustomLLM(llm=llm)

    answer_relevancy_metric = AnswerRelevancyMetric(model=custom_llm)
    faithfulness_metric = FaithfulnessMetric(model=custom_llm)
    #contextual_relevancy_metric = ContextualRelevancyMetric(model=custom_llm)
    metrics = [answer_relevancy_metric, faithfulness_metric]#, contextual_relevancy_metric]
    return metrics


def extract_context(response) -> list[str]:
    """
    extracts the context from the response object.
    differentiates between the qe and the agent by the type of the response object.
    Mainly used to pass the nodes to the deepeval metrics.
    :param response: the response object of the qe
    :return: a list of source nodes
    """
    if isinstance(response,dict):
        return response["observations"]
    else:
        return [node.get_content() for node in response.source_nodes]


def evaluate_response(metrics: list, input: str, actual_output: str, retrieval_context: list) -> dict:
    """
    evaluate a question on a set of deepeval metrics.
    :param metrics: the list of deepeval metrics to use
    :param input: the query
    :param actual_output: the response of the query engine
    :param retrieval_context: the nodes used to answer
    :return: the result in a dict that is ready to be logged
    """
    evaluation = {}
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    total_amount = len(metrics)
    current = 0
    for metric in metrics:
        current += 1
        name = metric.__name__ + "_metric"
        name = name.replace(" ", "_")
        print(f"\t\tStarting with Evaluator {current} out of {total_amount}: {name}.")

        # the metrics fail if the output is not a valid JSON format.
        # Since LLMs can sometimes be unpredictable in their outcome this usually fails at some point.
        # I hope that it is fixed by simply giving the LLM a few chances to generate a valid JSON,
        # but the amount of tries are limited in order to avoid an endless loop
        max_attempts = 500
        attempts = 0
        while attempts < max_attempts:
            try:
                metric.measure(test_case)
                break
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    raise e
                print(f"\t\t...Exception {e} occured, try again...")
                time.sleep(3)

        score = metric.score
        reason = metric.reason
        success = metric.success

        result = {name + "_success": success,
                  name + "_score": score,
                  name + "_reason": reason}
        evaluation.update(result)

    return evaluation


def create_context_log(response, identifier: str = "", ret_top_k: int = 3) -> dict[str, any]:
    """
    create a log item for the context information.
    :param response:
    :return:
    """
    n: int = 0
    source_nodes = {}
    if response is None:
        all_nodes = [None] * ret_top_k
    else:
        all_nodes = response.source_nodes

    for node in all_nodes:
        n += 1
        source_nodes.update(extract_from_node(node, n, identifier=identifier))



    return source_nodes


def create_agent_log(agent_nodes: list, ret_top_k: int = 3) -> dict[str, str]:
    """
    The Agent needs a few extra steps for data logging.
    It has a list of all response objects that were created in one run (one for each call to the query engine).
    But since the length of that list can vary, we need to pad the rest with placeholder values,
    otherwise the csv is off
    :param agent_nodes:
    :param ret_top_k:
    :return:
    """
    i: int = 0
    # log how many calls there were
    number_of_nodes = len(agent_nodes)
    source_nodes = {"Number of Calls": number_of_nodes}
    for response_object in agent_nodes:
        i += 1
        identifier: str = f"Call {i} "
        source_nodes.update(create_context_log(response_object, identifier))

    max_calls: int = 10     # depends on agent
    for i in range(max_calls):
        identifier: str = f"Call {i + 1} "
        if i < number_of_nodes:
            response_object = agent_nodes[i]
        else:
            response_object = None
        source_nodes.update(create_context_log(response_object, identifier, ret_top_k))

    return source_nodes

def extract_from_node(node, index, identifier: str = "") -> dict[str, str]:
    """
    :param node: LlamaIndex Node Object
    :param index: the index of the node
    :param identifier: in case of the agent, I might want to add from which call it is
    :return: the nodes as dict for data logging
    """
    number = f"{identifier}Node {index}"

    # create keys
    id_key = number + " ID"
    content_key = number + " content"
    score_key = number + " score"
    metadata_key = number + " Metadata: Name"

    # extract value or create placeholder
    # the placeholder is needed to ensure that the log for the agent has a consistent length
    # even if the amount of calls varies
    if node is None:
        placeholder: str = "_"
        id_value = placeholder
        content_value = placeholder
        score_value = placeholder
        metadata_content = placeholder
    else:
        id_value = node.id_
        content_value = node.get_text()
        score_value = node.get_score()
        metadata_content = node.metadata["Name"]

    node_dict = {id_key: id_value,
                 content_key: content_value,
                 metadata_key: metadata_content,
                 score_key: score_value}

    return node_dict




def compare_embeddings():
    """
    First find out which embedding works best.
    For that we only need a basic QE but all embeddings.
    Evaluation is probably excluded completely, because it can be easier done with a comparison of the nodes themselves.

    :return:
    """
    custom_qa_path = "PromptTemplates/german_qa_template.txt"
    custom_refine_path = "PromptTemplates/german_refine_template.txt"
    qes = ["base"]
    embedding_models = {
                        # "aari1995/German_Semantic_V3b": "HuggingFace",
                        # "T-Systems-onsite/cross-en-de-roberta-sentence-transformer": "HuggingFace",
                        # "jinaai/jina-embeddings-v2-base-de": "HuggingFace",
                        # "intfloat/multilingual-e5-large-instruct": "HuggingFace",
                        # "Alibaba-NLP/gte-multilingual-base": "HuggingFace",
                        # "dunzhang/stella_en_1.5B_v5": "HuggingFace",
                        "gritlm": "Ollama",
                        "embed-multilingual-v3.0": "Cohere",
                        "text-embedding-3-small": "OpenAI"}

    for model, type in embedding_models.items():
        run_experiment(custom_qa_path=custom_qa_path,
                       custom_refine_path=custom_refine_path,
                       questions="questions_extended.json",
                       evaluate=False,
                       response_mode="no_text",
                       embedding=model,
                       embedding_type=type,
                       use_query_engines=qes)


def main_experiment():
    # excluding iterative approaches for now, since I want to pick one of the below as the qe for them
    custom_qa_path = "PromptTemplates/german_qa_template.txt"
    custom_refine_path = "PromptTemplates/german_refine_template.txt"
    run_experiment(custom_qa_path=custom_qa_path,
                   custom_refine_path=custom_refine_path,
                   evaluate=False,
                   response_mode="no_text",
                   #use_query_engines=["base", "rerank", "hybrid"],
                   use_query_engines=["iter-retgen", "agent"],
                   retrieval_top_k=20,
                   rerank_top_n=3)


def reranker():
    custom_qa_path = "PromptTemplates/german_qa_template.txt"
    custom_refine_path = "PromptTemplates/german_refine_template.txt"

    reranking_models = {
        # "BAAI/bge-reranker-v2-m3": "sentenceTransformer",
        # "BAAI/bge-reranker-v2-gemma": "sentenceTransformer",
        # "ml6team/cross-encoder-mmarco-german-distilbert-base": "sentenceTransformer",
        "Alibaba-NLP/gte-multilingual-reranker-base": "sentenceTransformer",
        "castorini/monot5-base-msmarco": "sentenceTransformer",
        "cross-encoder/msmarco-MiniLM-L12-en-de-v1": "sentenceTransformer",
        "deepset/gbert-base-germandpr-reranking": "colbert",
        "jina-reranker-v2-base-multilingual": "jina",
        "rerank-multilingual-v3.0": "cohere"
    }

    for model, type in reranking_models.items():
        run_experiment(custom_qa_path=custom_qa_path,
                       custom_refine_path=custom_refine_path,
                       evaluate=False,
                       rerank_model=model,
                       rerank_type=type,
                       use_query_engines=["base", "rerank", "hybrid"],
                       response_mode="no_text",
                       retrieval_top_k=20,
                       rerank_top_n=3)


def llms():
    custom_qa_path = "PromptTemplates/german_qa_template.txt"
    custom_refine_path = "PromptTemplates/german_refine_template.txt"
    llms = {
        # "mistral": "Ollama",
        # "mixtral": "Ollama",
        # "command-r-plus": "Cohere",
        # "gpt-4o-mini": "OpenAI",
        # "wiedervereinigung_7b": "Ollama",
        # "sauerkraut_mixtral": "Ollama",
        # "sauerkraut_llama31_8b": "Ollama",
        # "llama31": "Ollama",
        # "sauerkraut_llama31": "Ollama",
        "llama31_8b": "Ollama",  # have to retry this!
    }

    failed = []
    for llm, type in llms.items():
        start_time = time.time()
        try:
            run_experiment(custom_qa_path=custom_qa_path,
                       custom_refine_path=custom_refine_path,
                       llm_type=type,
                       llm = llm,
                       evaluate=True,
                       use_query_engines=["rerank"],
                       retrieval_top_k=20,
                       rerank_top_n=3)
        except Exception as e:
            failed.append(llm)
            print(f"{llm} failed with {e}, trying next one.")
            time.sleep(3)

        stop_time = time.time() - start_time
        print(f"\nRunning this shit show took {stop_time} seconds.\n")

    print(f"{len(failed)} failed llms: {failed}")


def run_single(qe: str):
    custom_qa_path = "PromptTemplates/german_qa_template.txt"
    custom_refine_path = "PromptTemplates/german_refine_template.txt"
    run_experiment(custom_qa_path=custom_qa_path,
                   custom_refine_path=custom_refine_path,
                   llm_type="Ollama",
                   llm = "mixtral",
                   evaluate=True,
                   use_query_engines=[qe],
                   retrieval_top_k=20,
                   rerank_top_n=3)

if __name__ == "__main__":
    start_time = time.time()
    llms()
    stop_time = time.time() - start_time
    print(f"\nThis probably took longer than all Frodo scenes in LOTR...: {stop_time}s")
