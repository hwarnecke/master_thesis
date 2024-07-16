from CreateQueryEngines import create_query_engines
from DataLogging import DataLogging
import json, tiktoken, os
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from Agent import Response

"""
This is  the main file to run the experiment.
It will create the different kind of retrievers / query engines,
test them on a set of questions,
evaluate the results
and save them to disk.
"""


def run_experiment(questions: str = "questions.json",
                   custom_qa_path: str = None,
                   custom_refine_path: str = None,
                   embedding: str = "OpenAI/text-embedding-ada-002",
                   llm: str = "gpt-3.5-turbo",
                   rerank_top_n: int = 3,
                   retrieval_top_k: int = 6,
                   use_query_engines: list[str] = None
                   ):
    """
    Set up and run the experiment. The following parameters can be changed:
    :param questions: path of the json file containing the questions
    :param custom_qa_path: path to the prompt template to use
    :param custom_refine_path: path to the prompt template to use
    :param embedding: name of the HF embedding to use (as default text-embedding-ada-002 is used)
    :param llm: name of the llm to use (currently only OpenAI models are supported)
    :param rerank_top_n: how many documents the reranker should choose (default is 3)
    :param retrieval_top_k: how many documents the retriever should fetch (default is 6)
    :param use_query_engines: list of names if only some specific query engines should be used instead of all of them
    :return: No return
    """

    """
    Initialize variables and stuff
    """

    # the token counter needs to be initialized first, before the query engines
    # otherwise it will log nothing
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(llm).encode,
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
    all_query_engines = create_query_engines(llm=llm,
                                             embedding_name=embedding,
                                             rerank_top_n=rerank_top_n,
                                             retriever_top_k=retrieval_top_k,
                                             custom_qa_prompt=custom_qa_content,
                                             custom_refine_prompt=custom_refine_content)
    #
    query_engines = {}
    if use_query_engines:
        query_engines = {key: value for key, value in all_query_engines.items()
                         if any(name in key for name in use_query_engines)}
    else:
        query_engines.update(all_query_engines)

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

            # the auto retriever through an error because it tried using non-existent metadata as filter
            # I might be able to catch that for now and find a fix for that later:
            try:
                response = qe.query(query)
            except Exception as e:
                response = Response({"observation": f"Error: {str(e)}"})

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
            add_data: dict[str, any] = create_additional_log(qe_id=qe_id, qe=qe, response=response)
            if add_data:
                data_logger.write_csv(add_data, add_path)

            print("\t\tDone querying. Starting Evaluation.")

            tokens = collect_tokens(token_counter)

            context: list = extract_context(response)
            evaluation = evaluate_response(metrics=metrics,
                                           input=query,
                                           actual_output=response,
                                           retrieval_context=context)

            # save the information to disk
            data = {}
            data.update(info)
            data.update(nodes)
            data.update(tokens)
            data.update(evaluation)
            data_logger.write_csv(data)


def create_additional_log(qe_id: str, qe, response) -> dict[str, any]:
    if "auto" in qe_id:
        add_data: dict = qe.verbose_output
    elif "fusion" in qe_id:
        add_data: dict = qe.retriever.generated_questions
    elif "hyde" in qe_id:
        add_data: dict = qe.hyde_object
    elif "agent" in qe_id:
        add_data: dict = {"query": response["query"]}
        add_data["thought_process"] = response["thought_process"]
        add_data.update(response["retriever_log"])
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
    They use gpt-4o as standard evaluation model
    :return:
    """
    answer_relevancy_metric = AnswerRelevancyMetric()
    faithfulness_metric = FaithfulnessMetric()
    contextual_relevancy_metric = ContextualRelevancyMetric()
    metrics = [answer_relevancy_metric, faithfulness_metric, contextual_relevancy_metric]
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
        max_attempts = 6
        attempts = 0
        while attempts < max_attempts:
            try:
                metric.measure(test_case)
                break
            except ValueError as e:
                attempts += 1
                if attempts == max_attempts:
                    raise e
                print("\t\t...invalid JSON, try again...")

        score = metric.score
        reason = metric.reason
        success = metric.success

        result = {name + "_success": success,
                  name + "_score": score,
                  name + "_reason": reason}
        evaluation.update(result)

    return evaluation


def create_context_log(response) -> dict[str, str]:
    """
    create a log item for the context information.
    The agent will have some None values because it logs the observations as source nodes, instead of real source nodes
    :param response:
    :return:
    """
    if isinstance(response, dict):
        observations = extract_context(response)
        n: int = 0
        source_nodes = {}
        for observation in observations:
            n += 1
            number = f"Node {n}"
            id_key = number + " ID"
            id_value = None
            content_key = number + " content"
            content_value = observation
            score_key = number + " score"
            score_value = None
            metadata_key = number + " Metadata: Name"
            metadata_content = None
            node_dict = {id_key: id_value,
                         content_key: content_value,
                         metadata_key: metadata_content,
                         score_key: score_value}
            source_nodes.update(node_dict)
    else:
        source_nodes = extract_source_nodes(response)

    return source_nodes


# TODO: include some metadata (i.e. link) in the logs
#       in order to check if all nodes come from the same page
def extract_source_nodes(response) -> dict[str, str]:
    """
    :param response: LlamaIndex Response Object
    :return: the nodes as dict for data logging
    """
    n: int = 0
    source_nodes = {}
    all_nodes = response.source_nodes
    for node in all_nodes:
        n += 1
        number = f"Node {n}"
        # extract the ID
        id_key = number + " ID"
        id_value = node.node_id
        # the content
        content_key = number + " content"
        content_value = node.get_text()
        # the score
        score_key = number + " score"
        score_value = node.get_score()
        # and the name of the service the node came from
        metadata_key = number + " Metadata: Name"
        metadata_content = node.metadata["Name"]

        node_dict = {id_key: id_value,
                     content_key: content_value,
                     metadata_key: metadata_content,
                     score_key: score_value}
        source_nodes.update(node_dict)
    return source_nodes


def run_all():
    embedding_models = ["OpenAI/text-embedding-ada-002",
                        "jinaai/jina-embeddings-v2-base-de",
                        "intfloat/multilingual-e5-large-instruct",
                        "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
                        ]

    llms = ["gpt-3.5-turbo", "gpt-4"]

    custom_qa_path = "PromptTemplates/german_qa_template.txt"
    custom_refine_path = "PromptTemplates/german_refine_template.txt"

    for llm in llms:
        for embedding in embedding_models:
            run_experiment(custom_qa_path=custom_qa_path,
                           custom_refine_path=custom_refine_path,
                           embedding=embedding,
                           llm=llm)


def run_single_qe(name: str):
    custom_qa_path = "PromptTemplates/german_qa_template.txt"
    custom_refine_path = "PromptTemplates/german_refine_template.txt"

    run_experiment(custom_qa_path=custom_qa_path,
                   custom_refine_path=custom_refine_path)




if __name__ == "__main__":
    custom_qa_path = "PromptTemplates/german_qa_template.txt"
    custom_refine_path = "PromptTemplates/german_refine_template.txt"

    run_experiment(custom_qa_path=custom_qa_path,
                   custom_refine_path=custom_refine_path)
