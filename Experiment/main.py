from CreateQueryEngines import create_query_engines
from DataLogging import DataLogging
import json, tiktoken, os
from deepeval.integrations.llama_index import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalContextualRelevancyEvaluator,
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings

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

    """
    create the evaluators and the token counter
    """
    # the evaluators
    # TODO: find the evaluator that compares the two answers against each other
    threshold = 0.5
    eval_model = "gpt-4"
    include_reason = True
    answerRelevancyEvaluator = DeepEvalAnswerRelevancyEvaluator(threshold=threshold,
                                                                model=eval_model,
                                                                include_reason=include_reason)

    faithfulnessEvaluator = DeepEvalFaithfulnessEvaluator(threshold=threshold,
                                                          model=eval_model,
                                                          include_reason=include_reason)

    contextualRelevancyEvaluator = DeepEvalContextualRelevancyEvaluator(threshold=threshold,
                                                                        model=eval_model,
                                                                        include_reason=include_reason)

    evaluators = [answerRelevancyEvaluator, faithfulnessEvaluator, contextualRelevancyEvaluator]

    # the count is only reset manually, otherwise it would accumulate over multiple queries
    token_counter.reset_counts()

    """
    2. Run the experiment

    The experiment will consist of running a set of questions through the different retrievers
    This means, iterating through the list of query engines and query each with the same set of questions.
    For each question the response objects will be put into the DeepEval metrics and the results will be saved.
    I will also save the question, the response, the tokens used and the retrieved nodes.
    Additionally, the time it took to answer the question will be measured.
    All these values will be saved to a csv file with a retriever ID.
    Some retrievers like the RAG Fusion or the HyDE will have some additional information I want to look at later.
    I will create a separate file for these as they will have a different structure 
    and can't be easily compared to the other retrievers.
    """
    print("Starting Experiment")
    current_qe: int = 0
    total_amount = len(query_engines)
    total_amount_questions = len(questions)
    total_amount_evaluators = len(evaluators)

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

            response = qe.query(query)
            nodes: dict[str, str] = extract_source_nodes(response)

            correct_answer = question["answer"]

            # save the information in a dictionary
            info = {"ID": qe_id.split("/")[1],  # ignore the part of the ID that is used for the directory
                    "query": query,
                    "response": response,
                    "correct_answer": correct_answer,
                    "query_time": qe.query_time,
                    "generating_time": qe.generating_time,
                    "total_time": qe.total_time
                    }

            # collect additional data if necessary and log them in a separate file
            # add_path: str = path + "additional_data"
            base_name, extension = os.path.splitext(path)
            add_path: str = f"{base_name}_additional_data{extension}"
            add_data: dict = {}

            if "auto" in qe_id:
                add_data: dict = qe.verbose_output
            elif "fusion" in qe_id:
                add_data: dict = qe.retriever.generated_questions
            elif "hyde" in qe_id:
                add_data: dict = qe.hyde_object

            if add_data:
                data_logger.write_csv(add_data, add_path)

            print("\t\tDone querying. Starting Evaluation.")
            # collect token counts
            token_embeddings = token_counter.total_embedding_token_count
            token_prompt = token_counter.prompt_llm_token_count
            token_completion = token_counter.completion_llm_token_count
            token_total = token_counter.total_llm_token_count
            tokens = {"embedding_tokens": token_embeddings,
                      "prompt_tokens": token_prompt,
                      "completion_tokens": token_completion,
                      "total_tokens": token_total}
            # for testing
            token_counter.reset_counts()  # do not forget to reset the counts!

            # evaluate the response
            evaluation = {}
            current_evaluator = 0
            for evaluator in evaluators:
                current_evaluator += 1
                # turn the results into a dictionary based on the name of the evaluator
                eval_name = str(evaluator)
                eval_name = eval_name.replace("Evaluator", "")
                # in my initial tests this returns only the name itself
                # but if I run it here it will save the whole type in the csv, so I need to trim it further
                eval_name = eval_name.split("evaluators.")[1]
                eval_name = eval_name.split(" object")[0]

                print(f"\t\tStarting with Evaluator {current_evaluator} out of {total_amount_evaluators}: {eval_name}.")
                evaluation_result = evaluator.evaluate_response(query=query, response=response)

                results = {eval_name + "_passing": evaluation_result.passing,
                           eval_name + "_feedback": evaluation_result.feedback,
                           eval_name + "_score": evaluation_result.score}

                # all results from all evaluators can be stored in the same dictionary
                evaluation.update(results)

            # save the information to disk
            data = {}
            data.update(info)
            data.update(nodes)
            data.update(tokens)
            data.update(evaluation)
            data_logger.write_csv(data)


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


if __name__ == "__main__":
    run_experiment(use_query_engines=["base"])
