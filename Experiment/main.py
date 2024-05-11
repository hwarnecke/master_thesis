from CreateQueryEngines import create_query_engines
from DataLogging import DataLogging
import json, tqdm, sys, time, tiktoken
from io import StringIO
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

llm = "gpt-3.5-turbo"


# Create the query engines
# for a more detailed description of the query engines and the possible variations that can be made, see the CreateQueryEngines.py file
query_engines = create_query_engines(llm=llm)

# load the questions
# currently the idea is to store them as a JSON in the format of a list of dictionaries
questions = json.load(open("questions.json"))

"""
create the evaluators and the token counter
"""
# the evaluators
#TODO: find the evaluator that compares the two answers against each other
threshold = 0.5
eval_model = "gpt-4"
include_reason = True
answerRelevancyEvaluator = DeepEvalAnswerRelevancyEvaluator(threshold=threshold, model=eval_model, include_reason=include_reason)
faithfulnessEvaluator = DeepEvalFaithfulnessEvaluator(threshold=threshold, model=eval_model, include_reason=include_reason)
contextualRelevancyEvaluator = DeepEvalContextualRelevancyEvaluator(threshold=threshold, model=eval_model, include_reason=include_reason)

evaluators = [answerRelevancyEvaluator, faithfulnessEvaluator, contextualRelevancyEvaluator]

# the token counter
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model(llm).encode,
    verbose=False,
)
Settings.callback_manager = CallbackManager([token_counter])

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

for qe_id, qe in tqdm.tqdm(query_engines.items()):
    for question in tqdm.tqdm(questions):

        # create the data logging object
        path = "logs/" + qe_id + ".csv"
        data_logger = DataLogging(file_path=path)
        query = question["question"]

        response = qe.query(query)


        correct_answer = question["answer"]

        # save the information in a dictionary
        info = {"ID": qe_id,
                "query": query,
                "response": response,
                "correct_answer": correct_answer,
                "query_time": qe.query_time,
                "generating_time": qe.generating_time,
                "total_time": qe.total_time
                }

        # collect additional data if necessary and log them in a separate file
        add_path: str = path + "additional_data"
        add_data: dict = {}
        if qe_id.contains("auto"):
            add_data: dict = qe.verbose_output
        elif qe_id.contains("fusion"):
            add_data: dict = qe.retriever.generated_questions
        elif qe_id.contains("hyde"):
            add_data: dict = qe.hyde_object

        if add_data:
            data_logger.write_csv(add_data, add_path)

        # collect token counts
        token_embeddings = token_counter.total_embedding_token_count
        token_prompt = token_counter.prompt_llm_token_count
        token_completion = token_counter.completion_llm_token_count
        token_total = token_counter.total_llm_token_count
        tokens = {"embedding_tokens": token_embeddings,
                  "prompt_tokens": token_prompt,
                  "completion_tokens": token_completion,
                  "total_tokens": token_total}
        token_counter.reset_counts()    # do not forget to reset the counts!

        # evaluate the response
        evaluation = {}
        for evaluator in evaluators:
            evaluation_result = evaluator.evaluate_response(query=query, response=response)

            # turn the results into a dictionary based on the name of the evaluator
            eval_name = str(evaluator)
            eval_name = eval_name.replace("Evaluator", "")

            results = {eval_name + "_passing": evaluation_result.passing,
                       eval_name + "_feedback": evaluation_result.feedback,
                       eval_name + "_score": evaluation_result.score}

            # all results from all evaluators can be stored in the same dictionary
            evaluation.update(results)

        # save the information to disk
        data = {}
        data.update(info)
        data.update(tokens)
        data.update(evaluation)
        data_logger.write_csv(data)
