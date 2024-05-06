from io import StringIO

from CreateQueryEngines import create_query_engines
import json
import tqdm
import tiktoken
import time
import sys

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

        # TODO: add special cases for the RAG Fusion, AutoRetriever and the HyDE
        stdout_flag = False
        if qe_id.contains("auto") or qe_id.contains("fusion"):
            stdout_flag = True

        if stdout_flag:
            old_stdout = sys.stdout
            filter_info = StringIO()
            sys.stdout = filter_info



        # currently I can only record the complete response time (search + generation)
        # if I want a more granular approach, I need to create my own query engine
        query = question["question"]
        start_time = time.time()
        response = qe.query(query)
        end_time = time.time()
        response_time = end_time - start_time
        correct_answer = question["answer"]

        # reset the standard output in order to not overwrite the content in the variable
        if stdout_flag:
            sys.stdout = old_stdout

        """
        # I should really consider creating a custom query engine...
        # if not I need the transform object here to generate the hypothetical document...    
        if qe_id.contains("hyde"):
            query_bundle = hyde(query_str)
            hyde_doc = query_bundle.embedding_strs[0]
        """

        # collect token counts
        token_embeddings = token_counter.total_embedding_token_count
        token_prompt = token_counter.prompt_llm_token_count
        token_completion = token_counter.completion_llm_token_count
        token_total = token_counter.total_llm_token_count
        tokens = [token_embeddings, token_prompt, token_completion, token_total]
        token_counter.reset_counts()    # do not forget to reset the counts!

        # evaluate the response
        # TODO: this has likely not the correct structure, check what exactly is returned by the evaluators
        evaluation = []
        for evaluator in evaluators:
            evaluation_result = evaluator.evaluate_response(query=query, response=response)
            evaluation.append(evaluation_result)