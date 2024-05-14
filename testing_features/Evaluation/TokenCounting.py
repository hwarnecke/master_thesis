import tiktoken
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

"""
This script demonstrates how to use the TokenCountingHandler to count the number of tokens used in the LLM prompt and completion.
This is important to estimate the cost of each approach tested. More LLM calls might lead to better performance but also higher costs.
Using the pricing information I can calculate the exact cost and even compare GPT-3.5 with GPT-4.
"""



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

# create a token counter
# verbose means it prints out the token to the console
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
    verbose=False,
)

# add the token counter to the callback manager
Settings.callback_manager = CallbackManager([token_counter])

documents = SimpleDirectoryReader("../data").load_data()

index = VectorStoreIndex.from_documents(documents, show_progress=True)

print(token_counter.total_embedding_token_count)

# the count is only reset manually, otherwise it would accumulate over multiple queries
token_counter.reset_counts()

#response = index.as_query_engine().query("Which grad schools did the author apply for and why?")


# in my main experiment I don't use a query engine. I need to test if it can still track it then
# EDIT: that works fine, the issue must be somewhere else
retriever = index.as_retriever()
response_snyth = get_response_synthesizer()

query = "Which grad schools did the author apply for and why?"
response = response_snyth.synthesize(query, retriever.retrieve(query))

print(response)

# there are different counts for the prompt and the completion
print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
)