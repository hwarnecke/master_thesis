# testing different features of the OpenAI API

from dotenv import load_dotenv
import os
import tiktoken
from openai import OpenAI
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.openai import OpenAI as llamaOpenAI

"""
testing if the token counter can also count the OpenAI calls.
Since the FusionRetriever is doing this independently.
Edit: it cannot track that
"""
def main():
    # load api key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    # Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)

    # create a token counter
    # verbose means it prints out the token to the console
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-40-mini").encode,
        verbose=False,
    )

    # add the token counter to the callback manager
    Settings.callback_manager = CallbackManager([token_counter])

    client = OpenAI(api_key=api_key)

    while True:
        print("You:")
        question = input()
        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
        )
        print("GPT:" + chat_response.choices[0].message.content)

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

"""
I can use the LlamaIndex integration of the OpenAI library to include the token counter
"""
def LlamaTest():
    # Load API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Set the LLM
    Settings.llm = llamaOpenAI(model="gpt-34o-mini", api_key=api_key)

    # Create a token counter
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-4o-mini").encode,
        verbose=False,
    )

    # Add the token counter to the callback manager
    Settings.callback_manager = CallbackManager([token_counter])

    # Send a prompt to the LLM
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    response = Settings.llm.complete(prompt)

    # Print the response
    print(response)
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


def response_test():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Set the LLM
    Settings.llm = llamaOpenAI(model="gpt-3.5-turbo", api_key=api_key)
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    response = Settings.llm.complete(prompt)
    first_word = response.text.split(" ")[0]
    print(first_word)

def stop_words():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = llamaOpenAI(model="gpt-3.5-turbo", api_key=api_key)
    prompt = "translate the following english text into german: 'Hello, how are you?'"
    stop_words = ["wie"]
    response = Settings.llm.complete(prompt, stop=stop_words)
    print(response)

def new_model():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = llamaOpenAI(model = "gpt-4o-mini")
    response = Settings.llm.complete("Who are you?")
    print(response)


if __name__ == "__main__":
    new_model()
