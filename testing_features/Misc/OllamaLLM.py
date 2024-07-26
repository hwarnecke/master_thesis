from llama_index.llms.ollama import Ollama
import time
from transformers import AutoTokenizer
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings, get_response_synthesizer, StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore

def main():
    start = time.time()
    llm = Ollama(model="sauerkraut_hero", request_timeout=500.0)
    loaded = time.time()
    print(f"model loaded after {loaded - start} seconds.")
    resp = llm.complete("Wer ist Paul Graham?")
    completed = time.time()
    print(f"Response ready after {completed - loaded} seconds")
    print(resp)

def token_counting():
    token_counter = TokenCountingHandler(
        #tokenizer=AutoTokenizer.from_pretrained("VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct"),
        verbose=False,
    )
    Settings.callback_manager = CallbackManager([token_counter])

    llm = Ollama(model="sauerkraut_q4", request_timeout=500.0)
    resp = llm.complete("Who is Paul Graham? Answer in 5 words.")
    print(f"Response: {resp}")

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



if __name__ == "__main__":
    main()