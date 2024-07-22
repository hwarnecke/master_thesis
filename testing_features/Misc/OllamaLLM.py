from llama_index.llms.ollama import Ollama
import time

def main():
    start = time.time()
    llm = Ollama(model="sauerkraut_q4", request_timeout=500.0)
    loaded = time.time()
    print(f"model loaded after {loaded - start} seconds.")
    resp = llm.complete("Who is Paul Graham? Answer in 5 words.")
    completed = time.time()
    print(f"Response ready after {completed - loaded} seconds")
    print(resp)

if __name__ == "__main__":
    main()