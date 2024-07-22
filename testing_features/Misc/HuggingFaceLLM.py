from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer

def main():
    # hopefully this loads from the hub files I already downloaded
    locally_run = HuggingFaceLLM(model_name="VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct")

    # in their tutorial they say to change the global tokenizer to fit the model, not sure what this does exactly
    # using or ignoring this seems to have the same output
    # set_global_tokenizer(
    #     AutoTokenizer.from_pretrained("VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct").encode
    # )

    # somehow he loads the model but does not do any completion
    completion_response = locally_run.complete("Was ist der Sinn des Lebens?")
    print(completion_response)

if __name__ == "__main__":
    main()