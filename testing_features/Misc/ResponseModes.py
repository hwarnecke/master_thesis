from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

"""
Testing out how to change the prompt template for the response synthesizer
"""

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
#Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
gpt35_llm = OpenAI(model="gpt-3.5-turbo")

documents = SimpleDirectoryReader(
    "../data"
).load_data()

index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine(similarity_top_k=3, llm=gpt35_llm)

# return the current prompts
# defualt is the refine response mode, which includes a qa_template and a refine_template
prompt_dict = query_engine.get_prompts()

for k,p in prompt_dict.items():
    print(f"Name: {k}")
    print(f"Content: {p.get_template()}")
    print("\n")

# if I want to use custom prompts I can set them like this
# load the custom template for the qa part (the first query)
custom_qa_content = ""
with open("custom_qa_template.txt", "r") as file:
    custom_qa_content = file.read()
custom_qa_template = PromptTemplate(custom_qa_content)

# load the custom template for the refine part (the second query)
custom_refine_content = ""
with open("custom_refine_template.txt", "r") as file:
    custom_refine_content = file.read()
custom_refine_template = PromptTemplate(custom_refine_content)

query_engine.update_prompts(
    {
        "response_synthesizer:qa_template": custom_qa_template,
        "response_synthesizer:refine_template": custom_refine_template,
    }
)

