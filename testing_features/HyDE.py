from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core import Settings, get_response_synthesizer
from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

# some settings
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
Settings.llm = OpenAI(model="gpt-3.5", api_key=api_key)

# load documents and build index and query engine
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
query_engine = index.as_query_engine()

# run query with HyDE transform
query_str = "Which grad schools did the author apply for and why?"
hyde = HyDEQueryTransform(include_original=True)
query_engine = TransformQueryEngine(query_engine, query_transform=hyde)

response = query_engine.query(query_str)
print(response)

# check what the LLM halucinated for the HyDE transform
query_bundle = hyde(query_str)
hyde_doc = query_bundle.embedding_strs[0]
print(hyde_doc)

"""
this seems to be a simple way to use HyDE with the llama_index.
In the code there are also ways to change the prompt template if wanted.
But this way I am limited to create a query engine with the HyDE transform, which might be a bit to high level for me.
I.e. it doesn't provide a convenient way to directly save the halucinated document, instead you have to rerun the
transform on the query string, but this makes a new call to the LLM which is unnecessary and might give me different 
answers.
Additionally, it might get difficult to use the HyDE transform in combination with other retrieval methods this way.

I might be better of by implementing HyDE myself and just throw the generated answer into the standard retriever and
synthesizer, ignoring the query engine all together.
I can even use the HyDEQueryTransform to generate the response since it's generating a query_bundle which is basically
a tuple of the original query string and the halucinated document.

But I'm not sure if that is really the same, as I read somewhere that the document is embedded together with the query
or something like that, might have to check that again.
"""


# a potential low level implementation of HyDE
# generating the response myself and putting it into a normal retriever

# generate the hypothetical document
query_str = "Which grad schools did the author apply for and why?"
hyde = HyDEQueryTransform(include_original=True)
query_bundle = hyde(query_str)

# create a basic retriever
retriever = index.as_retriever(similarity_top_k=5)
context = retriever.retrieve(query_bundle.embedding_strs[0])

# create a response synthesizer and get a response
response_synthesizer = get_response_synthesizer()
response = response_synthesizer.synthesize(query_str, nodes=context)

print(response)

"""
Implementing it myself seemed to work fine.
Might want to use that to have a finer control over the process.
"""