# testing to load a custom embedding model from HuggingFace
# this is important as the application and data will be in german, so a german embedding model should be used

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import Document
import os

def old():
    model_name = "intfloat/multilingual-e5-large-instruct"

    embed_model = HuggingFaceEmbedding(model_name=model_name)

    # test if it worked
    embeddings = embed_model.get_text_embedding("Hallo Welt!")
    print(len(embeddings))
    print(embeddings[:5])


def get_documents():
    text_list = [
        "Hier findest du etwas deutschen Text.",
        "Berlin ist die Hauptstadt von Deutschland.",
        "Das Grundgesetz ist jetzt 75 Jahre alt."
    ]
    documents = [Document(text=t) for t in text_list]
    return documents

def get_test_documents():
    # Example string of length 512
    string_512: str = "a" * 512
    string2_512: str = "b" * 512
    doc_512 = Document(text = string_512)
    doc2_512 = Document(text = string2_512)

    # Example string of length 1024
    string_1024: str = "c" * 1000
    string2_1024: str = "b" * 1024
    doc_1024 = Document(text = string_1024)
    doc2_1024 = Document(text = string2_1024)

    documents = [doc_1024]
    return documents

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    # Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key)
    gpt35_llm = OpenAI(model="gpt-3.5-turbo")

    model_name = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    #model_name = "jinaai/jina-embeddings-v2-base-de"
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.embed_model = embed_model


    #documents = get_documents()
    documents = get_test_documents()
    # documents = SimpleDirectoryReader(
    #     "../data"
    # ).load_data()

    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    query_engine = index.as_query_engine(similarity_top_k=3, llm=gpt35_llm)

    query = "Wie alt ist das Grundgesetz?"
    #response = query_engine.query(query)
    #print(response)

if __name__ == "__main__":
    old()
