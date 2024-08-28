# testing to load a custom embedding model from HuggingFace
# this is important as the application and data will be in german, so a german embedding model should be used

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core import Document
import os

def old():
    model_name = "jinaai/jina-clip-v1"

    embed_model = HuggingFaceEmbedding(model_name=model_name, device="cpu", trust_remote_code=True, max_length=512)

    # test if it worked
    embeddings = embed_model.get_text_embedding("Die rasche Entwicklung vom Agrar- zum Industriestaat vollzog sich während der Gründerzeit in der zweiten Hälfte des 19. Jahrhunderts. Nach dem Ersten Weltkrieg wurde 1918 die Monarchie abgeschafft und die demokratische Weimarer Republik konstituiert. Ab 1933 führte die nationalsozialistische Diktatur zu politischer und rassistischer Verfolgung und gipfelte in der Ermordung von sechs Millionen Juden und Angehörigen anderer Minderheiten wie Sinti und Roma. Der vom NS-Staat 1939 begonnene Zweite Weltkrieg endete 1945 mit der Niederlage der Achsenmächte. Das von den Siegermächten besetzte Land wurde 1949 geteilt, nachdem bereits 1945 seine Ostgebiete teils unter polnische, teils sowjetische Verwaltungshoheit gestellt worden waren. Der Gründung der Bundesrepublik als demokratischer westdeutscher Teilstaat mit Westbindung am 23. Mai 1949 folgte die Gründung der sozialistischen DDR am 7. Oktober 1949 als ostdeutscher Teilstaat unter sowjetischer Hegemonie. Die innerdeutsche Grenze war nach dem Berliner Mauerbau (ab 13. August 1961) abgeriegelt. Nach der friedlichen Revolution in der DDR 1989 erfolgte die Lösung der deutschen Frage durch die Wiedervereinigung beider Landesteile am 3. Oktober 1990, womit auch die Außengrenzen Deutschlands als endgültig anerkannt wurden. Durch den Beitritt der fünf ostdeutschen Länder sowie die Wiedervereinigung von Ost- und West-Berlin zur heutigen Bundeshauptstadt zählt die Bundesrepublik Deutschland seit 1990 sechzehn Bundesländer. Seit der Wiedervereinigung 1990 hat sich Deutschland zu einer der führenden Wirtschaftsnationen weltweit entwickelt. Anfangs stellte die Integration der DDR eine große Herausforderung dar, doch durch umfangreiche Investitionen und Reformen konnte die Wirtschaft stabilisiert werden. Insbesondere die Arbeitsmarktreformen der Agenda 2010 führten zu einer deutlichen Reduzierung der Arbeitslosigkeit und erhöhten die Wettbewerbsfähigkeit des Landes. Heute ist Deutschland die größte Volkswirtschaft der EU und eine der bedeutendsten Exportnationen weltweit. Das Land verfügt über eine gut entwickelte Infrastruktur, ein starkes Bildungssystem und eine hoch qualifizierte Arbeitskraft, was es zu einem attraktiven Standort für Unternehmen und Investitionen macht. Deutschland gilt heutzutage als eine der stabilsten und wohlhabendsten Nationen der Welt.")
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

def instructions():
    text_instructions: str = "Repräsentiere das Dokument für eine Suche."
    query_instructions: str = "Finde relevante Dokumente, die die folgende Frage beantworten."


def ollama():
    modelname = "GritLM"
    ollama_embedding = OllamaEmbedding(
        model_name=modelname
    )
    Settings.embed_model = ollama_embedding
    text = "Moin Moin, was geht? Alles klar bei dir? Wie spät?"
    embedding = Settings.embed_model.get_text_embedding(text)
    print(len(embedding))
    print(embedding[:6])


if __name__ == "__main__":
    old()
