class Response(dict):
    """
    For compatibility reasons with the Llamaindex response class.
    """
    def __str__(self):
        return str(self.get("response", None))


def main():
    test = {"response": "Hallo"}
    response = Response(test)

    t = type(response)
    print(t)
    print(isinstance(response, dict))
    print(isinstance(response,Response))


def stringtest():
    string = """Beantworte die folgenden Fragen so gut wie möglich. Dir stehen die folgenden Werkzeuge zur Verfügung:


{tools}


Verwende immer das folgende Format:


Frage: die Eingangsfrage, die du beantworten musst
Überlegung: Du solltest immer darüber nachdenken, was zu tun ist
Aktion: die zu ergreifende Handlung, sollte eines der folgenden sein {tool_list}
Aktionseingabe: die Eingabe für die Aktion
Beobachtung: das Ergebnis der Aktion
... (diese Überlegung/Aktion/Aktionseingabe/Beobachtung kann sich 10-mal wiederholen)
Überlegung: Ich kenne jetzt die endgültige Antwort
Endgültige Antwort: die endgültige Antwort auf die ursprüngliche Eingangsfrage


Beginne!


Frage: {question}
Überlegung:"""

    last = string.split("Beginne!\n\n\n")[1]
    print(last)

def loops():
    embedding_models = {"HuggingFace": "aari1995/German_Semantic_V3b",
                        "HuggingFace": "T-Systems-onsite/cross-en-de-roberta-sentence-transformer",
                        "HuggingFace": "jinaai/jina-embeddings-v2-base-de",
                        "HuggingFace": "jinaai/jina-clip-v1",
                        "HuggingFace": "intfloat/multilingual-e5-large-instruct",
                        "HuggingFace": "Alibaba-NLP/gte-multilingual-base",
                        "HuggingFace": "dunzhang/stella_en_1.5B_v5",
                        "HuggingFace": "GritLM/GritLM-7B",
                        "Cohere": "embed-multilingual-v3.0",
                        "OpenAI": "text-embedding-3-small"}

    for key, value in embedding_models.items():
        print(value)

if __name__ == "__main__":
    loops()