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
    for i in range(1):
        print(i)

if __name__ == "__main__":
    loops()