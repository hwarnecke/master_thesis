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

if __name__ == "__main__":
    main()