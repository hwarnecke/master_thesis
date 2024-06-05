class QueryTool:

    def __init__(self, query_engine):
        self.description = ("Query Engine: Nützlich um verschiedene Fragen zu den Dienstleistungen der Stadt Osnabrück zu beantworten."
                            " Der Input sollte die Form einer Frage sein, die du beantwortet haben möchtest."
                            " Fragen sollten möglichst kleinschrittig gestellt werden.")
        self.name = "Query Engine"
        self.qe = query_engine

    def __call__(self, expression: str) -> str:
        # try:
        #     return self.qe.query(expression)
        # except:
        #     return "Leider konnte der Input nicht verarbeitet werden."
        return str(self.qe.query(expression))


    def get_time(self) -> dict[str, float]:
        return self.qe.get_time()