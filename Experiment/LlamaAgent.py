from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from io import StringIO
import sys
import time


class LlamaAgent:

    def __init__(self, query_engine):
        self.llm = Settings.llm
        self.query_engine = query_engine
        self.tool = FunctionTool.from_defaults(fn=self.diensleistungen)
        self.agent = ReActAgent.from_tools(llm=self.llm, tools=[self.tool], verbose=True)
        self.query_times: list[dict] = []
        self.query_response_objects: list = []
        self.chat_time: float = 0
        self.verbose_output: dict = {}

    def diensleistungen(self, query: str) -> str:
        """Beantwortet Fragen rund um Dienstleistungen, Adressen und Personen der Stadt Osnabrück."""
        answer = self.query_engine.query(query)
        self.query_times.append(self.query_engine.get_time())
        self.query_response_objects.append(answer)
        return answer

    def get_agent_nodes(self):
        return self.query_engine.get_agent_nodes()

    def query(self, query: str) -> list[str, str]:
        # reset logging values
        self.reset()

        # reroute output to gather more information
        old_stdout = sys.stdout
        verbose_output = StringIO()
        sys.stdout = verbose_output
        start_time = time.time()

        try:
            answer = self.agent.chat(query)
        except ValueError:
            answer = "Exceeded max iterations"

        self.chat_time = time.time() - start_time
        # reset output
        sys.stdout = old_stdout
        # FIXME: we need the content of this, not the IO Object
        self.verbose_output = {"Thought Process": verbose_output}

        return answer

    def reset(self):
        self.query_times = []
        self.query_response_objects = []
        self.verbose_output = {}

    def get_time(self) -> dict[str, float]:
        # needs to collect the time from query engine and add its own time to generating time
        total_query_time: float = 0
        for times in self.query_times:
            total_query_time += times["query_time"]
        generating_time = self.chat_time - total_query_time
        times = {"query_time": total_query_time,
                 "generating_time": generating_time,
                 "total_time": self.chat_time}
        return times

    def get_nodes(self) -> list:
        return self.query_response_objects
