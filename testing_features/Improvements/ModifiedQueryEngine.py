from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import QueryBundle
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
import time, sys
from io import StringIO


class ModifiedQueryEngine():

    def __init__(self, retriever: BaseRetriever, response_synthesizer: BaseSynthesizer,
                 reranker: SentenceTransformerRerank = None, hyde: HyDEQueryTransform = None, reroute_stdout: bool = False):

        self.retriever = retriever
        self.response_synthesizer = response_synthesizer
        self.reranker = reranker
        # we do want to test the base case without reranking
        self.use_reranker: bool = False if reranker is None else True

        # for measuring the passed time on each query
        self.query_time: float = 0
        self.generating_time: float = 0
        self.total_time: float = 0

        # for the special cases that store extra information like HyDE, Auto or Fusion
        self.hyde_object: dict
        self.verbose_output: dict
        self.hyde = hyde
        self.use_hyde: bool = False if hyde is None else True
        self.agent_nodes = []

        # in case the
        self.reroute = reroute_stdout

    def query(self, query_str: str):
        start_time = time.time()
        query_bundle = QueryBundle(query_str)

        if self.use_hyde:
            query_transformed = self.hyde(query_bundle)
            self.hyde_object = {"Question": query_str, "Generated Document": query_transformed.embedding_strs[0]}  # TODO: check format in which this is stored
            nodes = self.retriever.retrieve(query_transformed)
        else:
            # I included a flag to only reroute the output for the autoretriever because it otherwise
            # interfered with my bug testing where I wanted the retrievers to print something
            if self.reroute:
                old_stdout = sys.stdout
                filter_info = StringIO()
                sys.stdout = filter_info

            nodes = self.retriever.retrieve(query_str)

            # the auto retriever will have a certain structure to the verbose output
            # the structure includes one line for the query and one line for a list of filters
            if self.reroute:
                value = filter_info.getvalue()
                lines = value.split("\n")
                if len(lines) > 1:
                    query_str = lines[0].split(": ")[1]
                    filters = lines[1].split(": ")[1]
                    self.verbose_output = {"Auto Query": query_str, "Filter": filters}
                else:
                    self.verbose_output = {"Other": value}

                sys.stdout = old_stdout

        if self.use_reranker:
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)

        # for the agent
        self.agent_nodes.append(nodes)

        query_stop_time = time.time()
        self.query_time = query_stop_time - start_time

        response_obj = self.response_synthesizer.synthesize(query_str, nodes)

        generating_stop_time = time.time()
        self.generating_time = generating_stop_time - query_stop_time
        self.total_time = generating_stop_time - start_time

        return response_obj


    def get_time(self) -> dict[str, float]:
        return {"query_time": self.query_time,
                "generating_time": self.generating_time,
                "total_time": self.total_time}

    def get_agent_nodes(self):
        nodes = self.agent_nodes.copy()
        self.agent_nodes = []
        return nodes