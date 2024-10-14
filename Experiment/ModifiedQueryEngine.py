from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core import QueryBundle
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
import time, sys
from io import StringIO


class ModifiedQueryEngine:

    def __init__(self, retriever: BaseRetriever, response_synthesizer: BaseSynthesizer,
                 reranker=None, hyde: HyDEQueryTransform = None, reroute_stdout: bool = False):

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
        self.hyde_object: dict = {}
        self.verbose_output: dict = {}
        self.hyde = hyde
        self.use_hyde: bool = False if hyde is None else True

        # in case the
        self.reroute = reroute_stdout

    def query(self, query_str: str):
        start_time = time.time()
        query_bundle = QueryBundle(query_str)

        if self.use_hyde:
            query_transformed = self.hyde(query_bundle)
            self.hyde_object = {"Question": query_str, "Generated Document": query_transformed.embedding_strs[0]}
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

        query_stop_time = time.time()
        self.query_time = query_stop_time - start_time

        response_obj = self.response_synthesizer.synthesize(query_str, nodes)

        generating_stop_time = time.time()
        self.generating_time = generating_stop_time - query_stop_time
        self.total_time = generating_stop_time - start_time

        return response_obj

    def reranker_test(self, query_str: str, reranker: list):
        """
        hacked together to test different reranker on the same set of nodes
        :param query_str:
        :param reranker:
        :return:
        """
        query_bundle = QueryBundle(query_str)
        start_time = time.time()
        if not isinstance(query_str, str):
            raise ValueError("query_str must be a string")
        nodes = self.retriever.retrieve(str_or_query_bundle=query_str)
        query_time = time.time() - start_time
        response_objects: list = []
        base_response = self.response_synthesizer.synthesize(query_str, nodes)
        response_objects.append(base_response)
        times = [query_time]
        for rerank in reranker:
            rerank_start_time = time.time()
            rerank_nodes = rerank.postprocess_nodes(nodes, query_bundle)
            rerank_time = (time.time() - rerank_start_time) + query_time
            response = self.response_synthesizer.synthesize(query_str, rerank_nodes)
            response_objects.append(response)
            times.append(rerank_time)

        return response_objects, times

    def get_time(self) -> dict[str, float]:
        return {"query_time": self.query_time,
                "generating_time": self.generating_time,
                "total_time": self.total_time}
