from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import QueryBundle
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
import time, sys
from io import StringIO
class ModifiedQueryEngine(CustomQueryEngine):


    def __init__(self, retriever: BaseRetriever, response_synthesizer: BaseSynthesizer, reranker: SentenceTransformerRerank = None, hyde: HyDEQueryTransform = None):
        super().__init__()
        self.retriever = retriever
        self.response_synthesizer = response_synthesizer
        self.reranker = reranker
        # we do want to test the base case without reranking
        self.use_reranker: bool = False if reranker is None else True

        # for measuring the passed time on each query
        self.query_time: int = 0
        self.generating_time: int = 0
        self.total_time: int = 0

        # for the special cases that store extra information like HyDE, Auto or Fusion
        self.additional_data: str
        self.hyde = hyde
        self.use_hyde: bool = False if hyde is None else True


    def custom_query(self, query_str: str):
        start_time = time.time()
        query_bundle = QueryBundle(query_str)


        if self.use_hyde:
            query_transformed = self.hyde(query_bundle)
            self.additional_data = query_transformed    # TODO: check format in which this is stored
            nodes = self.retriever.retrieve(query_transformed)
        else:
            old_stdout = sys.stdout
            filter_info = StringIO()
            sys.stdout = filter_info

            nodes = self.retriever.retrieve(query_str)

            self.additional_data = filter_info  # TODO: check format in which this is stored
            sys.stdout = old_stdout


        if self.use_reranker:
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)

        query_stop_time = time.time()
        self.query_time = query_stop_time - start_time

        response_obj = self.response_synthesizer(query_str, nodes)

        generating_stop_time = time.time()
        self.generating_time = generating_stop_time - query_stop_time
        self.total_time = generating_stop_time - start_time

        return response_obj