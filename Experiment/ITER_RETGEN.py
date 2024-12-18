import time
from typing import Tuple
from llama_index.core import QueryBundle

class ITER_RETGEN:
    """
    Implements the ITER-RETGEN procedure from this paper:
    https://arxiv.org/abs/2305.15294

    generate an answer and then retrieve new documents with a combination of the previous answer and the query
    iterative process that could go on for multiple steps
    """
    def __init__(self, retriever, generator, reranker = None, max_iterations: int = 2):
        """

        :param retriever:
        :param max_iterations: 2 means 1 additional retrieval steps, 1 would be standard zero-shot retrieval
        """
        self.retriever = retriever
        self.generator = generator
        self.max_iterations = max_iterations
        self.reranker = reranker
        self.use_reranker = False if reranker is None else True

        # for data logging
        self.generating_time: float = 0
        self.query_time: float = 0
        self.total_time: float = 0

        self.additional_log: dict = {}

    def query(self, query: str):
        modified_query = query
        self.additional_log.update({"query": query})

        for i in range(self.max_iterations):
            # retrieval
            documents, retrieval_time = self.__measure_time(self.__retrieve, modified_query)
            self.query_time += retrieval_time

            # generation
            answer, generation_time = self.__measure_time(self.generator.synthesize, query=query, nodes=documents)
            self.generating_time += generation_time

            # updating query for generation augmented retrieval
            modified_query = query + " " + str(answer)

            # logging the current iteration for later
            self.__log_iteration(answer, i)

        return answer

    def __retrieve(self, query: str) -> list[any]:
        """
        simple wrapper function to include a reranker
        :param query:
        :return:
        """
        query_bundle = QueryBundle(query)
        nodes = self.retriever.retrieve(query)
        if self.use_reranker:
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        return nodes

    def __measure_time(self, func, *args, **kwargs) -> Tuple[any, float]:
        """
        Wrapper function to measure the time a function takes
        :param func: the function to execute
        :param args: the arguments for the function
        :param kwargs: the keyword arguments
        :return: the function return and the time it took
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    def __log_iteration(self, answer, iteration: int) -> None:
        """
        log each iteration, so I can later see if an additional iteration really improved anything
        :param answer: response object of that iteration
        :param iteration: count which iteration it is
        :return: None, it updates the global dictionary that stores the log
        """
        nodes = self.__extract_source_nodes(answer, iteration)
        answer = {f"Iter {iteration} Response": str(answer)}
        nodes.update(answer)
        self.additional_log.update(nodes)

    def __extract_source_nodes(self, response, iteration: int) -> dict[str, str]:
        """
        :param response: LlamaIndex Response Object
        :return: the nodes as dict for data logging
        """
        n: int = 0
        source_nodes = {}
        all_nodes = response.source_nodes
        iter: str = f"Iter {iteration}"
        for node in all_nodes:
            n += 1
            number = f"{iter} Node {n}"
            # extract the ID
            id_key = number + " ID"
            id_value = node.node_id
            # the content
            content_key = number + " content"
            content_value = node.get_text()
            # the score
            score_key = number + " score"
            score_value = node.get_score()
            # and the name of the service the node came from
            metadata_key = number + " Metadata: Name"
            metadata_content = node.metadata["Name"]

            node_dict = {id_key: id_value,
                         content_key: content_value,
                         metadata_key: metadata_content,
                         score_key: score_value}
            source_nodes.update(node_dict)
        return source_nodes

    def get_time(self) -> dict[str, float]:
        return {"query_time": self.query_time,
                "generating_time": self.generating_time,
                "total_time": self.total_time}

    def get_log(self) -> dict[str, any]:
        return self.additional_log
