from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from typing import List

class CustomRetriever(BaseRetriever):
    """
    Custom retriever that takes a list of retrievers and combines the results.
    This is a simpler version of the Hybrid Search example in the documentation.
    Simpler because I don't do type checks but mine does allow for multiple different retrievers,
    which in itself made it a bit more complex because of the data type juggeling.
    """

    def __init__(self, retrievers, mode = "AND"):
        self.retrievers = retrievers
        self.mode = mode
        super().__init__()

    def _retrieve(self, query):
        """
        retrieve nodes given query
        :param query: string
        :return: list of nodes
        """

        # collect all nodes from all retrievers
        # each retriever returns a list of nodes, so this return a list of lists
        nodes = []
        for retriever in self.retrievers:
            nodes.append(retriever.retrieve(query))

        # collect the node ids for each list of nodes
        node_ids = []
        for node in nodes:
            ids = {n.node.node_id for n in node}
            node_ids.append(ids)

        # in order to remove duplicates we need to compare the ids to each other and create intersections or unions
        retrieve_ids = node_ids.pop(0)
        for ids in node_ids:
            retrieve_ids = retrieve_ids.intersection(ids) if self.mode == "AND" else retrieve_ids.union(ids)


        # we need to create a combined dictionary with all nodes in it
        start_node = nodes.pop(0)
        combined_dict = {n.node.node_id: n for n in start_node}
        for node in nodes:
            combined_dict.update({n.node.node_id: n for n in node})

        # now we take all items from the combined dictionary that are in the retrieve_ids and put them in a list
        retrieve_nodes = [combined_dict[node_id] for node_id in retrieve_ids]

        return retrieve_nodes





