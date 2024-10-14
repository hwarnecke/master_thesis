from llama_index.core.retrievers import BaseRetriever

class CombinedRetriever(BaseRetriever):
    """
    renamed the CustomRetriever to CombinedRetriever to better reflect what it does
    It takes a list of retrievers and returns the combines results from each of them.
    It can combine the results in two ways:
    - AND: only return nodes that are in all lists
    - OR: return nodes that are in at least one list

    Most of this comes from the Hybrid Search example in the LlamaIndex documentation:
    https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/?h=hybrid+search

    I left out the type checks but added the possibility to have multiple different retrievers.
    """

    def __init__(self, retrievers, mode = "OR"):
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





