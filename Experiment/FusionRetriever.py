import os
import re
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
#from openai import OpenAI
from llama_index.core.retrievers import BaseRetriever


class FusionRetriever(BaseRetriever):
    """
    Implements RAG Fusion approach.
    Takes a retriever to use for the retrieval.
    The query is passed to an LLM tasked to create similar questions.
    All questions are used to retrieve nodes from the retriever.
    This should be used in combination with a reranker for the query engine.
    """
    def __init__(self, retriever, llm: str = "gpt-4o-mini"):
        self.retriever = retriever
        super().__init__()
        self.generated_questions: dict = {}
        self.mode: str = "OR"
        self.llm = llm

    def _remove_leading_numbers(self, s) -> str:
        return re.sub(r'^\d+\.\s*', '', s)

    def _generate_questions(self, question) -> list:
        # load_dotenv()
        # api_key = os.getenv("OPENAI_API_KEY")

        # client = OpenAI(api_key=api_key)
        # Settings.llm = OpenAI(model=self.llm, api_key=api_key)

        template = "Du bekommst eine Frage gestellt. Formuliere die Frage auf 5 unterschiedliche Weisen um, ohne dabei den Inhalt der Frage zu ändern. Beanwtorte nicht die Frage sondern gebe nur die neuen Fragen zuück. Die Frage lautet: \n {question}"

        user_input = template.format(question=question)

        chat_response = Settings.llm.complete(user_input)
        generated_questions = chat_response.text.split("\n")
        generated_questions = [self._remove_leading_numbers(s) for s in generated_questions]
        return generated_questions

    def _retrieve(self, query) -> list:
        """
        generate a list of questions from the given query and retrieve nodes for each of them
        :return: returns a list of nodes
        """

        # generate a list of questions
        questions = self._generate_questions(query)
        self.log_questions(questions)
        questions.append(query)

        # collect all nodes from all retrievers
        # each retriever returns a list of nodes, so this return a list of lists
        nodes = []
        for question in questions:
            nodes.append(self.retriever.retrieve(question))

        # remove duplicates
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
        retrieve_nodes: list = [combined_dict[node_id] for node_id in retrieve_ids]

        return retrieve_nodes

    def log_questions(self, questions: list) -> None:
        """
        Converts the list into a dictionary that is convenient for logging it to csv.
        :param questions: list of questions
        :return: None
        """
        question_log: dict = {}
        n: int = 1
        for question in questions:
            key = f"Question {n}"
            question_log[key] = question
            n += 1

        self.generated_questions = question_log
