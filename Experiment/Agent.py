import re
import time
from llama_index.core import Settings


class Agent:
    """
    The Agent class combines the LanguageModel, Prompt, and Tools classes to implement the ReAct pattern.
    More detail about the pattern can be found in this paper: https://arxiv.org/abs/2210.03629
    """

    def __init__(self, prompt_path: str = "PromptTemplates/agent_prompt.txt", tools: list = None):
        # currently I'll just use the Llamaindex integration of OpenAI as I do for the rest of the query engines
        self.llm = Settings.llm
        with open(prompt_path, "r") as file:
            self.prompt: str = file.read()

        # include the tools into the prompt template
        self.tools = tools if tools else []
        tool_description: str = ""
        tool_names: list[str] = []
        for tool in self.tools:
            description: str = tool.description
            tool_description += f"\n{description}"
            tool_names.append(tool.name)

        self.prompt = self.prompt.replace("{tools}", tool_description)
        self.prompt = self.prompt.replace("{tool_list}", str(tool_names))

        # for measuring the passed time for the whole query (summing the time from the individual calls)
        self.query_time: float = 0
        self.query_generating_time: float = 0
        self.generating_time: float = 0
        self.total_time: float = 0

        self.retriever_log: dict[str, any] = {}


    def __CreateLogItem(self, query: str, inputs: list[str], observations: list[str]) -> dict[str, str]:
        log: dict = {"query": query}
        for i in range(len(inputs)):
            input_key = f"Input {i+1}"
            observation_key = f"Observation {i+1}"
            log.update({input_key: inputs[i], observation_key: observations[i]})
        return log

    def __ExtractTime(self, tool):
        """
        adds the time it took for one tool call to the time score of the agent.
        :param tool:
        :return:
        """
        times: dict[str, float] = tool.get_time()
        if times:
            self.query_time += times["query_time"]
            self.query_generating_time += times["generating_time"]
            self.total_time += times["total_time"]

    def get_time(self) -> dict[str, float]:
        return {"query_time": self.query_time,
                "generating_time": self.generating_time + self.query_generating_time,
                "total_time": self.total_time}

    def __extract_source_nodes(self, response, retrieval_count: int = 1) -> dict[str, str]:
        """
        :param response: LlamaIndex Response Object
        :return: the nodes as dict for data logging
        """
        n: int = 0
        source_nodes = {}
        all_nodes = response.source_nodes
        for node in all_nodes:
            n += 1
            number = f"Call {retrieval_count} Node {n}"
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

    def __create_retriever_log(self, response, input: str, count: int = 1) -> dict:
        retrieval_log = self.__extract_source_nodes(response=response, retrieval_count=count)
        log = {f"Call {count} Input": input, f"Call {count} Observation": str(response)}
        log.update(retrieval_log)
        return log


    def query(self, question: str) -> dict:
        prompt = self.prompt.replace("{question}", question)
        self.retriever_log = {}
        max_iterations: int = 5
        action_inputs: list[str] = []
        observations: list[str] = []
        for i in range(max_iterations):
            # stop the generation when an after an action chosen and an action input is generated
            start_time: float = time.time()
            response = str(self.llm.complete(prompt, stop=["Beobachtung:"]))
            stop_time: float = time.time()
            self.generating_time += stop_time - start_time

            prompt += response

            # match the action if one was chosen
            action_regex = r"Aktion: (.+)"
            try:
                action = re.findall(action_regex, response)[0]
            except:
                action = None

            if action:
                # match the action to the tools and call with the action input
                input_regex = r"Aktionseingabe: (.+)"
                try:
                    action_input = re.findall(input_regex, response)[0]
                except:
                    action_input = ""

                # append the observation to the prompt
                observation = ""
                for tool in self.tools:
                    if tool.name == action:
                        response = tool(action_input)
                        observation += str(response)
                        self.__ExtractTime(tool)
                        retriever_log = self.__create_retriever_log(response=response, input=action_input, count=i+1)
                        self.retriever_log.update(retriever_log)


                action_inputs.append(action_input)
                observations.append(observation)

                prompt += f"\nObservation: {observation}\n"

            else:
                # if no action is found, return the response as a final answer is reached
                answer_regex = r"Endgültige Antwort: (.+)"
                try:
                    answer = re.findall(answer_regex, response)[0]
                except:
                    answer = "Es tut mir leid, ich konnte keine finale Antwort finden."

                thought_process = prompt.split("Beginne!\n\n\n")[1]     # remove the unnecessary prompt template
                result: dict = Response({"thought_process": thought_process,
                                         "response": answer,
                                         "query": question,
                                         "retriever_log": self.retriever_log,
                                         "observations": observations})
                return result

        # if no final answer is reached by now it means that he tried to do more than the max_iteration steps
        answer_template: str = "Vielleicht musst du doch selber nachdenken..."
        thought_process = prompt.split("Beginne!\n\n\n")[1]
        result: dict = Response({"thought_process": thought_process,
                                 "response": answer_template,
                                 "query": question,
                                 "retriever_log": self.retriever_log,
                                 "observations": observations})
        return result


class Response(dict):
    """
    For compatibility reasons with the Llamaindex response class.
    """
    def __str__(self):
        return str(self.get("response", None))
