import re
import os
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

    def __call__(self, question: str) -> dict:
        prompt = self.prompt.replace("{question}", question)
        max_iterations: int = 10
        for i in range(max_iterations):
            # stop the generation when an after an action chosen and an action input is generated
            response = str(self.llm.complete(prompt, stop=["Beobachtung:"]))
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
                        observation += tool(action_input)

                prompt += f"\nObservation: {observation}\n"

            else:
                # if no action is found, return the response as a final answer is reached
                answer_regex = r"Endgültige Antwort: (.+)"
                try:
                    answer = re.findall(answer_regex, response)[0]
                except:
                    answer = "Es tut mir leid, ich konnte keine finale Antwort finden."

                result: dict = {"thought_process": prompt, "answer": answer}
                return result

        # if no final answer is reached by now it means that he tried to do more than the max_iteration steps
        answer_template: str = "Vielleicht musst du doch selber nachdenken..."
        result: dict = {"thought_process": prompt, "answer": answer_template}
        return result


