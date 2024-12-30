from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser


class DeepEvalCustomLLM(DeepEvalBaseLLM):
    def __init__(self, llm):
        self.llm = llm

    def load_model(self):
        return self.llm

    def generate(self, prompt: str) -> str:
        return str(self.llm.complete(prompt))

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "DeepEvalCustomLLM"
