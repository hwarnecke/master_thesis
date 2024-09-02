import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, get_response_synthesizer, StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.agent import ReActAgent
from llama_index.core.agent import ReActAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import FunctionTool
from llama_index.core import PromptTemplate
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.embeddings.cohere import CohereEmbedding
from ModifiedQueryEngine import ModifiedQueryEngine


def main():
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    vector_store_name = "service_" + "embed-multilingual-v3.0"
    Settings.embed_model = CohereEmbedding(model_name="embed-multilingual-v3.0", api_key=api_key)

    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url="http://localhost:9200",
    )
    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)

    llm = Ollama(model="mistral_7b", request_timeout=900)
    Settings.llm = llm
    query_engine = index.as_query_engine()
    retriever = index.as_retriever()
    synth = get_response_synthesizer(llm=llm)
    modified_qe = ModifiedQueryEngine(retriever=retriever, response_synthesizer=synth)

    def dienstleistungen(query: str):
        """Beantwortet Fragen zu den Dienstleistungen der Stadt Osnabrück"""
        return modified_qe.query(query)
    dienstleistungen_tool = FunctionTool.from_defaults(fn=dienstleistungen)

    query = "Ich muss einen neuen Personalausweis, Reisepass und Ersatzführerschein beantragen. Wie teuer ist das jeweils?"
    description = "Beantwortet Fragen zu Dienstleistungen und Adressen rund um die Stadt Osnabrück."
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine, name="Dienstleistungen", description=description
    )
    react_agent = ReActAgent.from_tools(llm=llm, tools=[dienstleistungen_tool], verbose=True)

    # update prompts
    # with open("PromptTemplates/react_agent_prompt.txt", "r") as file:
    #     new_system_prompt = file.read()
    # new_prompt = PromptTemplate(new_system_prompt)
    # react_agent.update_prompts({"agent_worker:system_prompt": new_prompt})

    # prompt_dict = react_agent.get_prompts()
    # for k, v in prompt_dict.items():
    #     print(f"Prompt: {k}\n\nValue: {v.template}")
    answer = react_agent.chat(query)
    print(answer)
    nodes = modified_qe.get_agent_nodes()
    len_nodes = len(nodes)
    print(len_nodes)
    print(nodes)

    # query = "Wie lautet die Adresse der Stadtbibliothek?"
    # answer = react_agent.chat(query)
    # nodes2 = modified_qe.get_agent_nodes()
    # print(answer)
    # print(nodes2)



def low_level():
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    vector_store_name = "service_" + "embed-multilingual-v3.0"
    Settings.embed_model = CohereEmbedding(model_name="embed-multilingual-v3.0", api_key=api_key)

    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url="http://localhost:9200",
    )
    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)

    llm = Ollama(model="mistral_7b")
    retriever = index.as_retriever()
    synth = get_response_synthesizer(llm=llm)
    modified_qe = ModifiedQueryEngine(retriever=retriever, response_synthesizer=synth)

    def dienstleistungen(query: str):
        """Beantwortet Fragen zu den Dienstleistungen der Stadt Osnabrück"""
        return modified_qe.query(query)

    dienstleistungen_tool = FunctionTool.from_defaults(fn=dienstleistungen)

    query = "Wo muss ich hin um einen Büchereiausweis zu beantragen?"

    worker = ReActAgentWorker(llm=llm, tools=[dienstleistungen_tool])
    # update prompts
    with open("PromptTemplates/react_agent_prompt.txt", "r") as file:
        new_system_prompt = file.read()
    new_prompt = PromptTemplate(new_system_prompt)

    agent = AgentRunner(worker, verbose=True)
    agent.update_prompts({"system_prompt": new_prompt})
    task = agent.create_task(query)

    is_last = False
    while not is_last:
        step_output = agent.run_step(task.task_id)
        is_last = step_output.is_last
    response = agent.finalize_response(task.task_id)
    print(response)

def stupid_tool(number: int) -> int:
    """Useful for calculating square numbers"""
    return number * number

def update_prompts():
    llm = Ollama(model="gritlm")
    tool = FunctionTool.from_defaults(stupid_tool)
    worker = ReActAgentWorker(llm=llm, tools=[tool])
    prompts = worker.get_prompts()
    print(prompts)

    new_prompt = PromptTemplate("bums")
    worker.update_prompts({"system_prompt": new_prompt})

    prompts = worker.get_prompts()
    print(prompts)

if __name__ == "__main__":
    main()