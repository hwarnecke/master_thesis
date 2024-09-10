import os
from dotenv import load_dotenv
import datetime
import sys
import io
from llama_index.core import Settings, get_response_synthesizer, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
#from llama_index.core.retrievers import VectorIndexAutoRetriever
#from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank

from CombinedRetriever import CombinedRetriever
from FusionRetriever import FusionRetriever
from ModifiedQueryEngine import ModifiedQueryEngine
from LlamaAgent import LlamaAgent
from ITER_RETGEN import ITER_RETGEN


def generateID(name: str,
               llm: str,
               embedding: str,
               reranker: str,
               timestamp: str,
               prompt: str,
               retriever_top_k: int,
               rerank_top_n: int) -> str:
    """
    The ID doubles as the directory and name of the QE, that's why it has everything two times.
    Everything before the '/' will be used as the directory and everything after that will be used as file name.
    For the directory it seems better to start with the timestamps. This will put runs that were made in succession
    next to each other (i.e. different llms are most likely tested directly after each other).
    While for the name it is better to start with the name, as that is the most important piece of information.
    The timestamp is the least significant information, so it is placed last.
    :param name:
    :param llm:
    :param embedding:
    :param reranker
    :param timestamp:
    :param prompt:
    :return: str - the ID
    """
    reranker = reranker.split("/")[-1] # in case there is no '/' like cohere reranker
    return f"{timestamp}_{llm}_{embedding}_{reranker}_{prompt}_retriever{retriever_top_k}_rerank{rerank_top_n}/{name}_{llm}_{embedding}_{reranker}_{prompt}_{timestamp}"


def get_LLM(type: str, llm: str) -> None:
    match type:
        case 'OpenAI':
            openAI_api_key = os.getenv("OPENAI_API_KEY")
            Settings.llm = OpenAI(model=llm, api_key=openAI_api_key)
        case 'Cohere':
            cohere_api_key = os.getenv("COHERE_API_KEY")
            Settings.llm = Cohere(api_key=cohere_api_key, model=llm)
        case 'Ollama':
            Settings.llm = Ollama(model=llm, request_timeout=900)
        case _:
            raise ValueError(f"Unsupported llm type: {type}")

def get_Embedding(type: str, embedding_name: str):
    match type:
        case 'OpenAI':
            openAI_api_key = os.getenv("OPENAI_API_KEY")
            embedding_model = OpenAIEmbedding(model_name=embedding_name, api_key=openAI_api_key)
            embedding_id = embedding_name.split("-")[0]
        case 'Cohere':
            cohere_api_key = os.getenv("COHERE_API_KEY")
            embedding_model = CohereEmbedding(api_key=cohere_api_key, model_name=embedding_name)
            embedding_id = embedding_name.split("-")[0]
        case 'HuggingFace':
            text_instructions: str = "Repräsentiere das Dokument für eine Suche."
            query_instructions: str = "Finde relevante Dokumente, die die folgende Frage beantworten."
            # there have been OOM issues for some models, when running on cuda and I don't want to list
            # individual models, so I'll just run them all on cpu.
            # also, I wasn't sure what happens if I supply instructions to non instruct models, so I split it here
            if "instruct" in embedding_name:
                embedding_model = HuggingFaceEmbedding(model_name=embedding_name,
                                                       device='cpu',
                                                       query_instruction=query_instructions,
                                                       text_instruction=text_instructions,
                                                       trust_remote_code=True)
            else:
                embedding_model = HuggingFaceEmbedding(model_name=embedding_name, device='cpu', trust_remote_code=True)
            # HuggingFace models always come with the author like 'author/model-name', so I need to remove the author.
            # additionally, they can use both '-' and '_' to seperate words in the model_name, so if I want to split it,
            # I need to filter for both
            embedding_name = embedding_name.split("/")[1]
            embedding_id = embedding_name.split("-")[0].split("_")[0]
        case 'Ollama':
            # you have to set up the model before you can run this
            embedding_model = OllamaEmbedding(
                model_name=embedding_name
            )
            embedding_id = embedding_name
        case _:
            raise ValueError(f"Unsupported embedding type: {type}")

    return embedding_model, embedding_id

def get_Reranker(type: str, rerank_model: str, rerank_top_n: int):
    match type:
        case 'sentenceTransformer':
            # some models require trust_remote_code but STR does not have a parameter for this
            # instead it askes per user input for permission but if I miss it, it fails.
            # this should automatically give permission
            with as_stdin(io.StringIO('y\ny')):
                reranker = SentenceTransformerRerank(top_n=rerank_top_n, model=rerank_model, device="cpu")
        case 'cohere':
            cohere_api_key = os.getenv("COHERE_API_KEY")
            reranker = CohereRerank(api_key=cohere_api_key, top_n=rerank_top_n, model=rerank_model)
        case 'colbert':
            reranker = ColbertRerank(top_n=rerank_top_n, model=rerank_model, keep_retrieval_score=True)
        case 'jina':
            jina_api_key = os.getenv("JINAAI_API_KEY")
            reranker = JinaRerank(api_key=jina_api_key, top_n=rerank_top_n, model=rerank_model)
        case _:
            raise ValueError(f"Unsupported rerank type: {type}")

    return reranker

def create_query_engines(llm: str = "gpt-4o-mini",
                         llm_type: str = "OpenAI",
                         embedding_name: str ="text-embedding-3-small",
                         embedding_type: str = "OpenAI",
                         embedding_url: str = "http://localhost:9200",
                         rerank_top_n: int = 3,
                         rerank_model: str = "cross-encoder/stsb-distilroberta-base",
                         rerank_type: str = "SentenceTransformer",
                         retriever_top_k: int = 6,
                         custom_qa_prompt: str = None,
                         custom_refine_prompt: str = None,
                         use_query_engines: list[str] = None,
                         response_mode: str = "refine") -> dict:

    """
    This function creates the query engines for the experiment.
    Since most query engines need the same base elements, this function is used to avoid code duplication.
    :param llm: LLM to be used. Default is OpenAI with "gpt-4o-mini".
    :param llm_type: Where the LLM comes from, currently supports OpenAI, Cohere and Ollama.
    :param embedding_name: a name of a HuggingFace embedding. Default is the text-embedding-3-small from OpenAI.
    :param embedding_type: Where the embedding comes from, valid options are OpenAI, Cohere, Huggingface and Ollama
    :param embedding_url: the url of the docker container with the index
    :param rerank_top_n: the number of top nodes to be returned by the reranker. Default is 3.
    :param rerank_model: needs to fit to the rerank types, default it "cross-encoder/stsb-distilroberta-base"
    :param rerank_type: which Llamaindex module to use for reranking, supports SentenceTransformer and Cohere
    :param retriever_top_k: the number of top nodes to be returned by the retriever. Default is 6.
    :param custom_qa_prompt: the prompt to use for the qa part of the refine response mode.
    :param custom_refine_prompt: the prompt to use for the refine part of the refine response mode.
    :return: a dictionary of query engines with the retriever ID as key
    """

    """
    the first part contains settings that are the same for all query engines
    including the LLM, the vectorstore store and the reranker to use.
    """
    if use_query_engines == None:
        use_query_engines = ["base", "rerank", "hybrid", "auto", "hyde", "fusion", "agent", "iter-retgen"]

    load_dotenv()

    print(f"LLM: {llm_type} {llm}.")
    get_LLM(llm_type, llm)

    print(f"Embedding: {embedding_type} {embedding_name}")
    embedding_model, embedding_id = get_Embedding(embedding_type, embedding_name)

    Settings.embed_model = embedding_model
    vector_store_name = "service_" + embedding_name.lower()

    # if I want to test different embeddings, I can call a different vector store here
    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url=embedding_url,
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    reranker = None
    print(f"Reranker: {rerank_type} {rerank_model}")
    reranker = get_Reranker(rerank_type, rerank_model, rerank_top_n)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)

    basic_response_synthesizer = get_response_synthesizer(response_mode=response_mode)
    iterative_response_synthesizer = get_response_synthesizer(response_mode="refine")

    prompt_id = "retrieval_only"
    if custom_qa_prompt and custom_refine_prompt:
        custom_qa_template = PromptTemplate(custom_qa_prompt)
        custom_refine_template = PromptTemplate(custom_refine_prompt)
        # the synth for the two iterative approaches always need to create an answer otherwise they won't work.
        iterative_response_synthesizer.update_prompts(
                {
                    "qa_template": custom_qa_template,
                    "refine_template": custom_refine_template,
                }
        )

        if response_mode == "refine":
            prompt_id = "german_prompt"
            basic_response_synthesizer.update_prompts(
                {
                    "qa_template": custom_qa_template,
                    "refine_template": custom_refine_template,
                }
            )

    basic_retriever = index.as_retriever(similarity_top_k=retriever_top_k)

    """
    Each model will get a unique ID, so I can later identify which model was used in what configuration.
    this includes the following:
    1. name of the method used
        - base
        - rerank
        - hybrid
        - auto
        - hyde
        - fusion
        - agent
        - iter-retgen
        
    2. the LLM used
        - gpt3 for gpt-3.5-turbo
        - gpt4 for gpt-4-turbo
        - if local models are later added, I'll add a description here
        
    3. the embedding used
        will use the name that is also used on HuggingFace and shorten it to one keyword
        the keyword will be whatever is after the initial '/' and the first '-'
        i.e. 'OpenAI/text-embedding-ada-002' will turn into 'text'
        
        
    4. a timestamp to make sure that the ID is unique over multiple runs
    """

    # the llm used
    llm_id = llm
    if llm == "gpt-40-mini":
        llm_id = "gpt4mini"
    elif llm == "gpt-4-turbo":
        llm_id = "gpt4"

    # the timestamp needs to be formatted in order to allow it as a file name
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime('%Y-%m-%d_%H-%M-%S')

    query_engines = {}

    """
    the second part contains the specific query engines
    """

    """
    1. the base query engine without any modifications
    """
    name_id = "base"
    if name_id in use_query_engines:
        # here I don't use the basic_retriever I created above, because this one should retrieve fewer documents
        base_retriever = index.as_retriever(similarity_top_k=rerank_top_n)
        query_base = ModifiedQueryEngine(retriever=base_retriever, response_synthesizer=basic_response_synthesizer)

        retriever_id = generateID(name_id, llm_id, embedding_id, rerank_model, timestamp, prompt_id, retriever_top_k, rerank_top_n)
        query_engines[retriever_id] = query_base

    """
    2. the base query engine with the reranker
    """
    name_id = "rerank"
    if name_id in use_query_engines:
        query_rerank = ModifiedQueryEngine(retriever=basic_retriever,
                                           response_synthesizer=basic_response_synthesizer,
                                           reranker=reranker)

        retriever_id = generateID(name_id, llm_id, embedding_id, rerank_model, timestamp, prompt_id, retriever_top_k, rerank_top_n)
        query_engines[retriever_id] = query_rerank

    """
    3. the hybrid query engine with the basic retriever and the bm25 retriever
    """
    name_id = "hybrid"
    if name_id in use_query_engines:
        base = index.as_retriever(similarity_top_k=retriever_top_k)

        # apparently, using an external vector store does block some functionality in order to simplify storage
        # this means I cannot directly access the nodes in the docstore which are needed for the BM25
        # one hacky solution is to just retrieve all nodes with a standard retriever and a high top_k
        # with context windows of 512 there should be 1203 TextNodes total in the KB.
        hacky_retriever = index.as_retriever(similarity_top_k=1000)
        source_nodes = hacky_retriever.retrieve("und")
        hacky_nodes = [x.node for x in source_nodes]

        bm25 = BM25Retriever.from_defaults(nodes=hacky_nodes, similarity_top_k=retriever_top_k)
        hybrid_retriever = CombinedRetriever([base, bm25], mode="OR")
        query_hybrid = ModifiedQueryEngine(
            retriever=hybrid_retriever,
            response_synthesizer=basic_response_synthesizer,
            reranker=reranker,
        )
        retriever_id = generateID(name_id, llm_id, embedding_id, rerank_model, timestamp, prompt_id, retriever_top_k, rerank_top_n)
        query_engines[retriever_id] = query_hybrid

    """
    5. HyDE (Hybrid Document Embedding)
    
    If I want to use the standard query engine format, I can't capture the hypothetical answer created.
    I would need to rerun the hyde object with the query in order to store it, but I can't be sure that the answer
    will be the same all the time.
    
    An alternative would be to write a custom query engine that returns the hypothetical answer as well, but that
    would be a bit more complicated. So for now I will go the easy route.
    
    Generating and Accessing the answer can be done like this:
        query_bundle = hyde(query_str)
        hyde_doc = query_bundle.embedding_strs[0]
    """
    name_id = "hyde"
    if name_id in use_query_engines:
        retriever_id = generateID(name_id, llm_id, embedding_id, rerank_model, timestamp, prompt_id, retriever_top_k, rerank_top_n)
        hyde = HyDEQueryTransform(include_original=True)
        query_hyde = ModifiedQueryEngine(
            retriever=basic_retriever,
            response_synthesizer=basic_response_synthesizer,
            reranker=reranker,
            hyde=hyde
        )
        query_engines[retriever_id] = query_hyde

    """
    6. RAG Fusion
    
    Similar to the AutoRetriever, the stdout needs to be redirected in order to catch the verbose output.
    """
    name_id = "fusion"
    if name_id in use_query_engines:
        #base_retriever = index.as_retriever(similarity_top_k=5)
        fusion_retriever = FusionRetriever(retriever=basic_retriever)
        query_fusion = ModifiedQueryEngine(
            retriever=fusion_retriever,
            response_synthesizer=basic_response_synthesizer,
            reranker=reranker,
        )
        retriever_id = generateID(name_id, llm_id, embedding_id, rerank_model, timestamp, prompt_id, retriever_top_k, rerank_top_n)
        query_engines[retriever_id] = query_fusion

    """
    7. Agent
    """
    name_id = "agent"
    if name_id in use_query_engines:


        query_engine = ModifiedQueryEngine(retriever=basic_retriever,
                                           response_synthesizer=iterative_response_synthesizer,
                                           reranker=reranker)
        agent = LlamaAgent(query_engine=query_engine)
        with open("PromptTemplates/react_agent_prompt.txt", "r") as file:
            new_system_prompt = file.read()
        new_prompt = PromptTemplate(new_system_prompt)
        agent.agent.update_prompts({"agent_worker:system_prompt": new_prompt})
        retriever_id = generateID(name_id, llm_id, embedding_id, rerank_model, timestamp, prompt_id, retriever_top_k, rerank_top_n)
        query_engines[retriever_id] = agent

    """
    8. ITER-RETGEN
    """
    name_id = "iter-retgen"
    if name_id in use_query_engines:
        iter_retgen = ITER_RETGEN(retriever=basic_retriever, generator=iterative_response_synthesizer, reranker=reranker)
        retriever_id = generateID(name_id, llm_id, embedding_id, rerank_model, timestamp, prompt_id, retriever_top_k, rerank_top_n)
        query_engines[retriever_id] = iter_retgen

    return query_engines


class as_stdin:
    def __init__(self, buffer):
        self.buffer = buffer
        self.original_stdin = sys.stdin
    def __enter__(self):
        sys.stdin = self.buffer
    def __exit__(self, *exc):
        sys.stdin = self.original_stdin
