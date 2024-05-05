import os
from dotenv import load_dotenv
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings, get_response_synthesizer, VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

from CombinedRetriever import CombinedRetriever
from FusionRetriever import FusionRetriever


def create_query_engines(llm = "gpt-3.5-turbo", vector_store_name = "city_service_store", rerank_top_n = 3, retriever_top_k = 6):

    """
    This function creates the query engines for the experiment.
    Since most query engines need the same base elements, this function is used to avoid code duplication.
    :param llm: the OpenAI model to be used. Default is "gpt-3.5-turbo", alternative is gpt-4-turbo
    :param vector_store_name: the name of the vector store to be used. Default is "city_service_store"
    :param rerank_top_n: the number of top nodes to be returned by the reranker. Default is 3.
    :param retriever_top_k: the number of top nodes to be returned by the retriever. Default is 6.
    :return: a dictionary of query engines with the retriever ID as key
    """

    """
    the first part contains settings that are the same for all query engines
    including the LLM, the vectore store and the reranker to use.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    Settings.llm = OpenAI(model=llm, api_key=api_key) # if I want to use a local model, I need to make this more general

    # if I want to test different embeddings, I can call a different vector store here
    es_vector_store = ElasticsearchStore(
        index_name=vector_store_name,
        es_url="http://localhost:9200",
    )

    storage_context = StorageContext.from_defaults(vector_store=es_vector_store)

    reranker = reranker = SentenceTransformerRerank(top_n=rerank_top_n)

    index = VectorStoreIndex.from_vector_store(
        vector_store=es_vector_store,
        storage_context=storage_context,
        show_progress=False)

    response_synthesizer = get_response_synthesizer()

    query_engines = {}

    """
    the second part contains the specific query engines
    """

    """
    1. the base query engine without any modifications
    """

    query_base = index.as_query_engine(similarity_top_k=rerank_top_n)

    """
    2. the base query engine with the reranker
    """

    query_rerank = index.as_query_engine(similarity_top_k=retriever_top_k, node_postprocessors=[reranker])

    """
    3. the hybrid query engine with the basic retriever and the bm25 retriever
    """

    base = index.as_retriever(similarity_top_k=retriever_top_k)
    bm25 = BM25Retriever.from_defaults(index=index, similarity_top_k=retriever_top_k)
    hybrid_retriever = CombinedRetriever([base, bm25], mode="OR")
    query_hybrid = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker],
    )

    """
    4. AutoRetriever
    
    Don't forget to redirect stdout in order to catch the verbose output of the auto retriever.
    Otherwise information about which filter was used gets lost.
    Also don't forget to reset stdout afterwards.
    This can be done like this:
        filter_info = StringIO()
        old_stdout = sys.stdout
        sys.stdout = filter_info
        ...query...
        sys.stdout = old_stdout
    """
    # EDIT: I might want to outsource this into a separate file
    # create a vector store info that contains an overview over the metadata
    vector_store_info = VectorStoreInfo(
        content_info="Informationen über die Dienstleistungen der Stadt Osnabrück.",
        metadata_info=[
            MetadataInfo(
                name="Typ",
                description="Der Typ der Information, die in dem Dokument beschrieben ist.",
                type="String",
            ),
            MetadataInfo(
                name="Name",
                description="Der Name der Dienstleistung.",
                type="String",
            ),
            MetadataInfo(
                name="URL",
                description="Die URL, unter der die Dienstleistung zu finden ist.",
                type="String",
            ),
            MetadataInfo(
                name="Kategorie",
                description="Die Kategorie, zu der die Dienstleistung gehört.",
                type="String or list[String]",
            ),
            MetadataInfo(
                name="Anfangsbuchstabe",
                description="Der Anfangsbuchstabe des Namens der Dienstleistung.",
                type="String",
            ),
            MetadataInfo(
                name="Synonyme",
                description="Synonyme für den Namen der Dienstleistung.",
                type="list[String]",
            ),
            MetadataInfo(
                name="Fachbereich",
                description="Der Fachbereich, zu dem die Dienstleistung gehört.",
                type="String",
            ),
            MetadataInfo(
                name="Kontakt",
                description="Der Kontakt für die Dienstleistung.",
                type="String",
            ),
            MetadataInfo(
                name="Kontakt URL",
                description="Die URL des Kontakts für die Dienstleistung.",
                type="String",
            ),
        ],
    )

    auto_retriever = VectorIndexAutoRetriever(
        index=index,
        vector_store_info=vector_store_info,
        verbose=True,
    )

    query_auto = RetrieverQueryEngine(
        retriever=auto_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker],
    )

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

    hyde = HyDEQueryTransform(include_original=True)
    query_engine = index.as_query_engine()
    query_hyde = TransformQueryEngine(query_engine, query_transform=hyde)

    """
    6. RAG Fusion
    """

    # TODO: check again how the paper implemented this
    base_retriever = index.as_retriever(similarity_top_k=5)
    retriever = FusionRetriever(retriever=base_retriever)
    query_fusion = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker],
    )



