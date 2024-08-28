import pandas as pd
import numpy as np
from tabulate import tabulate
import json
import os


def get_filtered_and_sorted_filenames(directory, keywords):
    # Get all files and directories in the specified directory
    all_entries = os.listdir(directory)
    # Filter out directories and files containing "additional_data"
    filenames = [entry for entry in all_entries if
                 os.path.isfile(os.path.join(directory, entry)) and "additional_data" not in entry]

    # shorten filenames to the actual query engine description
    short = [name.split("_")[0] for name in filenames]
    names = {}
    for n in range(len(short)):
        names[short[n]] = filenames[n]

    # Create a dictionary to map keywords to filenames
    keyword_to_filename = {keyword: None for keyword in keywords}

    # Map filenames to the corresponding keyword
    for short, filename in names.items():
        for keyword in keywords:
            if keyword in short:
                keyword_to_filename[keyword] = directory + "/" + filename
                break

    # Sort filenames based on the given order of keywords
    sorted_filenames = [keyword_to_filename[keyword] for keyword in keywords if
                        keyword_to_filename[keyword] is not None]

    return sorted_filenames


def extract_control_keys(control_documents: dict, type: str) -> list:
    # extract either the names or the IDs:
    if type == "name":
        return control_documents['names']
    elif type == "id":
        return control_documents['ids']
    else:
        raise ValueError("Type must be either 'name' or 'id'")



def extract_test_keys(df: pd.DataFrame, columns:list, index: int) -> pd.array:
    nodes = df[columns]
    return nodes.iloc[index].astype(str).values


def compare_nodes(df: pd.DataFrame, control_path: str, columns: list, type: str = "name") -> [float, list[bool]]:
    # read in the control documents
    with open(control_path, 'r') as file:
        control_documents = json.load(file)

    # compare if the test_keys contain all elements of the control_keys
    # I.e. if one of the three retrieved nodes contains the name "Umwelt"
    comparison: list[bool] = []
    for i in range(len(control_documents)):
        control_keys = extract_control_keys(control_documents[i], type)
        test_keys = extract_test_keys(df, columns, i)
        comparison.append(all(s in test_keys for s in control_keys))

    # calculate the ratio of true to false
    ratio: float = np.mean(comparison)

    return ratio, comparison


def compare_embeddings():
    rerank_n = 2
    name_columns = []
    id_columns = []
    for i in range(rerank_n):
        name_columns.append(f"Node {i + 1} Metadata: Name")
        id_columns.append(f"Node {i + 1} ID")

    files = [
        "../logs/2024-08-28_14-41-31_gpt-4o-mini_text_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_text_stsb-distilroberta-base_retrieval_only_2024-08-28_14-41-31.csv",
        "../logs/2024-08-28_14-41-01_gpt-4o-mini_embed_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_embed_stsb-distilroberta-base_retrieval_only_2024-08-28_14-41-01.csv",
        "../logs/2024-08-28_14-40-48_gpt-4o-mini_gritlm_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_gritlm_stsb-distilroberta-base_retrieval_only_2024-08-28_14-40-48.csv",
        "../logs/2024-08-28_14-28-02_gpt-4o-mini_German_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_German_stsb-distilroberta-base_retrieval_only_2024-08-28_14-28-02.csv",
        "../logs/2024-08-28_14-28-07_gpt-4o-mini_cross_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_cross_stsb-distilroberta-base_retrieval_only_2024-08-28_14-28-07.csv",
        "../logs/2024-08-28_14-28-11_gpt-4o-mini_jina_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_jina_stsb-distilroberta-base_retrieval_only_2024-08-28_14-28-11.csv",
        "../logs/2024-08-28_14-28-13_gpt-4o-mini_multilingual_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_multilingual_stsb-distilroberta-base_retrieval_only_2024-08-28_14-28-13.csv",
        "../logs/2024-08-28_14-28-59_gpt-4o-mini_gte_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_gte_stsb-distilroberta-base_retrieval_only_2024-08-28_14-28-59.csv",
        "../logs/2024-08-28_14-35-34_gpt-4o-mini_stella_stsb-distilroberta-base_retrieval_only_retriever12_rerank3/base_gpt-4o-mini_stella_stsb-distilroberta-base_retrieval_only_2024-08-28_14-35-34.csv"
    ]
    order = [
        "OpenAI",
        "Cohere",
        "GritLM-7B",
        "German_Semantic",
        "cross-en-de",
        "jina-embeddings",
        "multilingual-e5",
        "gte-multilingual",
        "stella_en"
    ]

    ratios_name = []
    comparisons_name = []
    ratios_id = []
    comparisons_id = []
    control_path = "../questions_extended.json"
    for file in files:
        df = pd.read_csv(file, sep=';')
        ratio_name, comparison_name = compare_nodes(df=df, control_path=control_path, columns=name_columns, type="name")
        ratio_id, comparison_id = compare_nodes(df=df, control_path=control_path, columns=id_columns, type="id")

        ratios_name.append(ratio_name)
        comparisons_name.append(comparison_name)
        ratios_id.append(ratio_id)
        comparisons_id.append(comparison_id)

    data = [
        ["ratio name"] + ratios_name,
        ["ratio id"] + ratios_id
    ]

    print(tabulate(data, headers=order, tablefmt="grid"))


if __name__ == "__main__":
    compare_embeddings()
