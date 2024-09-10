import pandas as pd
import numpy as np
from tabulate import tabulate
import json
import os
import re


def get_filtered_and_sorted_filenames(directory, keywords) -> list:
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


def compare_nodes(df: pd.DataFrame, control_path: str, columns: list, type: str = "name", use_all: bool = True) -> [float, list[bool]]:
    # read in the control documents
    with open(control_path, 'r') as file:
        control_documents = json.load(file)

    # compare if the test_keys contain all elements of the control_keys
    # I.e. if one of the three retrieved nodes contains the name "Umwelt"
    comparison: list[bool] = []
    reciprocal_rank: list[float] = []
    for i in range(len(control_documents)):
        control_keys = extract_control_keys(control_documents[i], type)
        test_keys = extract_test_keys(df, columns, i)
        if use_all:
            comparison.append(all(s in test_keys for s in control_keys))
        else:
            comparison.append(any(s in test_keys for s in control_keys))

        # get reciprocal rank
        rank = 0
        for n in range(len(test_keys)):
            if test_keys[n] in control_keys:
                rank = 1/(n+1)
        reciprocal_rank.append(rank)


    # calculate the ratio of true to false
    ratio: float = np.mean(comparison)
    mrr: float = np.mean(reciprocal_rank)

    return ratio, comparison, mrr


def count_differences(listA: list[bool], listB: list[bool]) -> int:
    if len(listA) != len(listB):
        raise ValueError("Both lists must have the same length")

    difference_count = 0
    for a, b in zip(listA, listB):
        if a != b:
            difference_count += 1

    return difference_count


def find_n_highest_indices(values: list, n: int) -> list:
    # Pair each element with its index
    indexed_values = list(enumerate(values))
    # Sort the list of pairs based on the values in descending order
    sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1], reverse=True)
    # Extract the indices of the first n elements
    highest_indices = [index for index, value in sorted_indexed_values[:n]]
    return highest_indices


def print_question_comparison(order: list[str], comparisons: list[list[bool]]):
    header = []
    for n in range(1, 9):
        header.append(f"simple {n}")
    for n in range(1, 9):
        header.append(f"multi {n}")
    for n in range(1, 9):
        header.append(f"complex {n}")

    comparison_data = []
    for i in range(len(order)):
        comparison_data.append([order[i]] + comparisons[i])
    print(tabulate(comparison_data, headers=header,tablefmt="grid"))


def compare_best_n(best_n: int, ratios: list[float], comparisons: list[list[bool]], order: list[str]):
    indices = find_n_highest_indices(ratios, best_n)
    best_ordrer = [order[i] for i in indices]
    best_ratios = [ratios[i] for i in indices]
    best_comparisons = [comparisons[i] for i in indices]
    print_question_comparison(best_ordrer,best_comparisons)

def get_columns(df: pd.DataFrame) -> tuple[list,list]:
    column_names = list(df)
    names = []
    ids = []
    for column in column_names:
        id_regex = "Node [0-9] ID"
        name_regex = "Node [0-9] Metadata: Name"
        id_match = re.match(id_regex,column)
        name_match = re.match(name_regex,column)
        if not id_match is None:
            ids.append(column)
        elif not name_match is None:
            names.append(column)
    return ids, names

def get_avg_time(df: pd.DataFrame, type: str = "query") -> list[float]:
    match type:
        case "query":
            return np.mean(df["query_time"])
        case "completion":
            return np.mean(df["completion_time"])
        case "total":
            return np.mean(df["total_time"])


def compare(files: list, order: list, rerank_n: int = 3, use_all: bool = True):

    ratios_name = []
    comparisons_name = []
    ratios_id = []
    comparisons_id = []
    mrrs = []
    query_times = []
    control_path = "../questions_extended.json"
    for file in files:
        df = pd.read_csv(file, sep=';')
        id_columns, name_columns = get_columns(df)
        ratio_name, comparison_name, _ = compare_nodes(df=df, control_path=control_path, columns=name_columns, type="name", use_all=use_all)
        ratio_id, comparison_id, mrr = compare_nodes(df=df, control_path=control_path, columns=id_columns, type="id", use_all=use_all)
        query_times.append(get_avg_time(df,"query"))
        ratios_name.append(ratio_name)
        comparisons_name.append(comparison_name)
        ratios_id.append(ratio_id)
        comparisons_id.append(comparison_id)
        mrrs.append(mrr)

    data = [
        ["count name"] + [sum(inner_list) for inner_list in comparisons_name],
        ["ratio name"] + ratios_name,
        ["count id"] + [sum(inner_list) for inner_list in comparisons_id],
        ["ratio id"] + ratios_id,
        ["MRR"] + mrrs
    ]

    print(tabulate(data, headers=order,tablefmt="grid"))
    print("\n")
    #compare_best_n(3,ratios_id,comparisons_id,order)


def compare_embeddings():
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
        "GritLM",
        "Ger_Sem",
        "cross-de",
        "jina",
        "e5",
        "gte",
        "stella"
    ]
    compare(files=files, order=order, use_all=True)


def compare_reranker(use_all):
    folder = [
        "../logs/2024-09-09_12-16-36_gpt-4o-mini_text_bge-reranker-v2-m3_retrieval_only_retriever20_rerank3",
        "../logs/2024-09-09_12-37-24_gpt-4o-mini_text_bge-reranker-v2-gemma_retrieval_only_retriever20_rerank3",
        "../logs/2024-09-09_13-52-26_gpt-4o-mini_text_cross-encoder-mmarco-german-distilbert-base_retrieval_only_retriever20_rerank3",

    ]
    order = [
        "bge-m3",
        "bge-gemma",
        "distilbert",
        "gte",
        "monot5",
        "MiniLM",
        "gbert",
        "jina",
        "cohere"
    ]

    sorting = ["base", "rerank", "hybrid"]
    rerank_files = []
    for directory in folder:
        filenames = get_filtered_and_sorted_filenames(directory,sorting)
        rerank_files.append(filenames[1])   # currently we only need the reranking data

    compare(files=rerank_files, order=order, use_all=False)



if __name__ == "__main__":
    compare_embeddings()