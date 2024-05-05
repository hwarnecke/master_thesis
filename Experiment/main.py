"""
This is  the main file to run the experiment.
It will create the different kind of retrievers / query engines,
test them on a set of questions,
evaluate the results
and save them to disk.
"""

"""
1. Create the different retrievers

the retrievers are:
- default llama index retriever
- default llama index retriever with reranking
- hybrid retriever with keyword matching (BM25)
- AutoRetriever
- Hybrid Document Embdedding (HyDE)
- RAG Fusion
- possible combinations of the above which are yet to be defined

the retrievers will be put into standard LlamaIndex query engines
"""


"""
2. Run the experiment

The experiment will consist of running a set of questions through the different retrievers
This means, iterating through the list of query engines and query each with the same set of questions.
For each question the response objects will be put into the DeepEval metrics and the results will be saved.
I will also save the question, the response, the tokens used and the retrieved nodes.
Additionally, the time it took to answer the question will be measured.
All these values will be saved to a csv file with a retriever ID.
Some retrievers like the RAG Fusion or the HyDE will have some additional information I want to look at later.
I will create a separate file for these as they will have a different structure 
and can't be easily compared to the other retrievers.
"""