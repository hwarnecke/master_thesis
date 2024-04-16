# testing to load a custom embedding model from HuggingFace
# this is important as the application and data will be in german, so a german embedding model should be used

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_name = "T-Systems-onsite/cross-en-de-roberta-sentence-transformer"

embed_model = HuggingFaceEmbedding(model_name=model_name)

# test if it worked
embeddings = embed_model.get_text_embedding("Hallo Welt!")
print(len(embeddings))
print(embeddings[:5])

