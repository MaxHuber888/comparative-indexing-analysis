from dotenv import load_dotenv
import numpy as np
from llama_index.embeddings.openai import OpenAIEmbedding


def embed_query(query):
    # GENERATE EMBEDDING FOR QUERY
    load_dotenv()
    embed_model = OpenAIEmbedding()
    query_embedding = np.array(embed_model.get_text_embedding(query))

    return query_embedding
