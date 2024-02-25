import numpy as np
from numpy.linalg import norm


def cosine_similarity(a, b):
    A = np.array(a)
    B = np.array(b)
    return np.dot(A, B) / (norm(A) * norm(B))


def calculate_similarity(chunks_df, query_embedding):
    # Create matrix of query embedding
    chunks_df["query_vec"] = [query_embedding for _ in range(len(chunks_df))]
    chunks_df["similarity"] = 0
    # Calculate similarities
    for i in range(len(chunks_df)):
        chunks_df["similarity"][i] = cosine_similarity(chunks_df["embedding"][i], chunks_df["query_vec"][i])

    return chunks_df
