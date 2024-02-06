import numpy as np
from llama_index import OpenAIEmbedding
from sklearn.linear_model import LinearRegression


def retrieve_node(window_size, tokens, query, model, OPEN_AI_API_KEY):
    # GENERATE EMBEDDING FOR QUERY
    embed_model = OpenAIEmbedding(api_key=OPEN_AI_API_KEY)
    query_embedding = np.array(embed_model.get_text_embedding(query))
    query_embedding = query_embedding.reshape(1, -1)

    # GENERATE PREDICTED TOKEN
    predicted_token_fraction = model.predict(query_embedding)
    print(predicted_token_fraction)
    token_length = len(tokens)
    target_index = round(predicted_token_fraction[0] * token_length)
    print("Predicted Location:", target_index)

    # RETRIEVE PREDICTED CHUNK
    trailing_chunk = " ".join(tokens[target_index:target_index + window_size])
    #print("Chunk Retrieved:", trailing_chunk)

    return trailing_chunk
