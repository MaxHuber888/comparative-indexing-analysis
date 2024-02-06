import math
import os
from llama_index import OpenAIEmbedding
from get_tokens import get_tokens
from tqdm import tqdm
import csv


def generate_embeddings(doc_path, embed_path, open_ai_api_key, data_filename="example",
                        interval_size=50, window_size=20, verbose=True):
    # TOKENIZE DOC FROM FILE
    tokens = get_tokens(doc_path + data_filename + ".txt")

    # GET TOTAL TOKEN COUNT
    token_count = len(tokens)
    if verbose:
        print("TOKEN COUNT:", token_count)
        print("EMBEDDING COUNT:", math.ceil(token_count / interval_size))

    # GENERATE EMBEDDINGS/WRITE TO CSV
    tokens_chunked = []
    chunks = []

    with open(embed_path + data_filename + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in tqdm(iterable=range(0, token_count, interval_size), desc="GENERATING EMBEDDINGS", disable=not verbose):
            tokens_chunked.append(i)
            # Gather chunk
            chunk = tokens[i]
            if window_size > 0:
                for n in range(1, window_size + 1):
                    if not i - n < 0:
                        chunk = tokens[i - n] + " " + chunk
                    if not i + n > token_count - 1:
                        chunk += " " + tokens[i + n]
            chunks.append(chunk)

            # CREATE EMBEDDING
            embed_model = OpenAIEmbedding(api_key=open_ai_api_key)
            interval_embedding = embed_model.get_text_embedding(chunk)
            writer.writerow([float(i) / float(token_count)] + interval_embedding)

    if verbose:
        print("COMPLETE!")
