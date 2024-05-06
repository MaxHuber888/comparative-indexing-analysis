import os
import nltk
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import pandas as pd
from llama_index.embeddings.openai import OpenAIEmbedding
from tqdm import tqdm
from dotenv import load_dotenv


# FILE HANDLING HELPERS

def load_docs_from_folder(dir_path):
    files = os.listdir(dir_path)
    docs = []
    for file in files:
        if ".txt" in file:
            with open(dir_path + file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        docs.append("\n".join(lines))
    return docs


def save_embeddings(df, embeddings_path):
    # Turn embedding into columns
    df_no_embeddings = df.drop(columns="embedding")
    embeddings_list = df["embedding"].to_list()
    embeddings_df = pd.DataFrame(embeddings_list)
    df = pd.concat([df_no_embeddings, embeddings_df], axis=1)
    # Save to file
    df.to_csv(embeddings_path, index=False)


def load_embeddings(embeddings_path):
    # Load from file
    df = pd.read_csv(embeddings_path)
    # Recreate embedding column
    columns = df.columns
    drop_list = []
    for column in columns:
        if not column.isdigit():
            drop_list.append(column)

    embeddings_df = df.loc[:, df.columns.drop(drop_list)]
    embeddings_list = [[x for x in row[1]] for row in embeddings_df.iterrows()]
    df_no_embeddings = df[drop_list]
    df = df_no_embeddings.assign(embedding=embeddings_list)
    # Return dataframe
    return df


# VECTOR SIMILARITY HELPERS
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


# TEXT CHUNKING HELPERS

def chunk_text(docs, max_chunk_size=1000, interval_skips=0, delimiter=None):
    # TODO: add delimiter chunking
    chunks = []
    chunk_id = 0
    total_tokens = len(" ".join(docs).split())
    current_token = 0

    for i, doc in enumerate(tqdm(docs, desc="Chunking docs")):
        tokens = doc.split()
        current_chunk = {
            "chunk_id": chunk_id,
            "doc": i,
            "length": 0,
            "token_fraction": current_token / total_tokens,
            "chunk_text": ""
        }
        # Create chunks for each interval
        for x, token in enumerate(tokens):
            current_token += 1
            if current_chunk["length"] == max_chunk_size:
                # Finalize chunk
                stripped_chunk = current_chunk["chunk_text"].strip()
                current_chunk["chunk_text"] = stripped_chunk
                chunks.append(current_chunk)
                # Start new chunk
                chunk_id += 1
                current_chunk = {
                    "chunk_id": chunk_id,
                    "doc": i,
                    "length": 0,
                    "token_fraction": current_token / total_tokens,
                    "chunk_text": ""
                }

            # Add next token to chunk and increment chunk length
            current_chunk["chunk_text"] += token + " "
            current_chunk["length"] += 1

        # Finalize the last chunk in the doc
        stripped_chunk = current_chunk["chunk_text"].strip()
        current_chunk["chunk_text"] = stripped_chunk
        chunks.append(current_chunk)
        chunk_id += 1

    # Remove skipped intervals
    new_chunks = []
    skip = 0
    for chunk in chunks:
        if skip == 0:
            new_chunks.append(chunk)
            skip = interval_skips
        else:
            skip -= 1
    chunks = new_chunks

    # Return chunks and validation) chunks
    return pd.DataFrame(chunks)


def chunk_text_semantic(docs, max_chunk_size=1000, threshold=0.5):
    chunks = []
    chunk_id = 0
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    for i, doc in enumerate(tqdm(docs, desc="Chunking docs")):
        # SPLIT DOCS INTO SENTENCES
        sentences = nltk.sent_tokenize(doc)

        # TODO: logging for how many sentences per doc (to see how chunks reduce)

        # GENERATE SENTENCE EMBEDDINGS
        embeddings = sentence_model.encode(sentences)

        # CREATE CHUNKS BASED ON SENTENCE SIMILARITY
        current_chunk = {
            "chunk_id": chunk_id,
            "doc": i,
            "chunk_text": sentences[0]
        }
        for x in range(len(sentences) - 1):
            if cosine_similarity(embeddings[i], embeddings[i + 1]) > threshold:
                if len(current_chunk["chunk_text"].split()) < max_chunk_size:
                    # ADD TO CURRENT CHUNK
                    current_chunk["chunk_text"] += sentences[x + 1] + " "
                else:
                    # SAVE CURRENT CHUNK
                    chunks.append(current_chunk)
                    chunk_id += 1
                    # START NEW CHUNK
                    current_chunk = {
                        "chunk_id": chunk_id,
                        "doc": i,
                        "chunk_text": sentences[x + 1]
                    }
            else:
                # SAVE CURRENT CHUNK
                chunks.append(current_chunk)
                chunk_id += 1
                # START NEW CHUNK
                current_chunk = {
                    "chunk_id": chunk_id,
                    "doc": i,
                    "chunk_text": sentences[x + 1]
                }
        # SAVE LAST CHUNK OF DOC
        chunks.append(current_chunk)
        chunk_id += 1

    return pd.DataFrame(chunks)


# EMBEDDING HELPERS

def embed_chunks(chunks_df):
    load_dotenv()
    embed_model = OpenAIEmbedding()
    embeddings = []

    for chunk_id, chunk_text in tqdm(zip(chunks_df["chunk_id"], chunks_df["chunk_text"]),desc="Embedding chunks"):
        embeddings.append({"chunk_id": chunk_id, "embedding": embed_model.get_text_embedding(chunk_text)})

    return chunks_df.set_index("chunk_id").join(pd.DataFrame(embeddings).set_index("chunk_id")).reset_index()


def embed_query(query):
    # GENERATE EMBEDDING FOR QUERY
    load_dotenv()
    embed_model = OpenAIEmbedding()
    query_embedding = np.array(embed_model.get_text_embedding(query))

    return query_embedding


# TOKEN COUNT HELPERS

def num_tokens_from_string(text):
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    num_tokens = len(encoding.encode(text))
    return num_tokens


def count_embedded_tokens(df):
    sum = 0
    for index, row in df.iterrows():
        sum += num_tokens_from_string(row["chunk_text"])

    return sum
