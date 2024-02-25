from llama_index import OpenAIEmbedding
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd


def embed_chunks(chunks_df):
    load_dotenv()
    embed_model = OpenAIEmbedding()
    embeddings = []

    for chunk_id, chunk_text in zip(chunks_df["chunk_id"], chunks_df["chunk_text"]):
        embeddings.append({"chunk_id": chunk_id, "embedding": embed_model.get_text_embedding(chunk_text)})

    return chunks_df.set_index("chunk_id").join(pd.DataFrame(embeddings).set_index("chunk_id")).reset_index()

