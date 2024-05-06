import torch
from src.abstract_model import IndexModel
from src.indexing_helpers import embed_query, save_embeddings, load_embeddings, load_docs_from_folder, chunk_text, \
    embed_chunks, count_embedded_tokens
import src.nner_helpers as fn


# TODO: add parameters and logging

class NNERModel(IndexModel):
    def __init__(self):
        self.index = None
        self.nn_model = None
        self.tokens = []
        self.tokens_embedded = 0

    def generate_index(self, doc_path):
        # LOAD DOCS FROM FILE
        docs = load_docs_from_folder(doc_path)
        # EXTRACT TOKENS
        self.tokens = "\n".join(docs).split()
        # CHUNK TEXT (RULE BASED + SKIP INTERVALS)
        chunks_df = chunk_text(docs, max_chunk_size=250, interval_skips=1, delimiter=None)
        # GENERATE EMBEDDINGS
        self.index = embed_chunks(chunks_df)
        # COUNT EMBEDDED TOKENS
        self.tokens_embedded = count_embedded_tokens(self.index)
        # INITIALIZE FRESH NETWORK
        self.nn_model = fn.Net()
        # TRAIN NETWORK
        self.nn_model = fn.train_model(
            model=self.nn_model,
            num_epochs=1500,
            samples=self.index,
            verbose=True
        )

    def train_model(self):
        # Runs another training cycle
        self.nn_model = fn.train_model(
            model=self.nn_model,
            num_epochs=1500,
            samples=self.index,
            verbose=True
        )

    # TODO: CHECK TO MAKE SURE ACTIONS ARE VALID
    def save_index(self, index_name):
        # SAVE INDEX
        save_embeddings(self.index, "indexes/nner_model/" + index_name + ".csv")
        # SAVE MODEL
        torch.save(self.nn_model.state_dict(), "indexes/nner_model/" + index_name + ".pt")
        # SAVE TOKENS
        with open("indexes/nner_model/" + index_name + ".txt", "w", encoding="utf-8") as f:
            f.writelines("\n".join(self.tokens))

    def load_index(self, index_name):
        # LOAD INDEX
        self.index = load_embeddings("indexes/nner_model/" + index_name + ".csv")
        # LOAD MODEL
        self.nn_model = fn.Net()
        self.nn_model.load_state_dict(torch.load("indexes/nner_model/" + index_name + ".pt"))
        # LOAD TOKENS
        with open("indexes/nner_model/" + index_name + ".txt", "r", encoding="utf-8") as f:
            self.tokens = f.read().splitlines()

    def count_embedded_tokens(self):
        return self.tokens_embedded

    def query(self, query):
        # EMBED QUERY
        query_embedding = embed_query(query)
        # RETRIEVE NODES
        return [fn.retrieve_node(
            window_size=500,
            tokens=self.tokens,
            query_embedding=query_embedding,
            model=self.nn_model
        )]
