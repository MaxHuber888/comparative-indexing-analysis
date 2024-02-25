import torch
from models.abstract_model import IndexModel
from models.helpers.embed_query import embed_query
from models.helpers.file_handlers import save_embeddings, load_embeddings, load_docs_from_folder
from models.helpers.chunk_text import chunk_text
from models.helpers.embed_chunks import embed_chunks
import models.helpers.fit_net as fn


# TODO: add parameters and logging

class NetFitModel(IndexModel):
    def __init__(self):
        self.index = None
        self.nn_model = None
        self.tokens = []

    def generate_index(self, doc_path):
        # LOAD DOCS FROM FILE
        docs = load_docs_from_folder(doc_path)
        # EXTRACT TOKENS
        self.tokens = "\n".join(docs).split()
        # CHUNK TEXT (RULE BASED + SKIP INTERVALS)
        chunks_df = chunk_text(docs, max_chunk_size=300, interval_skips=1, with_semantic=False, delimiter=None)
        # GENERATE EMBEDDINGS
        self.index = embed_chunks(chunks_df)
        # INITIALIZE FRESH NETWORK
        self.nn_model = fn.Net()
        # TRAIN NETWORK
        self.nn_model = fn.train_model(
            model=self.nn_model,
            num_epochs=350,
            samples=self.index,
            verbose=True
        )

    def train_model(self):
        # Runs another training cycle
        self.nn_model = fn.train_model(
            model=self.nn_model,
            num_epochs=350,
            samples=self.index,
            verbose=True
        )

    # TODO: CHECK TO MAKE SURE ACTIONS ARE VALID
    def save_index(self, index_name):
        # SAVE INDEX
        save_embeddings(self.index, "indexes/net_fit_model/" + index_name + ".csv")
        # SAVE MODEL
        torch.save(self.nn_model.state_dict(), "indexes/net_fit_model/" + index_name + ".pt")
        # SAVE TOKENS
        with open("indexes/net_fit_model/" + index_name + ".txt", "w") as f:
            f.writelines("\n".join(self.tokens))

    def load_index(self, index_name):
        # LOAD INDEX
        self.index = load_embeddings("indexes/net_fit_model/" + index_name + ".csv")
        # LOAD MODEL
        self.nn_model = fn.Net()
        self.nn_model.load_state_dict(torch.load("indexes/net_fit_model/" + index_name + ".pt"))
        # LOAD TOKENS
        with open("indexes/net_fit_model/" + index_name + ".txt") as f:
            self.tokens = f.read().splitlines()

    def query(self, query):
        # EMBED QUERY
        query_embedding = embed_query(query)
        # RETRIEVE NODES
        return fn.retrieve_node(
            window_size=500,
            tokens=self.tokens,
            query_embedding=query_embedding,
            model=self.nn_model
        )
