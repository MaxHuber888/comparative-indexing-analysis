from models.abstract_model import IndexModel
from llama_index import VectorStoreIndex, load_index_from_storage, StorageContext, SimpleDirectoryReader
from dotenv import load_dotenv


# TODO : Identify parameters, add logging

class ChunkModel(IndexModel):
    def __init__(self):
        self.index = None
        self.max_chunk_size = 1000
        self.chunk_overlap = 50
        self.chunk_delimiter = None

    def generate_index(self, doc_path):
        load_dotenv()
        docs = SimpleDirectoryReader(doc_path).load_data()
        self.index = VectorStoreIndex.from_documents(docs)

    def save_index(self, index_name):
        if self.index:
            self.index.storage_context.persist(persist_dir="indexes/chunk_model/" + index_name)

    def load_index(self, index_name):
        storage_context = StorageContext.from_defaults(persist_dir="indexes/chunk_model/" + index_name)
        self.index = load_index_from_storage(storage_context)

    def query(self, query):
        retriever = self.index.as_retriever()
        response = retriever.retrieve(query)
        return response
