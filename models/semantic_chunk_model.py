from models.abstract_model import IndexModel


class SemanticChunkModel(IndexModel):
    def __init__(self):
        self.index = None

    def generate_index(self, doc_path):
        pass

    def save_index(self, index_path):
        pass

    def load_index(self, index_path):
        pass

    def query(self, query):
        pass
