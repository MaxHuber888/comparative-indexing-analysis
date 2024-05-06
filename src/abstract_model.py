from abc import ABC, abstractmethod


class IndexModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate_index(self, docs):
        pass

    @abstractmethod
    def save_index(self, index_path):
        pass

    @abstractmethod
    def load_index(self, index_path):
        pass

    @abstractmethod
    def count_embedded_tokens(self):
        pass

    @abstractmethod
    def query(self, query):
        pass
