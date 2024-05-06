import tiktoken
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, SimpleDirectoryReader, Settings
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from src.abstract_model import IndexModel
from dotenv import load_dotenv


# TODO : Identify parameters, add logging

class RuleBasedModel(IndexModel):
    def __init__(self):
        self.index = None
        self.max_chunk_size = 1000
        self.chunk_overlap = 50
        self.chunk_delimiter = None
        self.tokens_embedded = 0

    def generate_index(self, doc_path):
        load_dotenv()
        # Init token counter
        token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model("gpt-4").encode
        )
        Settings.callback_manager = CallbackManager([token_counter])

        # Create index
        docs = SimpleDirectoryReader(doc_path).load_data()
        self.index = VectorStoreIndex.from_documents(docs)

        # Get embedded token count and reset counter (for next indexing job)
        self.tokens_embedded = token_counter.total_embedding_token_count
        token_counter.reset_counts()

    def save_index(self, index_name):
        if self.index:
            self.index.storage_context.persist(persist_dir="indexes/rule_based_model/" + index_name)

    def load_index(self, index_name):
        load_dotenv()
        storage_context = StorageContext.from_defaults(persist_dir="indexes/rule_based_model/" + index_name)
        self.index = load_index_from_storage(storage_context)

    def count_embedded_tokens(self):
        return self.tokens_embedded

    def query(self, query):
        retriever = self.index.as_retriever()
        response_nodes = retriever.retrieve(query)
        response = []
        for node in response_nodes:
            score = node.score
            text = node.text
            response.append(f"Score: {str(score)}\n{text}")
        return response
