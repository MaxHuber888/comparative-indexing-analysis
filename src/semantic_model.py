from src.abstract_model import IndexModel
from src.indexing_helpers import embed_query, load_docs_from_folder, load_embeddings, save_embeddings, \
    calculate_similarity, chunk_text_semantic, embed_chunks, count_embedded_tokens


class SemanticModel(IndexModel):
    def __init__(self):
        self.index = None
        self.tokens_embedded = 0

    def generate_index(self, doc_path):
        # LOAD DOCS FROM FILE
        docs = load_docs_from_folder(doc_path)
        # CHUNK TEXT (RULE BASED)
        chunks_df = chunk_text_semantic(docs, max_chunk_size=1000, threshold=0.4)
        # GENERATE EMBEDDINGS
        self.index = embed_chunks(chunks_df)
        # COUNT EMBEDDED TOKENS
        self.tokens_embedded = count_embedded_tokens(self.index)

    def save_index(self, index_name):
        # SAVE INDEX
        save_embeddings(self.index, "indexes/semantic_model/" + index_name + ".csv")

    def load_index(self, index_name):
        # LOAD INDEX
        self.index = load_embeddings("indexes/semantic_model/" + index_name + ".csv")

    def count_embedded_tokens(self):
        return self.tokens_embedded

    def query(self, query):
        # GENERATE QUERY EMBEDDING
        query_embedding = embed_query(query)

        # PERFORM SIMILARITY SEARCH
        chunks_with_similarity = calculate_similarity(self.index, query_embedding)

        # Rank based on similarity
        topk_chunks = chunks_with_similarity.sort_values(by="similarity", ascending=False)

        responses = []

        for chunk in topk_chunks[["similarity", "chunk_text"]].head(2).iterrows():
            responses.append(f"Cosine Similarity: {chunk[1][0]}\n{chunk[1][1]}")

        return responses
