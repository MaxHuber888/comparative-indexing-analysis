from models.abstract_model import IndexModel
from models.helpers.chunk_text import chunk_text
from models.helpers.embed_chunks import embed_chunks
from models.helpers.file_handlers import save_embeddings, load_embeddings, load_docs_from_folder
from models.helpers.embed_query import embed_query
from models.helpers.similarity_search import calculate_similarity


# TODO : IMPLEMENT FULLY

class ChunkSlideModel(IndexModel):
    def __init__(self):
        self.index = None

    def generate_index(self, doc_path):
        # LOAD DOCS FROM FILE
        docs = load_docs_from_folder(doc_path)
        # CHUNK TEXT (RULE BASED + SKIP INTERVALS)
        chunks_df = chunk_text(docs, max_chunk_size=1000, interval_skips=0)
        # GENERATE EMBEDDINGS
        self.index = embed_chunks(chunks_df)

    def save_index(self, index_name):
        # SAVE INDEX
        save_embeddings(self.index, "indexes/chunk_slide_model/" + index_name + ".csv")

    def load_index(self, index_name):
        # LOAD INDEX
        self.index = load_embeddings("indexes/chunk_slide_model/" + index_name + ".csv")

    def query(self, query):
        # GENERATE QUERY EMBEDDING
        query_embedding = embed_query(query)

        # PERFORM SIMILARITY SEARCH
        chunks_with_similarity = calculate_similarity(self.index, query_embedding)

        # Rank based on similarity
        topk_chunks = chunks_with_similarity.sort_values(by="similarity", ascending=False)

        # COMBINE CHUNKS
        selected_chunk_ids = topk_chunks["chunk_id"].head(3).tolist()
        combined_chunks = []
        for chunk_id in selected_chunk_ids:
            combined_chunks.append({
                "lengths": [
                    chunks_with_similarity["length"][max(chunk_id - 1, 0)],
                    chunks_with_similarity["length"][chunk_id],
                    chunks_with_similarity["length"][min(chunk_id + 1, len(self.index))]
                ],
                "text": chunks_with_similarity["chunk_text"][max(chunk_id - 1, 0)] + "\n" +
                        chunks_with_similarity["chunk_text"][chunk_id] + "\n" +
                        chunks_with_similarity["chunk_text"][min(chunk_id + 1, len(self.index))]
                ,
                "similarities": [
                    chunks_with_similarity["similarity"][max(chunk_id - 1, 0)],
                    chunks_with_similarity["similarity"][chunk_id],
                    chunks_with_similarity["similarity"][min(chunk_id + 1, len(self.index))]
                ],
            })

        # ISOLATE IDEAL CHUNK
        ideal_chunks = []
        for chunk in combined_chunks:
            similarity_diff = chunk["similarities"][0] - chunk["similarities"][2] / chunk["similarities"][1]
            norm_diff = int(similarity_diff * chunk["lengths"][1])
            chunk_tokens = chunk["text"].split(" ")
            ideal_chunk_tokens = chunk_tokens[
                                 chunk["lengths"][0] - norm_diff: chunk["lengths"][0] - norm_diff + chunk["lengths"][1]]
            ideal_chunks.append({
                "chunk_text": " ".join(ideal_chunk_tokens),
                "original_similarity": chunk["similarities"][1],
                "slide": -1*norm_diff
            })

        return ideal_chunks
