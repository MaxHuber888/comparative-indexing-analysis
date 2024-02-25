import pandas as pd

# TODO: CLEAN IT UP, ADD SEMANTIC CHUNKING AND DELIMITER CHUNKING
def chunk_text(docs, max_chunk_size=1000, interval_skips=0, with_semantic=False, delimiter=None):
    chunks = []
    chunk_id = 0
    total_tokens = len(" ".join(docs).split())
    current_token = 0

    for i, doc in enumerate(docs):
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
