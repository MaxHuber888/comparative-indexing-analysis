import os

import pandas as pd


def load_docs_from_folder(dir_path):
    files = os.listdir(dir_path)
    docs = []
    for file in files:
        if ".txt" in file:
            with open(dir_path + file) as f:
                lines = f.readlines()
        docs.append("\n".join(lines))
    return docs


def save_embeddings(df, embeddings_path):
    # Turn embedding into columns
    df_no_embeddings = df.drop(columns="embedding")
    embeddings_list = df["embedding"].to_list()
    embeddings_df = pd.DataFrame(embeddings_list)
    df = pd.concat([df_no_embeddings, embeddings_df], axis=1)
    # Save to file
    df.to_csv(embeddings_path, index=False)


def load_embeddings(embeddings_path):
    # Load from file
    df = pd.read_csv(embeddings_path)
    # Recreate embedding column
    embeddings_df = df.loc[:, df.columns.drop(["chunk_id", "doc", "length", "token_fraction", "chunk_text"])]
    embeddings_list = [[x for x in row[1]] for row in embeddings_df.iterrows()]
    df_no_embeddings = df[["chunk_id", "doc", "length", "token_fraction", "chunk_text"]]
    df = df_no_embeddings.assign(embedding=embeddings_list)

    # Return dataframe
    return df
