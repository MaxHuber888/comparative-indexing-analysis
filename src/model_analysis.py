import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm


# ANALYSIS FUNCTIONS

def analyze_generation(model, model_name, doc_path):
    # Init results dict
    results = {"model": model_name}

    start = time.perf_counter()
    # GENERATE INDEX FROM DATA USING CURRENT MODEL
    model.generate_index(doc_path)

    # SAVE GENERATION TIME TO RESULTS
    generation_time = time.perf_counter() - start
    results['generation_time'] = generation_time

    # SAVE INDEX TO FILE
    model.save_index(model_name)

    # COUNT TOTAL EMBEDDED TOKENS (FROM FILE)
    emb_token_count = model.count_embedded_tokens()
    results['emb_token_count'] = emb_token_count

    return results


def analyze_retrieval(model, model_name, queries_path):
    # Init results dict
    results = []

    # LOAD QUERY DF
    query_df = pd.read_csv(queries_path)

    for index, row in query_df.iterrows():
        query = row["query"]
        answer = row["answer"]

        interval = time.perf_counter()

        # QUERY INDEX WITH CURRENT QUERY
        responses = model.query(query)

        query_time = time.perf_counter() - interval

        # SAVE QUERY TIME AND RESPONSE
        query_results = {"query_time": query_time, "query": query, "answer": answer}
        for i, resp in enumerate(responses):
            query_results["response_" + str(i)] = str(resp).replace(",", ":").replace("\n", " / ")
        results.append(query_results)

    return results


# QUERY/RESPONSE GENERATOR

def generate_questions_and_answers(doc_path, max_docs):
    load_dotenv()
    client = OpenAI()
    output_file = "generated_questions.txt"
    doc_count = 0

    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in tqdm(os.listdir(doc_path)):
            if filename.endswith(".txt"):
                with open(os.path.join(doc_path, filename), "r", encoding="utf-8") as infile:
                    doc_count += 1
                    text = infile.read()
                    prompt = f"Generate 5 questions and answers based on the following text:\n{text}\nQ1."
                    try:
                        completion = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt}
                            ]
                        )

                        msg = completion.choices[0].message.content
                        outfile.write(f"Document: {filename}\n")
                        outfile.write(f"Q1. {msg}\n\n")
                    except Exception as e:
                        print(e)

                    if doc_count >= max_docs:
                        return


# RESPONSE QUALITY ANALYSIS

def response_quality_analysis(responses_path):
    load_dotenv()
    client = OpenAI()

    # LOAD RESPONSES
    response_df = pd.read_csv(responses_path)

    # INITIALIZE RESULTS
    results = []

    for i, row in tqdm(response_df.iterrows(), desc="Responses"):
        query = row["query"]
        answer = row["answer"]
        response = row["response_0"]

        prompt = f'''You are comparing retrieved context to an expert answer on a given question. Here is the data:
                        [BEGIN DATA]
                        ************
                        [Question]: {query}
                        ************
                        [Expert Answer]: {answer}
                        ************
                        [Retrieved Context]: {response}
                        ************
                        [END DATA]
                        
                        Compare the factual content of the retrieved context with the expert answer.
                        
                        Ignore any differences in style, grammar, or punctuation.
                        
                        The retrieved context may either be a subset or superset of the expert answer,
                        or it may conflict with it. Determine which case applies. Keep your answer as short as possible.
                        
                        Answer the question by selecting one of the following options:
                        (A) The Retrieved Context contains all relevant information included in the Expert Answer.
                        (B) The Retrieved Context contains some but not all of the relevant information included in the Expert Answer.
                        (C) The Retrieved Context contains almost none of the relevant information included in the Expert Answer.
                        (D) The Retrieved Context contains none of the relevant information included in the Expert Answer.
                        (E) There is a disagreement between the Retrieved Context and the Expert Answer.'''

        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            msg = completion.choices[0].message.content
            results.append(msg)
        except Exception as e:
            results.append(str(e))

    return results
