import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# (change this path to test other token types)
with open("tfidf_tokenized_questions.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

model = SentenceTransformer("intfloat/e5-large-v2")

token_importance_all = []

for idx, entry in enumerate(dataset):
    tokens = entry.get("tfidf_tokens", [])
    original_text = " ".join(tokens)

    full_input = f"query: {original_text}"
    full_embedding = model.encode(full_input)

    token_scores = []

    for i in range(len(tokens)):
        reduced_tokens = tokens[:i] + tokens[i+1:]
        reduced_input = f"query: {' '.join(reduced_tokens)}"

        reduced_embedding = model.encode(reduced_input)
        similarity = cosine_similarity(full_embedding.reshape(1, -1), reduced_embedding.reshape(1, -1))[0][0]
        distance = float(1 - similarity) 

        token_scores.append((tokens[i], distance))

    token_importance_all.append({
        "original_question": original_text,
        "token_scores": token_scores
    })

    if (idx + 1) % 10 == 0:
        print(f"\n Processed question {idx + 1}/{len(dataset)}")
        print("Example output:")
        print(f"Question: {original_text}")
        for tok, score in token_scores:
            print(f"  {tok:20s} → {score:.4f}")

# (change this path to save other token types)
with open("semantic_importance_tfidf.jsonl", "w") as f:
    for item in token_importance_all:
        f.write(json.dumps(item) + "\n")
