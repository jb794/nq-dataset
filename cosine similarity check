import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

with open("expanded30qaset.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

model = SentenceTransformer("intfloat/e5-large-v2")

question_embeddings = []
answer_embeddings = []

for item in data:
    q_texts = [f"query: {q}" for q in item["questions"]]
    question_embs = model.encode(q_texts, convert_to_tensor=True, normalize_embeddings=True)
    avg_q_emb = question_embs.mean(dim=0)

    a_text = f"passage: {item['answer']}"
    answer_emb = model.encode(a_text, convert_to_tensor=True, normalize_embeddings=True)

    question_embeddings.append(avg_q_emb)
    answer_embeddings.append(answer_emb)

question_embeddings = torch.stack(question_embeddings)
answer_embeddings = torch.stack(answer_embeddings)

cos_sim_matrix = util.cos_sim(question_embeddings, answer_embeddings)

def compute_top_k_accuracy(sim_matrix, k):
    top_k = sim_matrix.topk(k=k, dim=1).indices
    correct = sum(i in top_k[i] for i in range(len(sim_matrix)))
    return correct / len(sim_matrix)

top1 = compute_top_k_accuracy(cos_sim_matrix, k=1)
top3 = compute_top_k_accuracy(cos_sim_matrix, k=3)
top5 = compute_top_k_accuracy(cos_sim_matrix, k=5)
avg_diag = cos_sim_matrix.diag().mean().item()

print("=== Cosine Similarity (e5-large-v2) ===")
print(f"Top-1 Accuracy: {top1 * 100:.2f}%")
print(f"Top-3 Accuracy: {top3 * 100:.2f}%")
print(f"Top-5 Accuracy: {top5 * 100:.2f}%")
print(f"Average True Cosine Similarity (diagonal): {avg_diag:.4f}")
