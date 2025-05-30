#BERT importance simulation

import torch
import random
from scipy.stats import norm
import json
import query_sampling
import similarity
import mean_std_computation
import transform
import bert_semantic_repetition as rep
from scipy.stats import multivariate_normal
import scipy.stats as stats
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer


## PARAMETERS
alpha = 1  # Zipfian exponent
n = 1000  # number of documents
num_iterations = 10000
R = 2.0 #repetition rate
dtype = torch.float32  

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") #same model used for original tokenization
model     = BertModel.from_pretrained("bert-base-uncased")# same model used for importance
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = model.to(device)

vocab_size = tokenizer.vocab_size
print(f"vocab_size: {vocab_size}")

qa_pairs = []
with open("expanded30qaset.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        qa_pairs.append(json.loads(line.strip()))

documents = [item["answer"] for item in qa_pairs]
queries = [item["questions"] for item in qa_pairs]


############ NEW
doc_id_lists = [tokenizer.encode(a, add_special_tokens=False) for a in documents[:n]]
doc_texts    = [" ".join(tokenizer.convert_ids_to_tokens(ids)) for ids in doc_id_lists]

with torch.no_grad():
    d_emb = torch.tensor(model.encode(
            [f"passage: {t}" for t in doc_texts],
            convert_to_numpy=True, show_progress_bar=True)).to(device) 

queries       = [q for qa in qa_pairs for q in qa["questions"]]  
query_ids     = [tokenizer.encode(q, add_special_tokens=False) for q in queries]

with open("semantic_importance_bert.jsonl") as f:
    importance = [json.loads(l)["token_scores"] for l in f]

imp_ids_scores = [
    (tokenizer.convert_tokens_to_ids([tok for tok, _ in ts]),   
     [score for _, score in ts])                                
    for ts in importance
]

##############################################
###################### Monte Carlo Simulation ###################################
pr_vector = np.linspace(0, 1, 11)

average_error_l2 = np.zeros(len(pr_vector))
average_error_cosine = np.zeros((len(pr_vector)))

for i,epsilon in enumerate(pr_vector):
    error_l2 = 0
    error_cosine = 0
    total_pr_e = 0
    actual_rate_total = 0
    
    for j in range(num_iterations):



        ###################### QUERY SAMPLING ########################
        q_idx        = random.randrange(len(query_ids))
        full_ids     = query_ids[q_idx]         
        tok_ids_r, _ = imp_ids_scores[q_idx]      
        _, r_opt     = rep.repetitions_from_scores(importance[q_idx], epsilon,R)
        r_vec = np.zeros(len(full_ids), dtype=int)
        id2pos = {tid:i for i,tid in enumerate(full_ids)}
        for tid, rep_cnt in zip(tok_ids_r, r_opt):
            if tid in id2pos: 
                r_vec[id2pos[tid]] = rep_cnt            

        survive = []
        for tid, rep_cnt in zip(full_ids, r_vec):
            keep = np.random.binomial(
                    1,
                    1 - epsilon if rep_cnt == 0          # TF-IDF rule for r = 0
                    else 1 - epsilon**rep_cnt            # rule for r > 0
                )            
            if keep:
                survive.append(tid)


        if len(survive) == 0:        # whole query wiped out → guaranteed error
            error_l2 += 1; error_cosine += 1
            continue


        text_survive = tokenizer.decode(
            survive,
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True
        )
        text_survive = f"query: {text_survive}"
        q_emb_e = model.encode(
            text_survive,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        ###### L2-Simunlation
        s_l2_e = torch.cdist(q_emb_e.unsqueeze(0), d_emb, p=2).squeeze(0)      
        idx_l2_e = torch.argmin(s_l2_e).item()
        l2_truth_idx = q_idx // 30                     
        if idx_l2_e != l2_truth_idx:
            error_l2 += 1

        ######### Cosine_Simulation
        s_cosine_e = torch.nn.functional.cosine_similarity(q_emb_e.unsqueeze(0), d_emb)      
        idx_cosine_e = torch.argmax(s_cosine_e).item()
        cosine_truth_idx = q_idx // 30                     
        if idx_cosine_e != cosine_truth_idx:
            error_cosine += 1

    average_error_l2[i] = error_l2 / num_iterations
    average_error_cosine[i] = error_cosine / num_iterations
        

print(f'num_iter:{num_iterations}_rate:{R}_average_error_l2: {average_error_l2}')
print(f'num_iter:{num_iterations}_rate:{R}_average_error_cosine: {average_error_cosine}')
