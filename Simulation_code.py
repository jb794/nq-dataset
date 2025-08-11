#top 100 docs dataset top-k evaluation with BERT embeddings
import sys
sys.path.insert(0, "/home/joshlocal/ISIT")    #my path to bert_semantic_repetition.py for optimization encoding

#imports
import torch
import random
import json
import numpy as np
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import bert_semantic_repetition as rep
import time

# PARAMETERS
num_iterations = 1000
R = 1.0                # repetition rate

# load tokenizer & model 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = SentenceTransformer("intfloat/e5-large-v2").to(device)

#needed to remove the trailing Normalize module so that I control normalization (when I was comparing l2 and cosine similarity)
last_key = list(model._modules.keys())[-1]               # usually 'normalize'
if model._modules[last_key]._get_name() == "Normalize":
    model._modules.pop(last_key)                         # now raw vectors

# load QA pairs
qa_pairs = []
with open("top100docs.jsonl","r",encoding="utf-8") as f: #set to dataset simulation is going to be run on 
    qa_pairs = [json.loads(l) for l in f]
num_docs = len(qa_pairs)
qa_pairs = qa_pairs[:num_docs]                      # keep only what we embed

# build document embeddings
doc_ids = [tokenizer.encode(a["answer"], add_special_tokens=False) for a in qa_pairs]
doc_texts = [" ".join(tokenizer.convert_ids_to_tokens(ids)) for ids in doc_ids]
with torch.no_grad():
    enc = model.encode(
            [f"passage: {t}" for t in doc_texts], #specific to infloat model (needs to change if using a different model)
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True) 
    d_emb = torch.tensor(enc, dtype=torch.float32, device=device)  # [num_docs, dim]   


# flatten all 30 questions into a single list of queries
queries, labels = [], []
for doc_id, qa in enumerate(qa_pairs):
    for q in qa["questions"]:
        queries.append(q)
        labels.append(doc_id) # ground-truth 

#tokenize queries
query_ids = [tokenizer.encode(q, add_special_tokens=False) for q in queries]

# load semantic importance
imp = [json.loads(l)["token_scores"] for l in open("semantic_importance_bert.jsonl")]
assert len(imp) >= len(queries), "importance file shorter than #queries"
imp = imp[:len(queries)]          
imp_ids_scores = [
    (
        tokenizer.convert_tokens_to_ids([tok for tok,_ in ts]), #token ids
        [score for _,score in ts] # scores
    )
    for ts in imp
]

# Top-k settings
ks = [1,3,5,10] #10 could be removed it was more for my own curiosity
epsilons = np.linspace(0,1,11)

# initialize error counters: for each k, keep separate counts for L2 and Cosine 
count_l2  = {k: np.zeros_like(epsilons) for k in ks}
count_cos = {k: np.zeros_like(epsilons) for k in ks}

#monte carlo simulation
start = time.perf_counter()
for ei, epsilon in enumerate(epsilons):
    for k in ks:
        count_l2[k][ei]  = 0
        count_cos[k][ei] = 0

    for _ in range(num_iterations):
        # sample a random query
        q_idx     = random.randrange(len(query_ids))
        full_ids  = query_ids[q_idx]
        tok_ids_r, _  = imp_ids_scores[q_idx]
        _, r_opt      = rep.repetitions_from_scores(imp[q_idx], epsilon, R) #optimization repetition vector

        # build repetition vector aligned to full_ids
        rep_map = {tid:int(round(r)) for tid,r in zip(tok_ids_r, r_opt)}  # cast to int
        r_vec   = np.array([rep_map.get(tid,0) for tid in full_ids], dtype=int) # [len(full_ids)]

        # simulate erasure
        MASK_ID  = tokenizer.mask_token_id   # 103 for BERT "[MASK]" token (no too important for this simulation more so for when I was doing LLM reconstructions)
        recv_ids = []
        for tid, rep_cnt in zip(full_ids, r_vec):
            p_keep = 1 - epsilon if rep_cnt==0 else 1 - epsilon**rep_cnt
            if random.random() < p_keep:
                recv_ids.append(tid) #one copy survived
            else:
                recv_ids.append(MASK_ID) #nothing survived, so the slot is masked

        # embed the survived query
        text_erased = tokenizer.decode(recv_ids, skip_special_tokens=True) #skip special tokens to avoid [MASK]
        q_emb_e = model.encode(f"query: {text_erased}", convert_to_tensor=True, normalize_embeddings=True).to(device)

        # compute distances/similarities to all docs
        sims_l2 = torch.cdist(q_emb_e.unsqueeze(0), d_emb, p=2).squeeze(0)
        sims_cos = torch.nn.functional.cosine_similarity(q_emb_e.unsqueeze(0), d_emb)

        # ground-truth doc index
        true_doc = labels[q_idx]

        l2_vals, l2_idx = torch.topk(-sims_l2, k=max(ks))  
        cos_vals, cos_idx = torch.topk(sims_cos, k=max(ks))


        for k in ks:
            topk_l2  = set(l2_idx[:k].cpu().tolist())
            topk_cos = set(cos_idx[:k].cpu().tolist())

            if true_doc not in topk_l2:
                count_l2[k][ei]  += 1
            if true_doc not in topk_cos:
                count_cos[k][ei] += 1

    for k in ks:
        count_l2[k][ei]  /= num_iterations
        count_cos[k][ei] /= num_iterations

elapsed = time.perf_counter() - start         
print(f"\nTotal simulation time: {elapsed:.2f} s")
for k in ks:    
    print(f"Top-{k:<2} L2 errors:     {count_l2[k]}")
for k in ks:
    print(f"Top-{k:<2} Cosine errors: {count_cos[k]}")
