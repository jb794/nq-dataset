import json
import numpy as np

def allocate_symbols_dp(s, eps, m):
    n = len(s)
    # Precompute value[i][k] = (1 - eps**k) * s[i]
    value = [[(1 - eps**k) * s[i] for k in range(m+1)] for i in range(n)]
    
    # DP table V and back-pointers choice
    V = [[0.0] * (m+1) for _ in range(n+1)]
    choice = [[0] * (m+1) for _ in range(n+1)]
    
    for i in range(1, n+1):
        for b in range(m+1):
            best_val, best_k = -float('inf'), 0
            for k in range(b+1):
                val = V[i-1][b-k] + value[i-1][k]
                if val > best_val:
                    best_val, best_k = val, k
            V[i][b] = best_val
            choice[i][b] = best_k
    
    # Backtrack to get repetitions per token
    r = [0] * n
    rem = m
    for i in range(n, 0, -1):
        k = choice[i][rem]
        r[i-1] = k
        rem -= k
    
    return r, V[n][m]

def process_all_questions(input_path, output_path, eps):
    with open(input_path, 'r', encoding='utf-8-sig') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            token_scores = data.get('token_scores', [])
            if not token_scores:
                continue
            tokens, scores = zip(*token_scores)
            total = sum(scores) or 1.0
            s = [score / total for score in scores]
            m = len(tokens)
            r_opt, max_val = allocate_symbols_dp(s, eps, m)
            out_rec = {
                "tokens": tokens,
                "repetitions": r_opt,
                "value": max_val
            }
            fout.write(json.dumps(out_rec) + "\n")

if __name__ == "__main__":
    # — User parameters — 
    INPUT_PATH = "semantic_importance_bert.jsonl"
    EPS_list = np.linspace(0, 1, 11)

    for EPS in EPS_list:
        OUTPUT_PATH = f"repetition_plan_eps_{EPS:.2f}.jsonl"
        process_all_questions(INPUT_PATH, OUTPUT_PATH, EPS)
