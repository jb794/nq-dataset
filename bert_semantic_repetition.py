#bert_semantic_repetition.py

import numpy as np

def allocate_symbols_dp(s, eps, m):
    n = len(s)
    value = [[(1 - eps**k) * s[i] for k in range(m+1)] for i in range(n)]    
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

def repetitions_from_scores(token_scores, eps, R, scale=1.0):
    tokens, scores = zip(*token_scores)
    n = len(tokens)
    m0 = int(R * n)
    total = sum(scores) or 1.0
    s_norm = [sc/total for sc in scores]
    r_opt, _ = allocate_symbols_dp(s_norm, eps, m0)
    if scale != 1.0:
        r_opt = [max(0, int(round(scale * r))) for r in r_opt]
    return tokens, r_opt
