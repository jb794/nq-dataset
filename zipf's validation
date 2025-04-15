import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

question_words = []
answer_words = []
questions = []

with open("expanded30qaset.jsonl", "r") as f:
    for line in f:
        qa = json.loads(line)
        questions = qa.get("questions", [])
        for i in range(min(len(questions), 10)):  
            question = questions[i]
            question_words.extend(question.split())         
        answer = qa.get("answer", "")
        answer_words.extend(answer.split())  

def zipf(words, title):
    word_counts = Counter(words)  
    sorted_word_counts = word_counts.most_common()  
    ranks = np.arange(1, len(sorted_word_counts) + 1) 
    frequencies = [count for word, count in sorted_word_counts]  

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker='o', linestyle = 'None')
    plt.title(f"Zipf's Law: {title}")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.show()
    return ranks, frequencies


Q_ranks, Q_frequencies = zipf(question_words, "Question Set")
A_ranks, A_frequencies = zipf(answer_words, "Answer Set")
