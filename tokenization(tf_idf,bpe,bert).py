import json
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase

with open("expanded30qaset.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

all_questions = [q for qa in dataset for q in qa["questions"]]

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_questions)

tfidf_tokenized = [tfidf_vectorizer.build_analyzer()(q) for q in all_questions]

with open("tfidf_tokenized_questions.jsonl", "w") as f:
    for i, q in enumerate(all_questions):
        json.dump({"original": q, "tfidf_tokens": tfidf_tokenized[i]}, f)
        f.write("\n")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_tokenized = [bert_tokenizer.tokenize(q) for q in all_questions]

with open("bert_tokenized_questions.jsonl", "w") as f:
    for i, q in enumerate(all_questions):
        json.dump({"original": q, "bert_tokens": bert_tokenized[i]}, f)
        f.write("\n")

corpus_path = "bpe_corpus.txt"
with open(corpus_path, "w") as f:
    f.write("\n".join(all_questions))

bpe_tokenizer = Tokenizer(BPE())
bpe_tokenizer.normalizer = Lowercase()
bpe_tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
bpe_tokenizer.train([corpus_path], trainer)

bpe_tokenized = [bpe_tokenizer.encode(q).tokens for q in all_questions]

with open("bpe_tokenized_questions.jsonl", "w") as f:
    for i, q in enumerate(all_questions):
        json.dump({"original": q, "bpe_tokens": bpe_tokenized[i]}, f)
        f.write("\n")

print("Outputs saved to tfidf_tokenized_questions.jsonl, bert_tokenized_questions.jsonl, bpe_tokenized_questions.jsonl")
