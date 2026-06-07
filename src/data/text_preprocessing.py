from collections import Counter
import pandas as pd
import torch 

def normalize_text(text):
    return str(text).lower().strip()

def normalize_answer(answer):
    return normalize_text(answer)

def build_answer_vocab(csv_path, answer_column, top_k=500):
    df = pd.read_csv(csv_path)
    answers = df[answer_column].dropna().astype(str).str.lower().str.strip().tolist()
    counter = Counter(answers)
    most_common = counter.most_common(top_k)
    answer_to_idx = {
        answer: idx for idx, (answer, _) in enumerate(most_common)
    }
    idx_to_answer = {
        idx: answer for answer, idx in answer_to_idx.items()
    }
    return answer_to_idx, idx_to_answer

def build_question_vocab(csv_path, question_column, min_freq=1):
    df = pd.read_csv(csv_path)
    questions = df[question_column].dropna().astype(str).str.lower().str.strip().tolist()
    counter = Counter()
    for question in questions:
        counter.update(question.split())

    word_to_idx = {
        "<pad>": 0,
        "<unk>": 1
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)

    idx_to_word = {
        idx: word for word, idx in word_to_idx.items()
    }
    return word_to_idx, idx_to_word

def encode_question(question, word_to_idx, max_question_len):
    question = normalize_text(question)
    tokens = question.split()
    ids = []
    for token in tokens[:max_question_len]:
        token_id = word_to_idx.get(token, word_to_idx["<unk>"])
        ids.append(token_id)
    while len(ids) < max_question_len:
        ids.append(word_to_idx["<pad>"])
    return torch.tensor(ids, dtype=torch.long)

def encode_answer(answer, answer_to_idx):
    answer = normalize_text(answer)
    label = answer_to_idx[answer]
    return torch.tensor(label, dtype=torch.long)