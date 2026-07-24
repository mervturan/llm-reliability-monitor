import re, string
from collections import Counter

def normalize_answer(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def exact_match(prediction, references):
    prediction = normalize_answer(prediction)
    return float(any(prediction == normalize_answer(r) for r in references))

def token_f1(prediction, references):
    pred = normalize_answer(prediction).split()
    if not pred:
        return 0.0
    values = []
    for reference in references:
        ref = normalize_answer(reference).split()
        overlap = sum((Counter(pred) & Counter(ref)).values())
        if overlap == 0:
            values.append(0.0)
            continue
        precision, recall = overlap / len(pred), overlap / max(len(ref), 1)
        values.append(2 * precision * recall / (precision + recall))
    return max(values, default=0.0)
