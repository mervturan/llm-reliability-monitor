from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class RetrievalResult:
    contexts: list
    scores: list
    indices: list

class DenseRetriever:
    def __init__(self, model_name, contexts, batch_size=64):
        self.contexts = contexts
        self.model = SentenceTransformer(model_name)
        self.context_embeddings = self.model.encode(contexts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    def retrieve(self, question, top_k):
        query = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.context_embeddings @ query
        indices = np.argsort(scores)[::-1][:top_k]
        return RetrievalResult([self.contexts[i] for i in indices], [float(scores[i]) for i in indices], [int(i) for i in indices])

    def semantic_similarity(self, first, second):
        embeddings = self.model.encode([first, second], normalize_embeddings=True, show_progress_bar=False)
        return float(embeddings[0] @ embeddings[1])
