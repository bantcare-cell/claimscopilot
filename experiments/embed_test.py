"""Learning experiment (not part of the app): embeddings from first principles.

Proves that embeddings capture MEANING, not words: "insulin" vs "sugar
medicine" share no words but score high; an unrelated sentence scores near 0.
Cosine similarity computed by hand with NumPy instead of a library helper.

Result on all-MiniLM-L6-v2: 0.58 (same meaning) vs 0.03 (unrelated).
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

a = model.encode("Is insulin covered under my plan?")
b = model.encode("Will my sugar medicine be paid for?")
c = model.encode("What is the capital of France?")


def closeness(v1, v2):
    """Cosine similarity: dot product scaled by vector lengths -> range -1..1."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


print("insulin vs sugar medicine:", closeness(a, b))
print("insulin vs capital of France:", closeness(a, c))
