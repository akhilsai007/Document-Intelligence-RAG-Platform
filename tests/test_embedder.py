import numpy as np

from src.embeddings.embedder import Embedder


def test_hashing_embeddings_are_normalized_and_stable():
    e = Embedder(backend="hashing", dim=128)
    v1 = e.encode(["the quick brown fox"])
    v2 = e.encode(["the quick brown fox"])
    assert v1.shape == (1, 128)
    np.testing.assert_allclose(v1, v2)  # deterministic
    assert abs(np.linalg.norm(v1[0]) - 1.0) < 1e-5


def test_similarity_orders_sensibly():
    e = Embedder(backend="hashing", dim=256)
    base = e.encode(["vacation paid time off policy"])[0]
    near = e.encode(["paid time off and vacation days"])[0]
    far = e.encode(["kubernetes rolling deployment canary"])[0]
    assert float(base @ near) > float(base @ far)
