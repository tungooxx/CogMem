import numpy as np
import pytest
from cogmem.memory.embeddings import LocalEmbedder


@pytest.fixture(scope="module")
def embedder():
    return LocalEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")


class TestLocalEmbedder:
    def test_embed_single(self, embedder):
        vec = embedder.embed("put a clean mug in shelf 1")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_embed_batch(self, embedder):
        texts = ["put a mug in shelf", "examine a book"]
        vecs = embedder.embed_batch(texts)
        assert len(vecs) == 2
        assert len(vecs[0]) == 384

    def test_cosine_similarity(self, embedder):
        v1 = embedder.embed("put a clean mug in shelf 1")
        v2 = embedder.embed("put a clean plate in shelf 2")
        v3 = embedder.embed("examine a book under desklamp")
        sim_close = embedder.cosine_sim(v1, v2)
        sim_far = embedder.cosine_sim(v1, v3)
        assert sim_close > sim_far  # similar tasks should be closer
