"""NMF 正确性。"""

import numpy as np

from topicforge.core.config import TopicForgeConfig
from topicforge.data.corpus import generate_corpus
from topicforge.eval.metrics import topic_recovery
from topicforge.models.nmf import NMF


def test_nmf_runs():
    cfg = TopicForgeConfig.from_env(n_topics=5, vocab_size=300, n_docs=100)
    c = generate_corpus(cfg, np.random.default_rng(20))
    res = NMF().fit(c, np.random.default_rng(21), K=5, n_iter=50)
    assert res.phi.shape == (5, 300)
    assert np.allclose(res.phi.sum(1), 1.0, atol=1e-6)
    rec, _, _ = topic_recovery(res.phi, c.true_beta)
    assert rec > 0.6
