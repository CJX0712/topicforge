"""VB-LDA 正确性：输出合法 + 可恢复合成主题。"""

import numpy as np
import pytest

from topicforge.core.config import TopicForgeConfig
from topicforge.core.errors import ModelError
from topicforge.data.corpus import generate_corpus
from topicforge.eval.metrics import topic_recovery
from topicforge.models.lda import vb_lda


def test_lda_recovers():
    cfg = TopicForgeConfig.from_env(n_topics=5, vocab_size=300, n_docs=100, avg_doc_len=40)
    c = generate_corpus(cfg, np.random.default_rng(10))
    phi, theta = vb_lda(c.docs, 5, 300, np.random.default_rng(11), n_iter=80)
    assert phi.shape == (5, 300)
    assert np.allclose(phi.sum(1), 1.0, atol=1e-6)
    assert theta.shape == (100, 5)
    rec, _, _ = topic_recovery(phi, c.true_beta)
    assert rec > 0.6


def test_lda_empty_raises():
    with pytest.raises(ModelError):
        vb_lda([], 5, 300, np.random.default_rng(0))
