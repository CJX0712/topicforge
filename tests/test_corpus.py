"""合成语料生成：确定性 + 形状 + held-out 无泄漏。"""

from collections import Counter

import numpy as np

from topicforge.core.config import TopicForgeConfig
from topicforge.data.corpus import generate_corpus, make_holdout


def test_corpus_deterministic():
    cfg = TopicForgeConfig.from_env(n_topics=5, vocab_size=300, n_docs=100)
    c1 = generate_corpus(cfg, np.random.default_rng(42))
    c2 = generate_corpus(cfg, np.random.default_rng(42))
    assert all(np.array_equal(a, b) for a, b in zip(c1.docs, c2.docs, strict=False))
    assert np.allclose(c1.true_beta, c2.true_beta)
    assert np.allclose(c1.true_theta, c2.true_theta)


def test_corpus_shapes():
    cfg = TopicForgeConfig.from_env(n_topics=5, vocab_size=300, n_docs=100)
    c = generate_corpus(cfg, np.random.default_rng(1))
    assert c.true_beta.shape == (5, 300)
    assert c.true_theta.shape == (100, 5)
    assert set(np.unique(c.true_topic.tolist())).issubset(set(range(5)))
    # 主题应可区分：真实主题-词支撑词有限
    assert (c.true_beta > 1e-4).sum(1).min() >= 2


def test_holdout_no_leakage():
    cfg = TopicForgeConfig.from_env(n_topics=5, vocab_size=300, n_docs=100, avg_doc_len=40)
    c = generate_corpus(cfg, np.random.default_rng(3))
    train, test = make_holdout(c, np.random.default_rng(4), frac=0.2)
    # held-out 是逐位置的干净切分：train + test 位置应还原原文档多重集
    for d in range(len(c.docs)):
        tw = [w for (dd, w) in test if dd == d]
        combined = list(np.asarray(train[d])) + tw
        assert Counter(combined) == Counter(np.asarray(c.docs[d]).tolist())
