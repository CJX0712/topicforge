"""基线：TFIDF+KMeans 与随机主题。"""

import numpy as np

from topicforge.core.types import TopicModelResult

_EPS = 1e-9


def _doc_term_matrix(docs, V):
    X = np.zeros((len(docs), V))
    for d, doc in enumerate(docs):
        doc = np.asarray(doc, dtype=np.int64)
        if len(doc) == 0:
            continue
        uniq, cnt = np.unique(doc, return_counts=True)
        X[d, uniq] = cnt
    return X


def _tfidf(X):
    df = (X > 0).sum(0)
    idf = np.log((1.0 + X.shape[0]) / (1.0 + df)) + 1.0
    tf = X / (X.sum(1, keepdims=True) + _EPS)
    M = tf * idf
    norms = np.linalg.norm(M, axis=1, keepdims=True) + _EPS
    return M / norms


def _kmeans(X, K, rng, n_iter=50):
    D = X.shape[0]
    if D == 0:
        return np.zeros(0, dtype=int), np.zeros((K, X.shape[1]))
    centers = X[rng.choice(D, K, replace=False)].copy()
    labels = np.zeros(D, dtype=int)
    for _ in range(n_iter):
        d2 = (X**2).sum(1)[:, None] + (centers**2).sum(1)[None, :] - 2 * X @ centers.T
        new = d2.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            m = labels == k
            if m.any():
                centers[k] = X[m].mean(0)
    return labels, centers


class TfidfKMeans:
    name = "tfidf_kmeans"

    def fit(self, corpus, rng, K=None, **kw):
        K = int(K or corpus.n_topics or 8)
        X = _doc_term_matrix(corpus.docs, corpus.vocab_size)
        M = _tfidf(X)
        labels, _ = _kmeans(M, K, rng)
        # 主题-词分布：簇内文档计数均值后归一化
        phi = np.zeros((K, corpus.vocab_size))
        for k in range(K):
            m = labels == k
            if m.any():
                phi[k] = X[m].sum(0)
            else:
                phi[k] = rng.random(corpus.vocab_size)
            s = phi[k].sum()
            phi[k] = (
                phi[k] / (s + _EPS) if s > 0 else np.ones(corpus.vocab_size) / corpus.vocab_size
            )
        theta = np.zeros((len(corpus.docs), K))
        theta[np.arange(len(labels)), labels] = 1.0
        return TopicModelResult(phi=phi, theta=theta, doc_topic_label=labels, name=self.name)


class RandomTopics:
    name = "random"

    def fit(self, corpus, rng, K=None, **kw):
        K = int(K or corpus.n_topics or 8)
        D = len(corpus.docs)
        phi = rng.dirichlet(np.full(corpus.vocab_size, 0.01), size=K)
        theta = rng.dirichlet(np.full(K, 0.1), size=D)
        return TopicModelResult(
            phi=phi, theta=theta, doc_topic_label=theta.argmax(1), name=self.name
        )
