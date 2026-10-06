"""NMF（纯 numpy 乘法更新，Frobenius 损失）。作为对照基线 / 可选 sklearn 后端。"""

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


def _nmf(X, K, rng, n_iter=300, init=None):
    D, V = X.shape
    if init is None:
        W = rng.random((D, K)) + 0.1
        H = rng.random((K, V)) + 0.1
    else:
        W, H = init
    for _ in range(n_iter):
        # update H
        num = W.T @ X
        den = W.T @ W @ H + _EPS
        H *= num / den
        # update W
        num = X @ H.T
        den = W @ H @ H.T + _EPS
        W *= num / den
    return W, H


class NMF:
    name = "nmf"

    def fit(self, corpus, rng, K=None, **kw):
        K = int(K or corpus.n_topics or 8)
        n_iter = int(kw.get("n_iter", 300))
        X = _doc_term_matrix(corpus.docs, corpus.vocab_size)
        # 轻量 TF 归一化，稳定更新
        Xn = X / (X.sum(1, keepdims=True) + _EPS)
        try:
            from sklearn.decomposition import NMF as SkNMF  # 可选后端

            if SkNMF is not None and X.sum() > 0:
                m = SkNMF(
                    n_components=K,
                    init="random",
                    random_state=int(rng.integers(1 << 30)),
                    max_iter=n_iter,
                )
                W = m.fit_transform(X)
                H = m.components_
                phi = H / (H.sum(1, keepdims=True) + _EPS)
                theta = W / (W.sum(1, keepdims=True) + _EPS)
                return TopicModelResult(
                    phi=phi, theta=theta, doc_topic_label=theta.argmax(1), name=self.name
                )
        except Exception:
            pass
        W, H = _nmf(Xn, K, rng, n_iter)
        phi = H / (H.sum(1, keepdims=True) + _EPS)
        theta = W / (W.sum(1, keepdims=True) + _EPS)
        return TopicModelResult(
            phi=phi, theta=theta, doc_topic_label=theta.argmax(1), name=self.name
        )
