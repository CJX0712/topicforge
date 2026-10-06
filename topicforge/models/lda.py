"""LDA 推断：变分贝叶斯 LDA（Blei et al. 2003，纯 numpy，可向量化，确定性）。

选择 VB 而非吉布斯采样的理由：
- 完全向量化、确定性强（同 seed 逐位一致），无标签置换导致的跨样本平均退化；
- 不依赖 scipy（自实现 digamma），满足纯 numpy 离线兜底；
- 收敛稳定，是 LDA 推断的顶级标准方法之一。
约定：phi / lambda 形状 (K, V)，theta / gamma 形状 (D, K)。
"""

import numpy as np

from topicforge.core.errors import ModelError
from topicforge.core.types import TopicModelResult

_EPS = 1e-12


def _digamma(x):
    """纯 numpy digamma（psi 函数），x>0。先递推到 x>=6 再用渐近展开。"""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    y = x.copy()
    # 递推 psi(x) = psi(x+1) - 1/x，把 x 抬到 >=6
    while True:
        small = y < 6.0
        if not small.any():
            break
        out = np.where(small, out - 1.0 / y, out)
        y = np.where(small, y + 1.0, y)
    inv = 1.0 / y
    out = out + np.log(y) - 0.5 * inv - (inv * inv) / 12.0 + (inv**4) / 120.0 - (inv**6) / 252.0
    return out


def vb_lda(train_docs, K, V, rng, alpha=0.1, eta=0.01, n_iter=100, init=None):
    """变分贝叶斯 LDA。返回 (phi (K,V), theta (D,K))。"""
    doc_of, tokens = [], []
    for d, doc in enumerate(train_docs):
        for w in doc:
            doc_of.append(d)
            tokens.append(int(w))
    doc_of = np.asarray(doc_of, dtype=np.int64)
    tokens = np.asarray(tokens, dtype=np.int64)
    Ntok = len(tokens)
    D = len(train_docs)
    if Ntok == 0:
        raise ModelError("训练语料为空")
    if K > V:
        raise ModelError("K 不能超过词表大小 V")
    if init is None:
        lambda_ = eta + (rng.random((K, V)) + 0.1)
        gamma = alpha + rng.random((D, K))
    else:
        lambda_, gamma = init

    elog_beta_all = _digamma(lambda_) - _digamma(lambda_.sum(1))[:, None]  # (K, V)
    elog_theta_d = _digamma(gamma[doc_of]) - _digamma(gamma.sum(1))[doc_of, None]  # (Ntok, K)
    elog_beta_i = elog_beta_all[:, tokens].T  # (Ntok, K)
    log_phi = elog_theta_d + elog_beta_i
    log_phi -= log_phi.max(1, keepdims=True)
    phi = np.exp(log_phi)
    phi /= phi.sum(1, keepdims=True) + _EPS

    for _ in range(n_iter):
        # M-step：用 phi 更新全局/局部变分参数（bincount 累加，避免 np.add.at 开销）
        idx_wk = (np.broadcast_to(np.arange(K)[:, None], (K, Ntok)) * V + tokens[None, :]).ravel()
        n_wk = np.bincount(idx_wk, weights=phi.T.ravel(), minlength=K * V).reshape(K, V)
        lambda_ = eta + n_wk
        idx_g = (np.broadcast_to(doc_of[None, :], (K, Ntok)) * K + np.arange(K)[:, None]).ravel()
        gamma = alpha + np.bincount(idx_g, weights=phi.T.ravel(), minlength=D * K).reshape(D, K)

        # E-step：重算 phi
        elog_beta_all = _digamma(lambda_) - _digamma(lambda_.sum(1))[:, None]
        elog_theta_d = _digamma(gamma[doc_of]) - _digamma(gamma.sum(1))[doc_of, None]
        elog_beta_i = elog_beta_all[:, tokens].T
        log_phi = elog_theta_d + elog_beta_i
        log_phi -= log_phi.max(1, keepdims=True)
        phi = np.exp(log_phi)
        phi /= phi.sum(1, keepdims=True) + _EPS

    phi_est = lambda_ / (lambda_.sum(1, keepdims=True) + _EPS)
    theta_est = gamma / (gamma.sum(1, keepdims=True) + _EPS)
    return phi_est, theta_est


def em_polish_phi(phi, train_docs, alpha, eta, steps=10):
    """给定 phi 初值，做几步 VB E/M 锐化（用于旗舰精炼），返回 (phi, theta)。"""
    K, V = phi.shape
    D = len(train_docs)
    rng = np.random.default_rng(0)
    gamma0 = alpha + rng.random((D, K))
    return vb_lda(
        train_docs, K, V, rng, alpha=alpha, eta=eta, n_iter=steps, init=(phi.copy() * V, gamma0)
    )


class LDA:
    name = "lda"

    def fit(self, corpus, rng, K=None, **kw):
        K = int(K or corpus.n_topics or 8)
        alpha = float(kw.get("alpha", 0.1))
        eta = float(kw.get("eta", 0.01))
        n_iter = int(kw.get("n_iter", 100))
        phi, theta = vb_lda(corpus.docs, K, corpus.vocab_size, rng, alpha, eta, n_iter)
        return TopicModelResult(
            phi=phi, theta=theta, doc_topic_label=theta.argmax(1), name=self.name
        )
