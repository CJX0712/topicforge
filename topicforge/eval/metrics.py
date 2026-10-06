"""主题建模评测指标（纯 numpy）。

核心约定（分数越大越优）：
- topic_recovery: 估计主题经 Hungarian 对齐真实主题后的平均余弦相似度（越高越好）
- doc_clustering_accuracy: 文档主导主题经 Hungarian 对齐真实主题后的聚类精度（越高越好）
- perplexity: held-out 困惑度（越低越好），与上面口径相反，单独标注
"""

import numpy as np

from topicforge.core.assign import hungarian

_EPS = 1e-12


def _unit_rows(M):
    M = np.asarray(M, dtype=float)
    n = np.linalg.norm(M, axis=1, keepdims=True) + _EPS
    return M / n


def _cosine_sim(A, B):
    A = _unit_rows(A)
    B = _unit_rows(B)
    return A @ B.T


def topic_recovery(phi_est, beta_true):
    """估计主题恢复度：返回 (mean_cosine, per_topic_cosine, permutation)。

    permutation[i] = 估计主题 i 匹配到的真实主题下标。
    """
    phi_est = np.asarray(phi_est, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float)
    C = 1.0 - _cosine_sim(phi_est, beta_true)  # (Kest, Ktrue)
    row_to_col, _ = hungarian(C)
    matched = 1.0 - C[np.arange(len(row_to_col)), row_to_col]
    return float(matched.mean()), matched, row_to_col


def doc_clustering_accuracy(theta_est, true_topic):
    """文档聚类精度（标签置换不变）。返回 (accuracy, permutation)。"""
    labels = np.asarray(theta_est).argmax(1)
    true_topic = np.asarray(true_topic)
    Kest = int(theta_est.shape[1])
    Ktrue = int(np.unique(true_topic).size)
    conf = np.zeros((Kest, Ktrue))
    for i in range(len(labels)):
        conf[labels[i], true_topic[i]] += 1
    row_to_col, _ = hungarian(-conf)  # 最大化匹配计数
    matched = conf[np.arange(Kest), row_to_col]
    return float(matched.sum() / len(labels)), row_to_col


def perplexity(theta, phi, test_tokens, eps=1e-12):
    """held-out 困惑度：test_tokens 为 [(doc_id, word_id), ...]。越低越好。"""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    logp = 0.0
    n = 0
    for d, w in test_tokens:
        pw = float(theta[d] @ phi[:, w])
        logp += np.log(max(pw, eps))
        n += 1
    if n == 0:
        return float("inf")
    return float(np.exp(-logp / n))
