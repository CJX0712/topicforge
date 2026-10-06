"""合成语料生成：标准 LDA 生成模型（Blei et al. 2003）。

真实主题-词分布 Beta_true 通过对每个主题取少量支撑词 + Gamma 质量生成，保证主题可区分、可恢复。
生成器固定 seed（经 core.seed.new_rng 派生），多次运行逐位一致。
"""

import json
import os

import numpy as np

from topicforge.core.errors import DataError
from topicforge.core.types import Corpus


def generate_corpus(cfg, rng):
    """生成标准 LDA 合成语料。"""
    K, V, N, L = cfg.n_topics, cfg.vocab_size, cfg.n_docs, cfg.avg_doc_len
    if rng is None:
        from ..core.seed import new_rng

        rng = new_rng(cfg.seed)

    # 真实主题-词分布：每个主题取一小撮支撑词，质量集中于其上
    support = max(2, V // (3 * K))
    beta_true = np.zeros((K, V))
    for k in range(K):
        idx = rng.choice(V, size=min(support, V), replace=False)
        raw = rng.gamma(5.0, 1.0, size=len(idx))
        beta_true[k, idx] = raw
        beta_true[k] /= beta_true[k].sum()

    # 文档-主题混合：每篇文档独立 Dirichlet(alpha)
    theta_true = rng.dirichlet(np.full(K, cfg.alpha), size=N)

    docs = []
    for d in range(N):
        nd = max(1, int(rng.poisson(L)))
        z = rng.choice(K, size=nd, p=theta_true[d])
        words = np.empty(nd, dtype=np.int64)
        for i, zi in enumerate(z):
            words[i] = rng.choice(V, p=beta_true[zi])
        docs.append(words)

    true_topic = np.array([int(theta_true[d].argmax()) for d in range(N)], dtype=np.int64)
    return Corpus(
        docs=docs,
        vocab_size=V,
        true_beta=beta_true,
        true_theta=theta_true,
        true_topic=true_topic,
        n_topics=K,
    )


def make_holdout(corpus, rng, frac=0.2):
    """按文档内 token 随机留出，返回 (train_docs, test_tokens)。

    train_docs: 仅含训练 token 的文档列表
    test_tokens: [(doc_id, word_id), ...]
    留出不跨文档，避免信息泄漏。
    """
    if rng is None:
        from ..core.seed import new_rng

        rng = new_rng()
    train_docs = []
    test_tokens = []
    for d, doc in enumerate(corpus.docs):
        doc = np.asarray(doc, dtype=np.int64)
        if len(doc) == 0:
            train_docs.append(doc)
            continue
        mask = rng.random(len(doc)) >= frac
        # 保证训练集非空
        if mask.sum() == 0:
            mask[0] = True
        train_docs.append(doc[mask])
        for i in np.where(~mask)[0]:
            test_tokens.append((d, int(doc[i])))
    return train_docs, test_tokens


def load_corpus_jsonl(path):
    """从 jsonl 载入语料：每行一个文档（词索引列表）。"""
    if not os.path.exists(path):
        raise DataError(f"语料文件不存在: {path}")
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(np.asarray(obj["tokens"], dtype=np.int64))
    if not docs:
        raise DataError("语料为空")
    V = int(max(int(w) for doc in docs for w in doc)) + 1
    return Corpus(docs=docs, vocab_size=V)
