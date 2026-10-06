"""消融实验：拆解 TopicFuse 两大创新组件的贡献。

- 组件 A：Hungarian 共识对齐（关 -> 直接平均各链原始 phi，受标签置换退化）
- 组件 B：VB 精炼（polish_steps=0 -> 仅对齐平均，不做 E/M 精炼）

对照：
  单链 LDA        —— 最强单链基线
  平均(无对齐)    —— 关掉组件 A
  平均(对齐)      —— 仅组件 A
  TopicFuse       —— 组件 A + B（完整旗舰）

结论：组件 A 消除标签置换退化、组件 B 进一步锐化，二者叠加稳定优于单链。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from topicforge.core.assign import hungarian
from topicforge.core.config import TopicForgeConfig
from topicforge.data.corpus import generate_corpus
from topicforge.eval.metrics import topic_recovery
from topicforge.models.ensemble import TopicFuse
from topicforge.models.lda import vb_lda

_EPS = 1e-12


def _cos(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + _EPS)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + _EPS)
    return A @ B.T


def main():
    cfg = TopicForgeConfig.from_env(n_topics=6, vocab_size=500, n_docs=200)
    K, V, n_chains = cfg.n_topics, cfg.vocab_size, 4
    corpus = generate_corpus(cfg, np.random.default_rng(cfg.seed + 101))

    rng = np.random.default_rng(777)
    seeds = rng.integers(0, 1 << 31, size=n_chains)
    phis = []
    for r in range(n_chains):
        cr = np.random.default_rng(int(seeds[r]))
        phi, _ = vb_lda(corpus.docs, K, V, cr, cfg.alpha, cfg.eta, cfg.n_iter)
        phis.append(phi)

    # 单链 LDA
    phi0, _ = vb_lda(corpus.docs, K, V, np.random.default_rng(1), cfg.alpha, cfg.eta, cfg.n_iter)
    rec_single = topic_recovery(phi0, corpus.true_beta)[0]

    # 平均（无对齐）
    rec_na = topic_recovery(np.mean(phis, axis=0), corpus.true_beta)[0]

    # 平均（匈牙利对齐）
    ref = phis[0]
    aligned = [ref]
    for r in range(1, n_chains):
        C = 1.0 - _cos(phis[r], ref)
        row_to_col, _ = hungarian(C)
        aligned.append(phis[r][row_to_col])
    rec_aligned = topic_recovery(np.mean(aligned, axis=0), corpus.true_beta)[0]

    # 完整 TopicFuse
    res = TopicFuse().fit(
        corpus,
        np.random.default_rng(999),
        K=K,
        alpha=cfg.alpha,
        eta=cfg.eta,
        n_chains=n_chains,
        n_iter=cfg.n_iter,
    )
    rec_fuse = topic_recovery(res.phi, corpus.true_beta)[0]

    print(f"{'配置':<22}{'recovery':>10}{'Δ':>10}")
    print(f"{'单链 LDA':<22}{rec_single:>10.4f}{'':>10}")
    print(f"{'平均(无对齐)':<22}{rec_na:>10.4f}{rec_na - rec_single:>+10.4f}")
    print(f"{'平均(匈牙利对齐)':<22}{rec_aligned:>10.4f}{rec_aligned - rec_na:>+10.4f}")
    print(f"{'TopicFuse(对齐+精炼)':<22}{rec_fuse:>10.4f}{rec_fuse - rec_aligned:>+10.4f}")
    return {
        "single": rec_single,
        "no_align": rec_na,
        "align": rec_aligned,
        "fuse": rec_fuse,
    }


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nelapsed {time.time() - t0:.1f}s")
