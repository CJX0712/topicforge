"""TopicPipeline：生成多数据集 → 多模型训练 → 评测 → 聚合。"""

import json

import numpy as np

from topicforge.core.types import Corpus
from topicforge.data.corpus import generate_corpus, make_holdout
from topicforge.eval.metrics import doc_clustering_accuracy, perplexity, topic_recovery
from topicforge.models.baselines import RandomTopics, TfidfKMeans
from topicforge.models.ensemble import TopicFuse
from topicforge.models.lda import LDA
from topicforge.models.nmf import NMF


def build_registry():
    return {
        "topicfuse": TopicFuse(),
        "lda": LDA(),
        "nmf": NMF(),
        "tfidf_kmeans": TfidfKMeans(),
        "random": RandomTopics(),
    }


def run(cfg, dataset_seeds=None, registry=None, compute_perplexity=True):
    """跑完整基准。返回 report dict（含每模型 mean±std 与首数据集困惑度）。"""
    if dataset_seeds is None:
        dataset_seeds = [101, 202, 303]
    if registry is None:
        registry = build_registry()
    K = cfg.n_topics
    report = {
        "meta": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "n_datasets": len(dataset_seeds),
        "results": {
            name: {"recovery": [], "clustering": [], "perplexity": None} for name in registry
        },
    }

    for di, ds in enumerate(dataset_seeds):
        corpus = generate_corpus(cfg, np.random.default_rng(cfg.seed + ds))
        for name, model in registry.items():
            mrng = np.random.default_rng(cfg.seed * 7 + ds * 131 + (sum(ord(c) for c in name)) + 1)
            res = model.fit(
                corpus,
                mrng,
                K=K,
                alpha=cfg.alpha,
                eta=cfg.eta,
                n_iter=cfg.n_iter,
                burnin=cfg.burnin,
                n_chains=cfg.n_chains,
            )
            rec, _, _ = topic_recovery(res.phi, corpus.true_beta)
            clu, _ = doc_clustering_accuracy(res.theta, corpus.true_topic)
            report["results"][name]["recovery"].append(rec)
            report["results"][name]["clustering"].append(clu)

            if compute_perplexity and di == 0 and name in ("topicfuse", "lda", "nmf"):
                ho_rng = np.random.default_rng(cfg.seed * 3 + ds * 17 + 5)
                train_docs, test_tokens = make_holdout(corpus, ho_rng, cfg.holdout_frac)
                train_corpus = Corpus(docs=train_docs, vocab_size=corpus.vocab_size, n_topics=K)
                pr = model.fit(
                    train_corpus,
                    mrng,
                    K=K,
                    alpha=cfg.alpha,
                    eta=cfg.eta,
                    n_iter=cfg.n_iter,
                    burnin=cfg.burnin,
                    n_chains=cfg.n_chains,
                )
                report["results"][name]["perplexity"] = float(
                    perplexity(pr.theta, pr.phi, test_tokens)
                )

    for name in registry:
        recs = report["results"][name]["recovery"]
        clus = report["results"][name]["clustering"]
        report["results"][name]["recovery_mean"] = float(np.mean(recs))
        report["results"][name]["recovery_std"] = float(np.std(recs))
        report["results"][name]["clustering_mean"] = float(np.mean(clus))
        report["results"][name]["clustering_std"] = float(np.std(clus))
        del report["results"][name]["recovery"]
        del report["results"][name]["clustering"]

    # 头条对比：旗舰 vs 最强基线（单链 LDA）
    tf = report["results"]["topicfuse"]
    ld = report["results"]["lda"]
    report["headline"] = {
        "flagship": "topicfuse",
        "strong_baseline": "lda",
        "recovery_delta_mean": round(tf["recovery_mean"] - ld["recovery_mean"], 4),
        "recovery_threshold": 0.02,
        "pass": bool(tf["recovery_mean"] - ld["recovery_mean"] >= 0.02),
    }
    return report


def summarize(report):
    lines = []
    lines.append(
        f"数据集数={report['n_datasets']}  K={report['meta']['n_topics']}  V={report['meta']['vocab_size']}  N={report['meta']['n_docs']}"
    )
    lines.append(f"{'model':<14}{'recovery':>12}{'cluster':>12}{'perplexity':>14}")
    for name, r in report["results"].items():
        ppl = r.get("perplexity")
        ppls = f"{ppl:.3f}" if ppl is not None else "   -  "
        lines.append(
            f"{name:<14}{r['recovery_mean']:>10.4f}±{r['recovery_std']:.3f}{r['clustering_mean']:>10.4f}±{r['clustering_std']:.3f}{ppls:>14}"
        )
    h = report["headline"]
    lines.append(
        f"头条: topicfuse - lda 恢复度 Δ={h['recovery_delta_mean']:+.4f}  (门槛 {h['recovery_threshold']}, {'PASS' if h['pass'] else 'FAIL'})"
    )
    return "\n".join(lines)


def write_benchmark(report, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
