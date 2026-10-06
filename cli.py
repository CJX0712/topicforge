"""TopicForge 命令行入口。"""

import argparse
import os
import sys
import time

# 允许 `python cli.py` 直接运行：把仓库根加入 path，使 topicforge 作为包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topicforge.core.config import TopicForgeConfig  # noqa: E402
from topicforge.pipeline.topic_pipeline import run, summarize, write_benchmark  # noqa: E402


def _add_common(sp):
    sp.add_argument("--seed", type=int, default=12345)
    sp.add_argument("--n-topics", type=int, default=None, dest="n_topics")
    sp.add_argument("--vocab-size", type=int, default=None, dest="vocab_size")
    sp.add_argument("--n-docs", type=int, default=None, dest="n_docs")
    sp.add_argument("--avg-doc-len", type=int, default=None, dest="avg_doc_len")
    sp.add_argument("--alpha", type=float, default=None)
    sp.add_argument("--eta", type=float, default=None)
    sp.add_argument("--n-chains", type=int, default=None, dest="n_chains")
    sp.add_argument("--n-iter", type=int, default=None, dest="n_iter")
    sp.add_argument("--burnin", type=int, default=None)
    sp.add_argument("--holdout-frac", type=float, default=None, dest="holdout_frac")
    sp.add_argument("--datasets", type=int, default=3)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="topicforge", description="TopicForge · 主题建模系统（LDA/NMF/TopicFuse）"
    )
    sub = p.add_subparsers(dest="cmd")

    runp = sub.add_parser("run", help="运行基准，打印表格并落盘 benchmark.json")
    _add_common(runp)
    runp.add_argument("--out", default="benchmark.json")
    runp.add_argument("--no-perplexity", action="store_true")

    sub.add_parser("version", help="打印版本")
    args = p.parse_args(argv)
    cmd = args.cmd or "run"

    if cmd == "version":
        print("topicforge 0.1.0 (author 晨星)")
        return 0
    if cmd in ("run",):
        overrides = {}
        for k in (
            "seed",
            "n_topics",
            "vocab_size",
            "n_docs",
            "avg_doc_len",
            "alpha",
            "eta",
            "n_chains",
            "n_iter",
            "burnin",
            "holdout_frac",
        ):
            v = getattr(args, k, None)
            if v is not None:
                overrides[k] = v
        cfg = TopicForgeConfig.from_env(**overrides)
        seeds = [101 + i * 101 for i in range(args.datasets)]
        t0 = time.time()
        report = run(cfg, dataset_seeds=seeds, compute_perplexity=not args.no_perplexity)
        write_benchmark(report, args.out)
        print(summarize(report))
        print(f"\nelapsed {time.time() - t0:.1f}s · benchmark 已写入 {os.path.abspath(args.out)}")
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
