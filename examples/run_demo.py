"""端到端演示：生成合成语料 → 多模型基准 → 落盘 benchmark.json。"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topicforge.core.config import TopicForgeConfig  # noqa: E402
from topicforge.pipeline.topic_pipeline import run, summarize, write_benchmark  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    cfg = TopicForgeConfig.from_env()  # 可用 TOPICFORGE_* 环境变量覆盖
    seeds = [101, 202, 303]
    t0 = time.time()
    report = run(cfg, dataset_seeds=seeds, compute_perplexity=True)
    out = os.path.join(ROOT, "benchmark.json")
    write_benchmark(report, out)
    print(summarize(report))
    print(f"\nelapsed {time.time() - t0:.1f}s · 结果已写入 {out}")
    return report


if __name__ == "__main__":
    main()
