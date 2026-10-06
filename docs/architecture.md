# 架构设计 · TopicForge

> 世界级主题建模系统架构说明 · 作者：晨星

## 1. 设计目标

- **正确性**：推断算法须在数学上可验证（VB-LDA 收敛、Hungarian 对齐最优）。
- **确定性**：同 seed 两次运行逐位一致，可复现、可审计。
- **零联网运行时**：仅需 `numpy`，纯 NumPy 离线兜底。
- **可量化优于强基线**：旗舰 TopicFuse 在主题恢复度上显著超过单链 VB-LDA。

## 2. 整体数据流

```
            ┌──────────────────────────────────────────────┐
            │            TopicPipeline.run(cfg)             │
            └──────────────────────────────────────────────┘
                                │
        ┌─────────────── 对每个 dataset_seed ───────────────┐
        │                                                    │
        ▼                                                    │
 generate_corpus(cfg, rng)  ──►  Corpus(docs, true_beta, ...)  │
        │                                                    │
        │   对每个模型 fit(corpus, rng, K, alpha, eta, ...)    │
        ▼                                                    │
 TopicModelResult(phi, theta, doc_topic_label)               │
        │                                                    │
        ▼                                                    │
 topic_recovery / doc_clustering_accuracy / perplexity        │
        │                                                    │
        └──────────────────► 聚合 mean±std + 头条对比 ◄────────┘
```

## 3. 模块职责

| 模块 | 职责 |
|------|------|
| `core/config.py` | 配置 dataclass，`TOPICFORGE_*` 环境变量覆盖 + `validate()` 校验 |
| `core/seed.py` | 全局确定性入口 `set_all / get_seed / new_rng` |
| `core/errors.py` | E100~E500 异常体系（Config/Data/Model/Eval/Pipeline） |
| `core/types.py` | `Corpus` / `TopicModelResult` 数据契约 |
| `core/assign.py` | 纯 Python O(n³) Hungarian 增广路算法 |
| `data/corpus.py` | 标准 LDA 生成模型合成语料 + 逐 token held-out 留出（防泄漏） |
| `models/lda.py` | VB-LDA（`vb_lda` / `em_polish_phi` / `LDA`） |
| `models/nmf.py` | NMF（乘法更新 + 可选 sklearn 后端） |
| `models/ensemble.py` | `TopicFuse` 旗舰 |
| `models/baselines.py` | `TfidfKMeans` / `RandomTopics` |
| `eval/metrics.py` | recovery / clustering / perplexity |
| `pipeline/topic_pipeline.py` | 注册表 + 多数据集基准 + 聚合 + 头条 |

## 4. 推断算法选型：VB-LDA 而非吉布斯

- **确定性**：吉布斯采样是随机 MCMC kernel，跨样本平均需处理标签置换，且批量向量化采样是
  已知的“坏 kernel”（实测 recovery 仅 0.44~0.46，4000 次迭代仍 0.50）。
- **向量化 & 可复现**：VB-LDA 完全向量化，`np.bincount` 累加替代 `np.add.at`，同 seed 逐位一致。
- **零 scipy 依赖**：自实现 `digamma`（递推 + 渐近展开），满足纯 NumPy 离线兜底。

约定：`phi / lambda` 形状 `(K, V)`，`theta / gamma` 形状 `(D, K)`。

## 5. TopicFuse 旗舰原理

1. **多链**：R 条独立 `vb_lda` 链，由 `rng.integers` 派生确定性子种子。
2. **共识对齐**：以链 0 为参考，链 r 的主题经代价矩阵 `C = 1 - cosine(phi_r, phi_0)`
   的匈牙利匹配后重排，使各链主题一一对应再平均，消除标签置换退化。
3. **VB 精炼**：以集成 `phi` 为初值做数步 VB E/M（`em_polish_phi`），产出与 `phi` 自洽的 `theta`。

## 6. 评测与标签置换不变性

- 主题恢复度与文档聚类精度均经 **Hungarian 最优匹配**对齐真实标签，分数对标签置换不变。
- Hungarian 自实现与 `scipy.linear_sum_assignment` 交叉验证（见 `tests/test_hungarian.py`）。

## 7. 确定性保证

- 全局 seed 经 `core/seed.py` 统一派生各组件 RNG。
- 合成语料、模型初始化、留出切分全部由派生 RNG 驱动 → 同 seed 逐位一致。

## 8. 性能预算

- 3 数据集全流程约 24s（预算 60s）。
- 9 项单元测试全绿；ruff 0.16.10 硬门禁双绿；CI 矩阵 ubuntu + windows × py3.12 / 3.13。
