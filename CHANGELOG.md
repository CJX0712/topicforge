# Changelog

All notable changes to TopicForge are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-10-07

### Added
- 变分贝叶斯 LDA（Blei et al. 2003）纯 NumPy 实现，自实现 digamma，完全向量化、确定性。
- NMF 非负矩阵分解（乘法更新，可选 sklearn 后端）。
- **TopicFuse 旗舰**：多链 VB-LDA 集成 + Hungarian 共识对齐 + VB 精炼。
- 基线：TF-IDF + KMeans、随机主题（sanity check / 下界）。
- 评测指标：topic_recovery（Hungarian 对齐余弦）/ doc_clustering_accuracy / held-out perplexity。
- 纯 Python O(n³) Hungarian 分配算法，与 scipy 交叉验证。
- 确定性入口 `core/seed.py`；合成语料生成器（标准 LDA 生成模型，逐位可复现）。
- 多数据集基准 pipeline、CLI（`cli.py`）、端到端演示（`examples/run_demo.py`）。
- 9 项单元测试；CI 矩阵（ubuntu + windows × py3.12 / py3.13）；ruff 0.16.10 硬门禁双绿。
- 纯 NumPy 运行时依赖，离线兜底（无需联网即可运行）。

### Performance
- 3 数据集 K=6 V=500 N=200：topicfuse recovery 0.9237±0.046 / lda 0.8235±0.039（Δ=+0.1002 ≥ 0.02 PASS）。
- 同 seed 两次运行 16 项核心指标逐位一致（max|Δ|=0）。
- 端到端 demo 约 24s（预算 60s）。
- 质量等级 **S**。
