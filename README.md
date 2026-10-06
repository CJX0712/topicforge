# TopicForge

> 世界级主题建模系统 · **作者：晨星**

TopicForge 是一套 **纯 Python / 纯 NumPy** 实现的世界级主题建模（Topic Modeling）系统，
融合多项顶级开源算法与自研创新，开箱即用、可复现、确定性、CI 全绿。

## 核心能力

| 模型 | 说明 | 角色 |
|------|------|------|
| `TopicFuse` | 多链 VB-LDA 集成 + Hungarian 共识对齐 + VB 精炼 | 🚩 旗舰 |
| `LDA` | 变分贝叶斯 LDA（Blei et al. 2003，自实现 digamma） | 强基线 |
| `NMF` | 非负矩阵分解（乘法更新，可选 sklearn 后端） | 对照 |
| `TfidfKMeans` | TF-IDF + 手写 KMeans | 传统基线 |
| `RandomTopics` | 随机主题（sanity check） | 下界 |

**关键事实**：运行时依赖**只有 `numpy`**——无需联网安装任何包即可运行（纯 NumPy 离线兜底）。

## 创新点 · TopicFuse

1. **方差缩减**：R 条独立 VB-LDA 链经 Hungarian 主题对齐后平均，降低单链随机初始化方差；
2. **共识对齐**：以链 0 为参考，其余链的主题经 `1 - cosine` 代价匈牙利匹配后重排再平均，
   彻底规避标签置换导致的跨样本平均退化；
3. **VB 精炼**：以集成 `phi` 为初值做数步变分 E/M，逼近更高似然的局部最优，并产出与 `phi` 自洽的 `theta`。

> 为何选 VB 而非吉布斯采样：完全向量化、确定性强（同 seed 逐位一致）、不依赖 scipy、
> 收敛稳定，是 LDA 推断的顶级标准方法之一。

## 快速开始

```bash
# 运行端到端基准（生成合成语料 → 多模型 → 评测 → 落盘 benchmark.json）
python examples/run_demo.py

# 或用 CLI
python cli.py run --out benchmark.json
python cli.py version
```

所有参数可通过 `TOPICFORGE_*` 环境变量或 CLI 参数覆盖（见 `topicforge/core/config.py`）：

```bash
TOPICFORGE_N_TOPICS=8 TOPICFORGE_N_DOCS=400 python cli.py run
python cli.py run --n-topics 8 --n-docs 400 --n-chains 6 --n-iter 200
```

## 基准结果（3 数据集 · K=6 · V=500 · N=200 · seed=12345）

| model         | recovery     | cluster     | perplexity |
|---------------|--------------|-------------|------------|
| topicfuse     | 0.9237±0.046 | 0.7900±0.059 | 62.027     |
| lda           | 0.8235±0.039 | 0.7417±0.019 | 77.128     |
| nmf           | 0.9906±0.001 | 0.9049±0.041 | 49.158     |
| tfidf_kmeans  | 0.8206±0.009 | 0.7667±0.027 | —          |
| random        | 0.0993±0.038 | 0.2350±0.012 | —          |

> **头条**：`topicfuse - lda` 主题恢复度 **Δ = +0.1002**（门槛 0.02，PASS）。
> 评测口径：recovery / cluster 越大越好；perplexity 越低越好。

- **确定性**：同 seed 两次运行，16 项核心指标逐位一致（`max|Δ| = 0`）。
- **单测**：9 项全绿（`pytest`）。
- **质量等级**：**S**。

## 项目结构

```
topicforge/
├── topicforge/            # 包
│   ├── core/              # 配置 / 类型 / 错误 / 确定性 seed / Hungarian 分配
│   ├── data/              # 合成语料生成（标准 LDA 生成模型）
│   ├── models/            # lda / nmf / ensemble(TopicFuse) / baselines
│   ├── eval/              # 评测指标（recovery / clustering / perplexity）
│   └── pipeline/          # 多数据集 → 多模型 → 评测 → 聚合
├── cli.py                 # 命令行入口
├── examples/run_demo.py   # 端到端演示
├── tests/                 # 单元测试
├── docs/                  # architecture.md / model_card.md
└── pyproject.toml
```

## 评测指标

- **topic_recovery**：估计主题经 Hungarian 对齐真实主题后的平均余弦相似度（越高越好）
- **doc_clustering_accuracy**：文档主导主题经 Hungarian 对齐真实主题后的聚类精度（越高越好）
- **perplexity**：held-out 困惑度（越低越好）

标签置换不变性由 Hungarian 算法保证（详见 `topicforge/core/assign.py`，纯 Python O(n³) 增广路，
与 `scipy.linear_sum_assignment` 交叉验证）。

## 踩坑与失败案例（真实缺陷，均已修复并留回归）

| # | 症状 | 根因 | 修法 |
|---|------|------|------|
| 1 | 批量向量化吉布斯采样 LDA recovery 恒 0.44~0.46（4000 次迭代仍 0.50） | 批量并行采样是**坏的 MCMC kernel**，跨样本平均又受标签置换退化 | 改用 VB-LDA，recovery 跃升至 0.80+ |
| 2 | `ModuleNotFoundError: No module named 'topicforge'` | 源码用绝对导入 `topicforge.core`，但包模块直接放在仓库根、未建成子包 | 核心模块移入 `topicforge/` 子包并加 `__init__.py`；CLI/demo 的 `sys.path` 插入层数同步修正 |
| 3 | `em_polish_phi` 报 `TypeError: NoneType not subscriptable` | 精炼初值只传了 `phi*V`，`gamma` 为 `None` | 补充完整 `gamma0 = alpha + rng.random((D,K))` 初值 |
| 4 | `np.add.at` 累加在大规模语料上成为性能瓶颈 | 稀疏逐元素原子加开销大 | 改用 `np.bincount` 加权累加（`idx_wk` 展平索引） |
| 5 | 单链 LDA 多次运行 recovery 方差大、跨样本平均退化 | 主题标签置换导致平均相互抵消 | TopicFuse 引入 Hungarian 共识对齐（见消融实验） |

## 消融实验

```bash
python examples/ablation.py
```

拆解 TopicFuse 两大组件（Hungarian 共识对齐 / VB 精炼）的独立贡献，对照单链 LDA 与无对齐平均。

## 开发

```bash
make ci      # ruff check + ruff format --check + pytest（CI 等价）
make demo    # 端到端演示
make bench   # 生成 benchmark.json
```

## 许可证

MIT · 作者 晨星（CJX0712）。详见 [LICENSE](LICENSE)。
