"""核心数据类型。"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Corpus:
    """一个文本语料（以词索引 token 表示）。

    - docs: 每个元素是一篇文档的词索引数组 (int)
    - vocab_size: 词表大小 V
    - true_beta: (K, V) 真实主题-词分布（合成数据有；真实数据为 None）
    - true_theta: (N, K) 真实文档-主题混合
    - true_topic: (N,) 每篇文档的主导真实主题
    - n_topics: 真实主题数 K（已知域才有）
    """

    docs: list
    vocab_size: int
    true_beta: np.ndarray | None = None
    true_theta: np.ndarray | None = None
    true_topic: np.ndarray | None = None
    n_topics: int | None = None


@dataclass
class TopicModelResult:
    """单个主题模型的产物。

    - phi: (K, V) 主题-词分布（每行已归一化到和为 1）
    - theta: (N, K) 文档-主题分布
    - doc_topic_label: (N,) 每篇文档主导主题（argmax theta）
    - name: 模型名
    """

    phi: np.ndarray
    theta: np.ndarray
    doc_topic_label: np.ndarray
    name: str
    extra: dict = field(default_factory=dict)
