"""配置：环境变量 TOPICFORGE_* 覆盖 + schema 校验。"""

import os
from dataclasses import dataclass

from .errors import ConfigError


@dataclass
class TopicForgeConfig:
    seed: int = 12345
    n_topics: int = 6
    vocab_size: int = 500
    n_docs: int = 200
    avg_doc_len: int = 50
    alpha: float = 0.1  # 文档-主题 Dirichlet 先验
    eta: float = 0.01  # 主题-词 Dirichlet 先验
    n_chains: int = 4  # TopicFuse 集成链数
    n_iter: int = 150  # VB-LDA 迭代次数
    burnin: int = 60  # 兼容字段（VB 不使用）
    holdout_frac: float = 0.2  # held-out  perplexity 留出比例

    @classmethod
    def from_env(cls, **overrides):
        cfg = cls()
        for key in list(cfg.__dataclass_fields__):  # type: ignore[attr-defined]
            envk = "TOPICFORGE_" + key.upper()
            if envk in os.environ:
                raw = os.environ[envk]
                cur = getattr(cfg, key)
                try:
                    if isinstance(cur, bool):
                        setattr(cfg, key, raw.lower() in ("1", "true", "yes"))
                    else:
                        setattr(cfg, key, type(cur)(raw))
                except (ValueError, TypeError):
                    raise ConfigError(f"无法解析环境变量 {envk}={raw!r} 为 {type(cur).__name__}")
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self):
        if self.n_topics < 1:
            raise ConfigError("n_topics 必须 >= 1")
        if self.vocab_size < 2:
            raise ConfigError("vocab_size 必须 >= 2")
        if self.n_docs < 1:
            raise ConfigError("n_docs 必须 >= 1")
        if not (0.0 < self.holdout_frac < 1.0):
            raise ConfigError("holdout_frac 必须在 (0,1)")
        if self.n_iter <= 0 or self.burnin < 0:
            raise ConfigError("n_iter/burnin 配置非法")
        if self.alpha <= 0 or self.eta <= 0:
            raise ConfigError("alpha/eta 必须 > 0")
