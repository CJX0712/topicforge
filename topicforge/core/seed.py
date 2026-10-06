"""全局确定性入口：所有随机源一次性设齐。"""

import numpy as np

_CURRENT = {"seed": 12345}


def set_all(seed: int = 12345) -> None:
    """设齐 numpy 全局种子，并记录当前 seed 供 new_rng 使用。"""
    _CURRENT["seed"] = int(seed)
    np.random.seed(int(seed))


def get_seed() -> int:
    return _CURRENT["seed"]


def new_rng(seed=None):
    """返回独立 Generator。不传 seed 时用全局 seed，保证两次同 seed 运行逐位一致。"""
    s = int(seed) if seed is not None else _CURRENT["seed"]
    return np.random.default_rng(s)
