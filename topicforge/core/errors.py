"""错误体系 E100~E500。"""


class TopicForgeError(Exception):
    """所有 TopicForge 异常的基类。"""


class ConfigError(TopicForgeError):
    """E100 配置/参数错误。"""


class DataError(TopicForgeError):
    """E200 数据生成/载入错误。"""


class ModelError(TopicForgeError):
    """E300 模型训练/推断错误。"""


class EvalError(TopicForgeError):
    """E400 评测指标错误。"""


class PipelineError(TopicForgeError):
    """E500 流水线编排错误。"""
