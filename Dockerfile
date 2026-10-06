# TopicForge — 纯 NumPy 主题建模系统
# 运行时仅需 numpy；CI / 开发额外装 pytest + ruff。
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 锁版本优先，失败回退到 requirements.txt
COPY requirements.lock.txt requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt -r requirements-dev.txt || \
    pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . .

# 默认跑 CI 等价检查（ruff 双绿 + pytest），可直接 `docker build` 验证
RUN ruff check . && ruff format --check . && pytest -q

CMD ["python", "examples/run_demo.py"]
