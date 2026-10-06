# TopicForge 开发快捷命令
PY ?= python

.PHONY: help install test lint format demo bench ci clean

help:
	@echo "Targets: install test lint format demo bench ci clean"

install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

test:
	$(PY) -m pytest -q -W ignore::UserWarning

lint:
	ruff check .

format:
	ruff format --check .

demo:
	$(PY) examples/run_demo.py

bench:
	$(PY) cli.py run --out benchmark.json

ci: lint format test

clean:
	$(PY) -m pytest --cache-clear >/dev/null 2>&1 || true
	rm -rf .pytest_cache .ruff_cache __pycache__ ./*/__pycache__ 2>/dev/null || true
