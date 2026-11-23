# rag-enterprise-agent — developer entrypoints
# Everything runs offline by default (--fake): no model downloads, no cloud.

PYTHON ?= python3
Q ?= What is the refund rule for event cancellation?

.PHONY: install test ingest ask serve eval prompt-ab benchmark agent-demo openapi docker-build lint clean

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

ingest:
	$(PYTHON) scripts/ingest.py --fake

ask:
	$(PYTHON) scripts/ask.py --fake "$(Q)"

serve:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

eval:
	$(PYTHON) rag/evaluation/run_eval.py --fake

prompt-ab:
	$(PYTHON) rag/evaluation/prompt_ab.py --fake

benchmark:
	$(PYTHON) scripts/benchmark.py --n 5000 --dim 64 --nlist 16 --probes 1 4 16

agent-demo:
	$(PYTHON) scripts/run_agent_demo.py

openapi:
	$(PYTHON) scripts/export_openapi.py

docker-build:
	docker build -t rag-enterprise-agent:latest .

lint:
	$(PYTHON) -m ruff check rag agent api observability azure_adapters scripts tests

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache indexes cache logs eval_out openapi.json
	find . -name "*.pyc" -delete
