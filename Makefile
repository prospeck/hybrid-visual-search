.PHONY: install install-ml test import index benchmark api frontend up up-ml down

install:
	uv sync --extra dev --extra frontend

install-ml:
	uv sync --extra dev --extra frontend --extra ml

test:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
	uv run pytest --cov=visual_search --cov-fail-under=80

import:
	uv run visual-search import-zip "$(ZIP)" --limit 30

index:
	uv run visual-search index runtime/source-images

benchmark:
	uv run visual-search benchmark --output reports/benchmark.json

api:
	uv run uvicorn visual_search.api:app --reload --port 8020

frontend:
	VISUAL_SEARCH_API_URL=http://localhost:8020 uv run streamlit run frontend/app.py --server.port 8504

up:
	docker compose up --build -d

up-ml:
	docker compose -f docker-compose.yml -f docker-compose.ml.yml up --build -d

down:
	docker compose down

