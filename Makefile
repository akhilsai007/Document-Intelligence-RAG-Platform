.PHONY: install seed ingest train run test docker up down lint

install:
	pip install --break-system-packages -r requirements.txt

seed:
	python scripts/seed_data.py

ingest:
	python scripts/ingest.py --docs data/sample_docs

train:
	python scripts/train_router.py

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

docker:
	docker build -t rag-doc-intelligence:latest .

up:
	docker compose up --build

down:
	docker compose down
