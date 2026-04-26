.PHONY: ingest run demo test clean

ingest:
	PYTHONPATH=. python src/ingest.py

run:
	PYTHONPATH=. python src/ingest.py
	PYTHONPATH=. python src/evaluate.py

demo:
	PYTHONPATH=. python src/demo.py

test:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	rm -rf data/chroma_db/ reports/figures/*.png reports/*.csv
	@echo "Cleaned."
