.PHONY: install backfill train demo dashboard test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

backfill:
	python scripts/run_backfill.py --lookback-days 180

train:
	python scripts/run_training_pipeline.py

feature-update:
	python scripts/run_feature_pipeline.py

demo:
	python scripts/run_full_demo.py

dashboard:
	streamlit run app/streamlit_app.py

test:
	pytest -v

clean:
	find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info src/*.egg-info
