# format by ruff
fmt:
    poetry run ruff format .

# lint by ruff
lint:
    poetry run ruff check --fix .

# type-check by mypy
type-check:
    poetry run mypy .

# run pytest in tests directory
test:
    poetry run pytest tests

# upload to kaggle dataset
upload:
	poetry run python src/tools/upload_code.py
	poetry run python src/tools/upload_model.py

# run streamlit app
streamlit:
	poetry run streamlit run visualizer.py --server.address 0.0.0.0
