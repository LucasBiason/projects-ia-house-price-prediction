
runapp:
	docker compose up --build -d;

runapp-dev:
	docker compose down;
	docker compose up --build; 

lint:
	isort **/*.py
	python3 -m black app/* 
	flake8 app/*

test:
	poetry run python -m pytest tests/ -v

test-with-coverage:
	poetry run coverage run -m pytest --cov=app --cov-report term-missing --cov-config=.coveragerc tests/ -s -vv
