
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
	docker compose up --build --abort-on-container-exit test; docker compose logs test
