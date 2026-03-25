up:
	docker compose up -d postgres redis backend frontend

up-ml:
	docker compose --profile ml up -d

up-ops:
	docker compose --profile ops up -d

down:
	docker compose down

logs:
	docker compose logs -f backend frontend
