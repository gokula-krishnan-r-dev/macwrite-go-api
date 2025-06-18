.PHONY: build run dev clean docker-build docker-run docker-dev test lint

# Go binary
BINARY_NAME=api-server
MAIN_PATH=./api

# Development variables
DEV_DOCKER_COMPOSE=docker-compose.yml

# Production variables
DOCKER_IMAGE=gokul/ai-api:latest

build:
	go build -o $(BINARY_NAME) $(MAIN_PATH)

run: build
	./$(BINARY_NAME)

dev:
	air -c .air.toml

clean:
	rm -f $(BINARY_NAME)
	rm -rf ./tmp

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run: docker-build
	docker run -p 8080:8080 $(DOCKER_IMAGE)

docker-dev:
	docker-compose -f $(DEV_DOCKER_COMPOSE) up --build

test:
	go test -v ./...

lint:
	go vet ./...

setup:
	go mod tidy
	go install github.com/cosmtrek/air@latest

help:
	@echo "Available commands:"
	@echo "  make build        - Build the Go binary"
	@echo "  make run          - Build and run the Go binary"
	@echo "  make dev          - Run with hot reloading using Air"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run the application in Docker"
	@echo "  make docker-dev   - Run with hot reloading in Docker"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make setup        - Set up development tools" 