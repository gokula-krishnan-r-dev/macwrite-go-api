# AI API Proxy Service

A high-performance, scalable Go API service that acts as a proxy for AI services.

## Features

- Scalable, production-ready Go API service
- Docker support for easy deployment
- Hot reloading for rapid development
- Configuration via environment variables
- Graceful shutdown support
- RESTful API design

## Prerequisites

- Go 1.21+
- Docker and Docker Compose (for containerized development)

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/gokul/ai-core.git
cd ai-core

# Install dependencies
make setup

# Run with hot reloading
make dev
```

### Docker Development

```bash
# Run with docker-compose (includes hot reloading)
make docker-dev
```

### Production Build

```bash
# Build docker image
make docker-build

# Run in Docker
make docker-run
```

## API Endpoints

### Health Check

```
GET /health
```

Returns the service health status.

### AI Generation

```
POST /api/v1/ai/generate
```

Request body:
```json
{
  "prompt": "Your input text here",
  "model": "optional-model-name"
}
```

## Configuration

Configuration is managed through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| SERVER_PORT | Server port | 8080 |
| SERVER_SHUTDOWN_TIMEOUT | Graceful shutdown timeout (seconds) | 5 |
| AI_ENDPOINT | AI service endpoint URL | https://auto-comment.gokulakrishnanr812-492.workers.dev/ |
| AI_TIMEOUT | AI request timeout (seconds) | 30 |
| AI_DEFAULT_MODEL | Default AI model if not specified | default |
| LOG_LEVEL | Logging level | info |
| GO_ENV | Environment (development/production) | development |

## Development

```bash
# Run tests
make test

# Run linters
make lint

# Clean build artifacts
make clean
```

## License

[MIT License](LICENSE) # macwrite-go-api
# macwrite-go-api
