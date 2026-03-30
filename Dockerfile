FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application
COPY . .
RUN uv sync --frozen --no-dev

# psycopg binary (bundled libpq, no system dependency needed)
RUN uv pip install psycopg-binary

EXPOSE 2024

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD uv run python -c "import httpx; httpx.get('http://localhost:2024/ok').raise_for_status()"

CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--port", "2024", "--no-browser", "--no-reload"]
