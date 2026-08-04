FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# FAST: Grab the pre-compiled uv binary instantly
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

# ULTRA-FAST: Leverage Docker caching for uv downloads
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src

# RUN WITH UV: Let uv manage the environment execution
CMD ["uv", "run", "python", "-m", "digital_twin.main"]
