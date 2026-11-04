# Runs the feature pipeline. Dependencies come from the committed uv.lock.
FROM python:3.11.18-slim-bookworm

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv==0.13.7

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-cache

RUN useradd --create-home pipeline
USER pipeline

# Pass keys at run time, for example: docker run --env-file .env neghour
CMD [".venv/bin/python", "-m", "neghour.pipelines.feature_pipeline"]
