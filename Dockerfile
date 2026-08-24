# Multi-stage build: install deps in a builder layer, ship a slim runtime image.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src
COPY scripts ./scripts
COPY data/sample ./data/sample

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/models /app/results /app/docs && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "insurance_analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
