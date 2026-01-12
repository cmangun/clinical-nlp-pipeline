# Clinical Nlp Pipeline - Production Container
# Multi-stage build for optimized image size

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Production image
FROM python:3.11-slim as runtime

WORKDIR /app

# Security: Run as non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Switch to non-root user
USER appuser

# Default command (override in docker-compose or k8s)
CMD ["python", "-m", "src.api.main"]

# Labels for container registry
LABEL org.opencontainers.image.title="Clinical Nlp Pipeline" \
      org.opencontainers.image.description="Production Clinical Nlp Pipeline service" \
      org.opencontainers.image.vendor="Christopher Mangun" \
      org.opencontainers.image.licenses="MIT"
