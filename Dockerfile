FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY mcp_image_server/ ./mcp_image_server/

# Install dependencies
RUN uv sync --frozen --no-dev

# Create data directories
RUN mkdir -p /app/data/generated_images /app/data/cache

# Set environment variables
ENV PYTHONPATH="/app"
ENV UV_PROJECT_ENVIRONMENT="/app/.venv"

# Expose port (if needed for future HTTP interface)
# EXPOSE 8000

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'mcp_image_server'); import server; print('OK')" || exit 1

# Set working directory to server
WORKDIR /app/mcp_image_server

# Run the MCP server
CMD ["uv", "run", "python", "server.py"]