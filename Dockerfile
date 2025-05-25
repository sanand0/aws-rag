FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

COPY app.py .

EXPOSE 8000
CMD ["uv", "run", "app.py"]
