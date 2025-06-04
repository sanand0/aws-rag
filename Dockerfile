FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

# Copy app.py
COPY app.py .

EXPOSE 8000
CMD ["uv", "run", "app.py", "serve"]
