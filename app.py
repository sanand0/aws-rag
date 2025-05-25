# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "httpx",
#     "uvicorn",
# ]
# ///
from fastapi import FastAPI
import os
import httpx

app = FastAPI()


@app.get("/")
def models():
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get("https://api.openai.com/v1/models", headers=headers)
    return resp.json()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
