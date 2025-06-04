# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "semantic-text-splitter",
#     "tqdm",
#     "fastapi",
#     "uvicorn",
# ]
# ///
"""Combined CLI tool and FastAPI app for document indexing and RAG search."""

from fastapi import FastAPI, Request
from pathlib import Path
from pydantic import BaseModel
from semantic_text_splitter import MarkdownSplitter, TextSplitter
from starlette import status
from starlette.responses import JSONResponse
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import argparse
import glob
import httpx
import json
import logging
import os
import sys
import traceback
import uvicorn

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Environment variables
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
APP_NAME = os.getenv("APP_NAME", "aws-rag")


# FastAPI models
class IndexRequest(BaseModel):
    """Request model for indexing documents."""

    paths: List[str]
    root: Optional[str] = None
    index: str = APP_NAME
    splitter: str = "md"
    max_chars: int = 1000
    model: str = "text-embedding-3-small"
    dimensions: int = 384


class SearchRequest(BaseModel):
    """Request model for searching documents."""

    query: str
    index: str = APP_NAME
    model: str = "text-embedding-3-small"
    dimensions: int = 384


class SearchResponse(BaseModel):
    """Response model for search results."""

    query: Dict[str, Any]
    references: List[Dict[str, Any]]
    answer: str


# FastAPI app
app = FastAPI(title="Document RAG API", description="Document indexing and search API")


@app.exception_handler(Exception)
async def server_error(request: Request, exc: Exception):
    """Handles all exceptions and returns a JSON response with a stack trace."""
    logger.exception("Internal server error")
    content = {
        "message": "Internal Server Error",
        "exception": str(exc),
        "endpoint": str(request.url),
        "stacktrace": traceback.format_exc(),
        "new": "new"
    }
    # If it's an HTTPX error, unpack the request details
    if isinstance(exc, (httpx.RequestError, httpx.HTTPStatusError)):
        req: httpx.Request = exc.request
        try:
            body = req.content.decode("utf-8")
        except Exception:
            body = repr(req.content)
        content["httpx_request"] = {
            "url": str(req.url),
            "headers": dict(req.headers),
            "body": body,
        }
    else:
        content["exception_type"] = type(exc).__name__
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content,
    )


def get_http_clients() -> tuple[httpx.Client, httpx.Client]:
    """Create HTTP clients for OpenSearch and OpenAI."""
    os_client = httpx.Client(auth=("admin", OPENSEARCH_PASSWORD), verify=False, timeout=30)
    openai_client = httpx.Client(headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}, timeout=30)
    return os_client, openai_client


def gather_files(paths: List[str]) -> List[Path]:
    """Expand globs and directories into file paths."""
    files = []
    for p in paths:
        if Path(p).is_dir():
            files += list(Path(p).rglob("*"))
            continue
        files += [Path(f) for f in glob.glob(p, recursive=True)]
    return files


def ensure_index(client: httpx.Client, name: str, dimensions: int) -> None:
    """Ensure the index exists with the correct mapping."""
    if client.head(f"{OPENSEARCH_ENDPOINT}/{name}").status_code == 200:
        return

    body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "filename": {"type": "text"},
                "chunk": {"type": "float"},
                "text_vector": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "engine": "faiss",
                        "name": "hnsw",
                        "space_type": "innerproduct",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }
    r = client.put(f"{OPENSEARCH_ENDPOINT}/{name}", json=body)
    r.raise_for_status()


def get_embeddings(
    client: httpx.Client, texts: List[str], model: str, dimensions: int
) -> List[List[float]]:
    """Get embeddings for a list of texts."""
    if not texts:
        return []

    response = client.post(
        f"{OPENAI_BASE_URL}/embeddings",
        json={"input": texts, "model": model, "dimensions": dimensions},
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


def transform_query(query: str, client: httpx.Client) -> Dict[str, Any]:
    """Transform query using GPT-4o-mini with structured JSON output."""
    system_prompt = """You are a dialogue-aware *Query Transformer*.
Write a JSON response for the last user query.
{
 "summary": string  // ≤2 sentences summarising the entire chat so far
 "rewrite": string  // ≤40 tokens, lowercase-NFC, UK→US spelling, append synonyms in ()
 "sub_q": list[string]  // list (≤3) atomic sub-questions or []
 "hyde_a": list[string] // list of 1-sentence hypothetical answers aligned to sub_q or []
 "route": string // "FACTOID"|"PROCEDURAL"|"OPINION"|"OTHER"
}
Rules:
- Preserve numerals / code / names verbatim.
- If route ≠ FACTOID|PROCEDURAL, set "hyde_a":[] .
- Output ONLY valid JSON—nothing else."""

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "rewrite": {"type": "string"},
            "sub_q": {"type": "array", "items": {"type": "string"}},
            "hyde_a": {"type": "array", "items": {"type": "string"}},
            "route": {"type": "string", "enum": ["FACTOID", "PROCEDURAL", "OPINION", "OTHER"]},
        },
        "required": ["summary", "rewrite", "sub_q", "hyde_a", "route"],
        "additionalProperties": False,
    }

    response = client.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "query_transform", "schema": schema},
            },
        },
    )
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


def ensure_pipeline_exists(client: httpx.Client) -> None:
    """Create RAG hybrid pipeline if it doesn't exist."""
    pipeline_config = {
        "description": "Normalise BM25 + k-NN for RAG",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {"technique": "arithmetic_mean"},
                }
            }
        ],
    }

    # Check if pipeline exists
    try:
        response = client.get(f"{OPENSEARCH_ENDPOINT}/_search/pipeline/rag-hybrid")
        if response.status_code == 200:
            return  # Pipeline already exists
    except httpx.HTTPError:
        pass

    # Create pipeline
    response = client.put(f"{OPENSEARCH_ENDPOINT}/_search/pipeline/rag-hybrid", json=pipeline_config)
    response.raise_for_status()


def build_hybrid_queries(
    transformed: Dict[str, Any], embeddings: List[List[float]]
) -> List[Dict[str, Any]]:
    """Build hybrid query list from transformed query and embeddings."""
    queries = []
    embed_idx = 0

    # Main rewrite query
    queries.append({"match": {"text": {"query": transformed["rewrite"]}}})
    if embed_idx < len(embeddings):
        queries.append({"knn": {"text_vector": {"vector": embeddings[embed_idx], "k": 50}}})
        embed_idx += 1

    # Sub-questions
    for sub_q in transformed["sub_q"]:
        queries.append({"match": {"text": {"query": sub_q}}})
        if embed_idx < len(embeddings):
            queries.append({"knn": {"text_vector": {"vector": embeddings[embed_idx], "k": 50}}})
            embed_idx += 1

    # Hypothetical answers
    for hyde_a in transformed["hyde_a"]:
        queries.append({"match": {"text": {"query": hyde_a}}})
        if embed_idx < len(embeddings):
            queries.append({"knn": {"text_vector": {"vector": embeddings[embed_idx], "k": 50}}})
            embed_idx += 1

    return queries


def search_opensearch(
    index: str, queries: List[Dict[str, Any]], client: httpx.Client
) -> List[Dict[str, Any]]:
    """Execute hybrid search against OpenSearch."""
    search_body = {
        "size": 15,
        "_source": {"exclude": ["content_vector"]},
        "query": {"hybrid": {"queries": queries}},
    }

    response = client.post(
        f"{OPENSEARCH_ENDPOINT}/{index}/_search?search_pipeline=rag-hybrid",
        json=search_body,
    )
    response.raise_for_status()
    return [hit["_source"] for hit in response.json()["hits"]["hits"]]


def generate_answer(
    query: str, transformed: Dict[str, Any], references: List[Dict[str, Any]], client: httpx.Client
) -> str:
    """Generate answer using GPT-4o-mini based on retrieved documents."""
    context = "\n\n".join(
        [
            f"Document {i + 1}:\n{ref.get('text', ref.get('content', str(ref)))}"
            for i, ref in enumerate(references[:10])  # Limit to top 10 for token efficiency
        ]
    )

    system_prompt = f"""You are a helpful assistant answering questions based on retrieved documents.
Query Analysis:
- Original query: {query}
- Rewritten query: {transformed["rewrite"]}
- Sub-questions: {transformed["sub_q"]}
- Query type: {transformed["route"]}
Instructions:
- Answer based ONLY on the provided documents
- If information is insufficient, state "Based on the available documents..."
- Be concise but comprehensive
- Cite specific details from the documents when relevant. Use [1] for the first document, etc."""

    response = client.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def index_documents(req: IndexRequest, show_progress: bool = True) -> Dict[str, Any]:
    """Index documents with vector embeddings."""
    files = gather_files(req.paths)
    if not files:
        return {"status": "error", "message": "No files found"}

    splitter_class = MarkdownSplitter if req.splitter == "md" else TextSplitter
    splitter = splitter_class(req.max_chars)
    root = Path(os.path.commonpath(files)) if req.root is None else Path(req.root)

    os_client, openai_client = get_http_clients()

    try:
        ensure_index(os_client, req.index, req.dimensions)
        processed_files = 0

        file_iter = tqdm(files, desc="Files", unit="file") if show_progress else files

        for file_path in file_iter:
            if not file_path.is_file():
                continue

            filename = str(file_path.relative_to(root))
            content = file_path.read_text(encoding="utf-8")
            chunks = splitter.chunks(content)

            # Get embeddings
            documents = []
            chunk_iter = (
                tqdm(list(enumerate(chunks)), desc=f"Chunks {filename}", leave=False)
                if show_progress
                else enumerate(chunks)
            )

            for i, chunk in chunk_iter:
                embeddings = get_embeddings(openai_client, [chunk], req.model, req.dimensions)
                documents.append(
                    {
                        "text": chunk,
                        "filename": filename,
                        "chunk": i,
                        "text_vector": embeddings[0],
                    }
                )

            # Delete existing documents
            response = os_client.post(
                f"{OPENSEARCH_ENDPOINT}/{req.index}/_delete_by_query",
                headers={"Content-Type": "application/json"},
                json={"query": {"term": {"filename": filename}}},
            )
            response.raise_for_status()

            # Upload documents
            response = os_client.post(
                f"{OPENSEARCH_ENDPOINT}/{req.index}/_bulk",
                headers={"Content-Type": "application/x-ndjson"},
                content="\n".join(
                    json.dumps({"index": {"_index": req.index}}) + "\n" + json.dumps(doc)
                    for doc in documents
                )
                + "\n",
            )
            response.raise_for_status()

            result = response.json()
            if result.get("errors"):
                return {"status": "error", "message": "Bulk indexing errors", "details": result}

            processed_files += 1

        return {"status": "success", "files_processed": processed_files}

    finally:
        os_client.close()
        openai_client.close()


def search_documents(req: SearchRequest) -> SearchResponse:
    """Search documents using hybrid RAG."""
    os_client, openai_client = get_http_clients()

    try:
        # Step 1: Transform query
        transformed = transform_query(req.query, openai_client)

        # Step 2: Get embeddings
        texts_to_embed = [transformed["rewrite"]] + transformed["sub_q"] + transformed["hyde_a"]
        embeddings = get_embeddings(openai_client, texts_to_embed, req.model, req.dimensions)

        # Step 3: Search OpenSearch
        ensure_pipeline_exists(os_client)
        queries = build_hybrid_queries(transformed, embeddings)
        references = search_opensearch(req.index, queries, os_client)

        # Step 4: Generate answer
        answer = generate_answer(req.query, transformed, references, openai_client)

        return SearchResponse(query=transformed, references=references, answer=answer)

    finally:
        os_client.close()
        openai_client.close()


# FastAPI endpoints
@app.post("/index")
async def api_index(request: IndexRequest) -> Dict[str, Any]:
    """Index documents via API."""
    return index_documents(request, show_progress=False)


@app.post("/search")
async def api_search(request: SearchRequest) -> SearchResponse:
    """Search documents via API."""
    return search_documents(request)


# CLI functions
def cli_index() -> None:
    """CLI entry point for indexing."""
    parser = argparse.ArgumentParser(description="Index documents with vector embeddings")
    parser.add_argument("paths", nargs="+", help="File paths, globs, or directories")
    parser.add_argument("--root", default=None, help="Root directory for relative paths")
    parser.add_argument("--index", default=APP_NAME, help="Vector index name")
    parser.add_argument("--splitter", choices=("md", "text"), default="md", help="Splitter type")
    parser.add_argument("--max-chars", type=int, default=1000, help="Max chunk size (chars)")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--dimensions", type=int, default=384, help="Embedding dimensions")
    args = parser.parse_args(sys.argv[2:])

    req = IndexRequest(
        paths=args.paths,
        root=args.root,
        index=args.index,
        splitter=args.splitter,
        max_chars=args.max_chars,
        model=args.model,
        dimensions=args.dimensions,
    )

    result = index_documents(req)
    print(json.dumps(result, indent=2))


def cli_search() -> None:
    """CLI entry point for searching."""
    parser = argparse.ArgumentParser(description="OpenSearch hybrid RAG query tool")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--index", default=APP_NAME, help="Vector index name")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--dimensions", type=int, default=384, help="Embedding dimensions")
    args = parser.parse_args(sys.argv[2:])

    req = SearchRequest(
        query=args.query, index=args.index, model=args.model, dimensions=args.dimensions
    )

    result = search_documents(req)
    response = {
        "query": result.query,
        "references": result.references,
        "answer": result.answer,
    }
    print(json.dumps(response, indent=2))


def cli_serve() -> None:
    """CLI entry point for serving FastAPI."""
    parser = argparse.ArgumentParser(description="Start FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args(sys.argv[2:])

    uvicorn.run(app, host=args.host, port=args.port)


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: uv run app.py <command> [args...]")
        print("Commands: serve, index, search")
        sys.exit(1)

    command = sys.argv[1]

    if command == "serve":
        cli_serve()
    elif command == "index":
        cli_index()
    elif command == "search":
        cli_search()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: serve, index, search")
        sys.exit(1)


if __name__ == "__main__":
    main()
