# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "semantic-text-splitter",
#     "tqdm",
# ]
# ///
from pathlib import Path
from semantic_text_splitter import MarkdownSplitter, TextSplitter
from tqdm import tqdm
from typing import List
import argparse
import glob
import httpx
import json
import os


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Index documents with vector embeddings")
    parser.add_argument("paths", nargs="+", help="File paths, globs, or directories")
    parser.add_argument("--root", default=None, help="Root directory for relative paths")
    parser.add_argument("--index", default="aws-rag", help="Vector index name")
    parser.add_argument("--splitter", choices=("md", "text"), default="md", help="Splitter type")
    parser.add_argument("--max-chars", type=int, default=1000, help="Max chunk size (chars)")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--dimensions", type=int, default=384, help="Embedding dimensions")
    return parser.parse_args()


def gather_files(paths: List[str]) -> List[Path]:
    """Expand globs and directories into file paths."""
    files = []
    for p in paths:
        if Path(p).is_dir():
            files += list(Path(p).rglob("*"))
            continue
        files += [Path(f) for f in glob.glob(p, recursive=True)]
    return files


def ensure_index(client: httpx.Client, endpoint: str, name: str, dimensions: int) -> None:
    """Ensure the index exists with the correct mapping."""
    if client.head(f"{endpoint}/{name}").status_code == 200:
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
    r = client.put(f"{endpoint}/{name}", json=body)
    r.raise_for_status()


def get_embeddings(
    client: httpx.Client, texts: List[str], model: str, dimensions: int
) -> List[List[float]]:
    """Get embeddings for a list of texts."""
    response = client.post(
        "https://api.openai.com/v1/embeddings",
        json={"input": texts, "model": model, "dimensions": dimensions},
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


def main():
    args = parse_args()
    files = gather_files(args.paths)
    splitter_class = MarkdownSplitter if args.splitter == "md" else TextSplitter
    splitter = splitter_class(args.max_chars)
    root = Path(os.path.commonpath(files)) if args.root is None else Path(args.root)

    endpoint = os.getenv("OS_ENDPOINT")
    http_auth = ("admin", os.getenv("TF_VAR_opensearch_password"))
    openai_headers = {"Authorization": f"Bearer {os.getenv('TF_VAR_openai_api_key')}"}
    with (
        # Disable SSL verification for local OpenSearch's self-signed certificate
        httpx.Client(auth=http_auth, verify=False, timeout=30) as os_httpx,
        httpx.Client(headers=openai_headers, timeout=30) as openai_httpx,
    ):
        ensure_index(os_httpx, endpoint, args.index, args.dimensions)

        for file_path in tqdm(files, desc="Files", unit="file"):
            filename = str(file_path.relative_to(root))
            content = file_path.read_text(encoding="utf-8")
            chunks = splitter.chunks(content)

            # Get embeddings
            documents = []
            for i, chunk in tqdm(list(enumerate(chunks)), desc=f"Chunks {filename}", leave=False):
                embeddings = get_embeddings(openai_httpx, [chunk], args.model, args.dimensions)
                documents.append(
                    {
                        "text": chunk,
                        "filename": filename,
                        "chunk": i,
                        "text_vector": embeddings[0],
                    }
                )

            # Delete existing documents
            response = os_httpx.post(
                f"{endpoint}/{args.index}/_delete_by_query",
                headers={"Content-Type": "application/json"},
                json={"query": {"term": {"filename": filename}}},
            )
            response.raise_for_status()

            # Upload documents
            response = os_httpx.post(
                f"{endpoint}/{args.index}/_bulk",
                headers={"Content-Type": "application/x-ndjson"},
                content="\n".join(
                    json.dumps({"index": {"_index": args.index}}) + "\n" + json.dumps(doc)
                    for doc in documents
                )
                + "\n",
            )
            response.raise_for_status()
            result = response.json()
            if result.get("errors"):
                print(result)


if __name__ == "__main__":
    main()
