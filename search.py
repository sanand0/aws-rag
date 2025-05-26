# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
# ]
# ///
"""CLI tool for OpenSearch hybrid RAG querying with GPT-4 query transformation."""

from typing import Dict, List, Any
import argparse
import httpx
import json
import os


def transform_query(query: str, openai_key: str) -> Dict[str, Any]:
    """Transform query using GPT-4.1-mini with structured JSON output."""
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

    with httpx.Client() as client:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_key}"},
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


def get_embeddings(
    texts: List[str], model: str, dimensions: int, openai_key: str
) -> List[List[float]]:
    """Get embeddings for multiple texts in single API call."""
    if not texts:
        return []

    with httpx.Client() as client:
        response = client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_key}"},
            json={"input": texts, "model": model, "dimensions": dimensions},
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]


def ensure_pipeline_exists(os_endpoint: str, os_password: str) -> None:
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

    with httpx.Client(verify=False) as client:
        # Check if pipeline exists
        try:
            response = client.get(
                f"{os_endpoint}/_search/pipeline/rag-hybrid", auth=("admin", os_password)
            )
            if response.status_code == 200:
                return  # Pipeline already exists
        except httpx.HTTPError:
            pass

        # Create pipeline
        response = client.put(
            f"{os_endpoint}/_search/pipeline/rag-hybrid",
            auth=("admin", os_password),
            json=pipeline_config,
        )
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
    index: str, queries: List[Dict[str, Any]], os_endpoint: str, os_password: str
) -> List[Dict[str, Any]]:
    """Execute hybrid search against OpenSearch."""
    search_body = {
        "size": 15,
        "_source": {"exclude": ["content_vector"]},
        "query": {"hybrid": {"queries": queries}},
    }

    with httpx.Client(verify=False) as client:
        response = client.post(
            f"{os_endpoint}/{index}/_search?search_pipeline=rag-hybrid",
            auth=("admin", os_password),
            json=search_body,
        )
        response.raise_for_status()
        return [hit["_source"] for hit in response.json()["hits"]["hits"]]


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="OpenSearch hybrid RAG query tool")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--index", default="aws-rag", help="Vector index name")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--dimensions", type=int, default=384, help="Embedding dimensions")

    args = parser.parse_args()

    # Get environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    os_endpoint = os.getenv("OS_ENDPOINT")
    os_password = os.getenv("TF_VAR_opensearch_password")

    # Step 1: Transform query
    transformed = transform_query(args.query, openai_key)
    print(f"Query transformed: {transformed}")

    # Step 2: Get embeddings
    texts_to_embed = [transformed["rewrite"]] + transformed["sub_q"] + transformed["hyde_a"]
    embeddings = get_embeddings(texts_to_embed, args.model, args.dimensions, openai_key)
    print(f"Generated {len(embeddings)} embeddings")

    # Step 3: Search OpenSearch
    ensure_pipeline_exists(os_endpoint, os_password)
    queries = build_hybrid_queries(transformed, embeddings)
    results = search_opensearch(args.index, queries, os_endpoint, os_password)

    # Output results
    print(f"\nFound {len(results)} results:")
    for i, hit in enumerate(results, 1):
        print(f"{i}. {hit['filename']} (chunk {hit['chunk']}): {hit['text']}")


if __name__ == "__main__":
    main()
