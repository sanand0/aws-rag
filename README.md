# Retrieval-Augmented Generation on AWS

A unified command-line tool and FastAPI application for document indexing and retrieval-augmented generation (RAG) search using OpenSearch and OpenAI embeddings.

## Learnings

- AWS offers [OpenSearch](https://opensearch.org/), their open-source fork of [ElasticSearch](https://www.elastic.co/elasticsearch).
  They don't update ElasticSearch versions, so it's a security risk.
  OpenSearch _was_ ElasticSearch compatible but RAG features like ML pipelines, embeddings, etc. have diverged considerabl. [ChatGPT](https://chatgpt.com/share/6833e9e0-a954-800c-b5c8-89f465cade33)
- [OpenSearch integrates with ML models](https://docs.opensearch.org/docs/latest/ml-commons-plugin/integrating-ml-models/).
  But pass your own embeddings because:
  - **Models are subpar**. The pretrained models have poor quality and running custom models requires higher resources, potentially GPUs
  - **Integration is complex**.v API integration setup is complex and brittle
  - **... and opaque**. When something fails, you have little or no visibility into what went wrong
- Use these settings for vector indexes (see [k-NN Index docs](https://docs.opensearch.org/docs/2.1/search-plugins/knn/knn-index)):
  - `settings.index.knn: true`: Enables k-NN plugin, required for vector/hybrid search
  - `mappings.properties.text.type: "text"`: Applies full-text analysis (tokenization, lowercasing) for keyword search
  - `mappings.properties.text_embedding.type: "knn_vector"`: Defines a field for storing vectors for k-NN search
  - `mappings.properties.text_embedding.dimension: 384`: # of vector dimensions. Large models provide 1,536 dimensions, but the first few dimensions are the most important. It's OK to truncate
  - `mappings.properties.text_embedding.engine: "faiss"`: Fastest engine at scale. Alternatives: `lucene` (slow), `nmslib` (deprecated). See [Engines](https://docs.opensearch.org/docs/latest/field-types/supported-field-types/knn-methods-engines/)
  - `mappings.properties.text_embedding.name: "hnsw"`: Better quality than `ivf` and just as fast.
  - `mappings.properties.text_embedding.space_type: "innerproduct"`: Can also use `l2` but `innerproduct` is closer to cosine similarity.
  - `mappings.properties.text_embedding.parameters.ef_construction: 128`: Size of k-NN list. High values are more accurate but slower
  - `mappings.properties.text_embedding.parameters.m: 24`: # of bidirectional links. More links are better but take up MUCH more memory.
- Use hybrid search: BM25 for keyword search, k-NN for vector search, to get the best of both worlds
- Use query rewriting to improve retrieval #TODO
- Use citations as [1], [2], etc. This is consistent with OpenAI, Bing/Copilot, Perplexity, and DeepSeek. (Claude and Gemini use a different format.)

## Setup

You need

- [uv](https://docs.astral.sh/uv/)
- [docker](https://www.docker.com/) or [podman](https://podman.io/)
- OpenAI API access

```bash
# Clone the repo
git clone https://github.com/sanand0/aws-rag.git

# Set environment variables
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional, defaults to OpenAI
export OPENAI_API_KEY="your-openai-api-key"
export OPENSEARCH_ENDPOINT="https://localhost:9200"  # Or whenever your OpenSearch cluster is running
export OPENSEARCH_PASSWORD="your-opensearch-password"
export APP_NAME="aws-rag"

# Run a single-node opensearch cluster
docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=$OPENSEARCH_PASSWORD" opensearchproject/opensearch:2
# Test OpenSearch
curl -ku admin:$OPENSEARCH_PASSWORD $OPENSEARCH_ENDPOINT

# Install with uv (recommended)
uv run app.py --help
```

## Command Line Usage

```bash
# Index all files in a directory
uv run app.py index docs/

# Search your documents
uv run app.py search "How do I configure authentication?"
```

### API Server Usage

```bash
# Start FastAPI server
uv run app.py serve

# Index via API
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"paths": ["docs/"]}'

# Search via API
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication setup"}'
```

## CLI Reference

```bash
# Index command
uv run app.py index docs/ config/ *.md \
  --root /project/root \
  --index production-docs \
  --splitter text \
  --max-chars 1500 \
  --model text-embedding-3-large \
  --dimensions 1536
```

**Parameters:**

- `paths`: File paths, globs, or directories (supports multiple)
- `--root`: Root directory for relative paths (default: common path)
- `--index`: Vector index name (default: "aws-rag")
- `--splitter`: Text splitter type - "md" or "text" (default: "md")
- `--max-chars`: Maximum chunk size in characters (default: 1000)
- `--model`: OpenAI embedding model (default: "text-embedding-3-small")
- `--dimensions`: Embedding dimensions (default: 384)

```bash
# Search command
uv run app.py search "How to configure SSL certificates?" \
  --index production-docs \
  --model text-embedding-3-large \
  --dimensions 1536
```

**Parameters:**

- `query`: Search query string
- `--index`: Vector index name (default: "aws-rag")
- `--model`: OpenAI embedding model (default: "text-embedding-3-small")
- `--dimensions`: Embedding dimensions (default: 384)

```bash
# Serve command
uv run app.py serve
```

**Parameters:**

- `--host`: Host to bind to (default: "127.0.0.1")
- `--port`: Port to bind to (default: 8000)
  uv run app.py serve --host 0.0.0.0 --port 8080

```

**Parameters:**
- `--host`: Host to bind to (default: "127.0.0.1")
- `--port`: Port to bind to (default: 8000)
```

## API Reference

```bash
# Index documents into the vector database.
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{"paths": ["docs/"]}'

# Complete Request
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "paths": ["docs/", "config/", "*.md"],
    "root": "/project/root",
    "index": "production-docs",
    "splitter": "text",
    "max_chars": 1500,
    "model": "text-embedding-3-large",
    "dimensions": 1536
  }'
```

**Response:**

```json
{
  "status": "success",
  "files_processed": 42
}
```

```bash
# Search documents using hybrid RAG.
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "How to configure authentication?"}'

# Complete request
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to configure SSL certificates in production?",
    "index": "production-docs",
    "model": "text-embedding-3-large",
    "dimensions": 1536
  }'
```

**Response:**

```json
{
  "query": {
    "summary": "User asking about SSL certificate configuration",
    "rewrite": "ssl certificate configuration production setup",
    "sub_q": ["How to generate SSL certificates?", "Where to install certificates?"],
    "hyde_a": ["SSL certificates are configured in the web server.", "Certificate files are placed in /etc/ssl/."],
    "route": "PROCEDURAL"
  },
  "references": [
    {
      "text": "To configure SSL certificates...",
      "filename": "ssl-guide.md",
      "chunk": 0
    }
  ],
  "answer": "Based on the available documents, SSL certificates are configured by..."
}
```

## License

[MIT](LICENSE)

### Deploy on AWS

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...
export TF_VAR_openai_base_url="$OPENAI_BASE_URL"
export TF_VAR_openai_api_key="$OPENAI_API_KEY"
export TF_VAR_opensearch_password="$OPENSEARCH_PASSWORD"
export TF_VAR_app_name="$APP_NAME"

# Install tofu (one-time)
sudo snap install tofu --classic

# Deploy
tofu init && tofu apply --auto-approve
```

Test via:

```bash
# Check the AWS runner endpoint
curl "https://$(tofu output -raw service_url)/"

# Check the OpenSearch endpoint
curl -u admin:$OPENSEARCH_PASSWORD "https://$(tofu output -raw opensearch_endpoint)/_cluster/health?pretty"
```

To destroy, use:

```bash
tofu destroy --auto-approve
```
