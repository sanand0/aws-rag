# Retrieval-Augmented Generation on AWS

Learnings:

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

## Setup

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=...
export TF_VAR_app_name=aws-rag
export TF_VAR_openai_api_key=...
export TF_VAR_opensearch_password=...

# Run OpenSearch 2.x as a daemon
docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=$TF_VAR_opensearch_password" opensearchproject/opensearch:2
export OS_ENDPOINT=https://localhost:9200

# Test OpenSearch
curl -ku admin:$TF_VAR_opensearch_password $OS_ENDPOINT

# Index documents
uv run index.py --index $TF_VAR_app_name path/to/docs

# Check index size
curl -ku admin:$TF_VAR_opensearch_password "$OS_ENDPOINT/$TF_VAR_app_name/_count"

# Search
uv run search.py "What is the meaning of life?" --index $TF_VAR_app_name
```

### Deploy on AWS

```bash
docker build -t aws-rag .
docker run -e OPENAI_API_KEY="$TF_VAR_openai_api_key" -p 8000:8000 aws-rag

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
curl -u admin:$TF_VAR_opensearch_password "https://$(tofu output -raw opensearch_endpoint)/_cluster/health?pretty"
```

To destroy, use:

```bash
tofu destroy --auto-approve
```
