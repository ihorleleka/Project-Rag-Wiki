# kb-service

Repository-scoped MCP knowledge service for Markdown wiki content.

It indexes Markdown files from a mounted wiki folder, stores vectors in ChromaDB, and serves:
- MCP endpoint (streamable HTTP)
- REST endpoints for search/read/write/reindex
- health/readiness endpoints

## What This Image Expects

- A wiki folder mounted at `/workspace/wiki`
- A writable KB state folder mounted at `/workspace/.kb`
- A shared models cache KB state folder mounted at `/root/.cache/huggingface/hub`

Do not bake runtime `.kb` state into images.

## Runtime Defaults

- `KB_WIKI_ROOT=/workspace/wiki`
- `KB_ROOT=/workspace/.kb`
- `KB_PORT=1111`
- `KB_MCP_PATH=/mcp/`
- `KB_HEALTH_PATH=/health`
- `KB_EMBEDDING_MODEL=all-MiniLM-L6-v2`
- `KB_CHUNK_SIZE=500`
- `KB_CHUNK_OVERLAP=150`
- `KB_TOP_K=8`
- `KB_MERGE_ADJACENT_WINDOW=1`
- `KB_WATCH_INTERVAL_SECONDS=15`

## Run

```bash
docker run --rm \
  -p 1111:1111 \
  -v "$(pwd)/wiki:/workspace/wiki" \
  -v "$(reponame)-kb-data:/workspace/.kb" \
  -v "kb-models:/root/.cache/huggingface/hub" \
  ihorleleka/project-rag-wiki:latest
```

## Release Automation

Image versioning is driven from the Git tag.

- Tag releases as `X.Y.Z`.
- The GitHub Actions workflow at [`.github/workflows/docker-release.yml`] builds and pushes the Docker image on tag pushes.
- The workflow passes the tag name directly into the Docker build as `VERSION`.
- That same `VERSION` value is used for the OCI image label and the installed Python package version inside the image.

Set these repository settings before using the workflow:

- Secret `DOCKERHUB_USERNAME`
- Secret `DOCKERHUB_TOKEN`

## Endpoints

- Health: `GET /health`
- Ready: `GET /ready`
- MCP: `POST /mcp/` (also mounted at `/mcp`)
- Reindex: `POST /reindex`
- Search: `POST /search`
- List docs: `GET /list`
- Read doc: `POST /read`
- Write doc: `POST /write`
- Append doc: `POST /append`

## License

MIT. See `LICENSE`.
