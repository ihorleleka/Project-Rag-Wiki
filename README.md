# Project RAG wiki

[![Docker Hub](https://img.shields.io/docker/v/ihorleleka/project-rag-wiki?sort=semver&label=docker%20hub)](https://hub.docker.com/repository/docker/ihorleleka/project-rag-wiki)

Repository-scoped MCP knowledge service for Markdown wiki content.

It indexes Markdown files from a mounted wiki folder, stores vectors in ChromaDB, and serves:
- MCP endpoint (streamable HTTP)
- health endpoint

The MCP surface is intentionally small:
- Active tools: `wiki_search`, `wiki_read`, `wiki_list`, `wiki_write`

## Retrieval Model

Markdown files remain the saved and editable source of truth. During indexing,
the service derives additional context packet records from well-structured wiki
notes and stores those packet records alongside raw chunks in ChromaDB.

A note can compile into a decision-ready packet when it uses frontmatter such as:

```yaml
---
id: stable-note-id
scope: project-specific
last_verified: YYYY-MM-DD
status: active
applies_to:
  - domain-or-component
---
```

and semantic sections:

```markdown
## Use this when
## Decision
## Do
## Do not
## Evidence
## Retrieval hints
```

`wiki_search` prefers matching packet records before raw chunks. Packet results
include normalized fields such as `rule`, `confidence`, `source`,
`last_verified`, `needs_verification`, `applies_to`, `do`, `do_not`, `evidence`,
and `gaps`.

Packet embeddings are built from the decision-ready sections and `applies_to`.
The full source prose is kept as metadata/fallback, not as the primary packet
embedding text.

## Write Model

Use `wiki_write` to create or replace complete Markdown notes. The service
reindexes after each write and regenerates derived packet records automatically.

There is no append tool by design. Agents should read the current note, merge
changes locally, and write a complete coherent document so frontmatter,
semantic sections, links, and retrieval hints stay consistent.

## Agent Harness

For an agent consumer of this service, see [@ihorleleka/harness](https://github.com/ihorleleka/harness).

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
- `KB_STALENESS_DAYS=90`
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
- MCP: `POST /mcp/` (also mounted at `/mcp`)

The health response is `200` only when the service startup reindex has completed successfully and the MCP session manager is running.

## License

MIT. See [LICENSE](LICENSE).
