# Deployment and operations

## Local development

```bash
uv sync --extra dev
cp .env.example .env      # fill in ANTHROPIC_API_KEY

docker compose up -d postgres redis          # persistence + cache
uv run python scripts_index_documents.py     # index the sample filings

uv run uvicorn api.main:app --reload         # terminal 1
uv run streamlit run frontend/app.py         # terminal 2  -> http://localhost:8501
```

The MCP servers are launched automatically as stdio subprocesses by the MCP client. You
do not start them yourself.

> **Port 5432 in use?** Set `POSTGRES_PORT=5433` in `.env` — a local PostgreSQL often
> holds 5432. The compose file honours it.

## Docker Compose (full stack)

```bash
cp .env.example .env      # POSTGRES_PASSWORD is required, not defaulted
docker compose up -d --build
docker compose ps         # all 9 services should be (healthy)

docker compose exec research-mcp python scripts_index_documents.py

open http://localhost:8501     # UI
open http://localhost:8000/docs  # API docs
```

Nine services:

| Service | Port | Purpose |
| --- | --- | --- |
| `ui` | 8501 | Streamlit |
| `api` | 8000 | FastAPI |
| `calculator-mcp` | 8001 | deterministic maths |
| `market-mcp` | 8002 | prices, profiles, statistics |
| `financial-mcp` | 8003 | statements and ratios |
| `news-mcp` | 8004 | news search |
| `research-mcp` | 8005 | document RAG |
| `postgres` | 5432* | transcripts + LangGraph checkpoints |
| `redis` | 6379 | response cache |

\* configurable via `POSTGRES_PORT`.

### How the API finds the MCP servers

Compose sets one variable per server:

```yaml
CALCULATOR_MCP_URL: http://calculator-mcp:8001/mcp
MARKET_MCP_URL:     http://market-mcp:8002/mcp
# ...
```

Unset them and the identical image launches the servers as stdio subprocesses instead.
Same code, same protocol, different transport.

### Health

```bash
curl -s http://localhost:8000/health | jq
```

```json
{
  "status": "degraded",
  "mcp_servers": {"calculator": true, "market": true, "financial": true,
                  "news": true, "research": true},
  "failed_servers": {},
  "agent_ready": false,
  "database_connected": true,
  "checkpointer_backend": "postgres",
  "cache": {"backend": "redis", "hits": 0, "misses": 0, "hit_rate": 0.0}
}
```

`status` is `ok` only when **every** dependency is healthy. `degraded` with
`agent_ready: false` above means no `ANTHROPIC_API_KEY` is configured — `/chat` returns
503 while `/tools`, `/companies` and `/health` keep working. That is deliberate.

## Configuration

All configuration is environment-based, read once in `services/config.py`. Nothing is
hardcoded; `.env` is git-ignored and excluded from the Docker build context.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `ANTHROPIC_API_KEY` | for `/chat` | — | LLM |
| `LLM_MODEL` | no | `claude-sonnet-5` | model id |
| `NEWSAPI_API_KEY` | no | — | historical news search; falls back to the ticker feed |
| `FMP_API_KEY`, `ALPHA_VANTAGE_API_KEY` | no | — | alternate providers |
| `DATABASE_URL` | for persistence | local postgres | transcripts + checkpoints |
| `POSTGRES_PASSWORD` | yes (Compose) | — | **required**, no default |
| `POSTGRES_PORT` | no | 5432 | host port |
| `REDIS_URL` | no | `redis://localhost:6379/0` | cache |
| `CACHE_TTL_SECONDS` | no | 900 | `0` disables caching |
| `CHROMA_PERSIST_DIR` | no | `./data/chroma` | vector store |
| `<NAME>_MCP_URL` | no | — | HTTP transport for that MCP server |
| `API_BASE_URL` | no | `http://localhost:8000` | used by the UI |
| `LOG_LEVEL` | no | `INFO` | structured log level |
| `HTTP_TIMEOUT_SECONDS` | no | `15` | provider call timeout |

## Secrets

- `.env` is git-ignored; only `.env.example` (all values blank) is tracked.
- `tests/test_secrets.py` scans every source directory for connection strings and key
  prefixes on each run.
- Logs pass through a redaction processor as the last step before rendering: 15 sensitive
  keys are dropped and credential-shaped text is scrubbed, including inside third-party
  exception messages.
- Cache keys refuse credential-shaped parameters outright — a key containing an API key
  would sit in Redis in plaintext.
- The Docker image contains no `.env`, has zero secret-bearing layers, and runs as
  uid 10001.

## Observability

Every log line is JSON (a coloured console renderer in a TTY) and carries the same context:

```json
{"event": "mcp_tool_called", "server": "calculator", "tool": "calculate_revenue_growth",
 "agent_stage": "calculation", "duration_ms": 0.83, "request_id": "trace-e2e-001",
 "conversation_id": "3f9a…", "level": "info", "timestamp": "2026-09-07T02:41:00Z"}
```

`request_id` comes from the `X-Request-ID` header (or is generated) and is bound to a
context variable at the API boundary. Because structlog's contextvars are async-aware,
every downstream line — agent stage, MCP client, MCP server, tool — carries it with no
parameter threading.

Trace one request:

```bash
docker compose logs api | grep '"request_id":"<id>"'
```

## Scaling

| Concern | Now | Next step |
| --- | --- | --- |
| MCP servers | one container each | replicas behind a service DNS name; they are stateless except research |
| Provider rate limits | per-kind TTL cache | a shared token bucket in Redis, keyed per provider |
| Agent throughput | async, one process | more API replicas; state already lives in PostgreSQL |
| Vector store | embedded Chroma | a managed store (pgvector, Qdrant) once the corpus outgrows one volume |
| Cost | temperature 0, bounded tool loops (8 collect / 6 calculate) | prompt caching, a cheaper model for the understanding and planning stages |

## Securing MCP servers

Right now the servers trust their caller, which is correct for a single-tenant deployment
on a private Compose network. For anything shared:

1. Do not publish the MCP ports; keep them on the internal network only.
2. Put authentication in front of each server (FastMCP supports auth middleware).
3. Enforce per-caller rate limits at the server, not just the client.
4. Split read from write: `index_financial_document` mutates the corpus and should require
   a stronger credential than the search tools.
5. Keep the ticker and date validation at the MCP boundary — it is the trust boundary
   between LLM-chosen arguments and a real provider.

## Known limitations

- **Data provider.** yfinance is unofficial, rate-limited and occasionally reshapes its
  payloads. The service layer normalises and the provider adapter is the only file that
  knows about it, so swapping in a paid provider is a contained change — but the current
  data quality is what it is.
- **News without a key.** Without `NEWSAPI_API_KEY` the news server filters a recent
  ticker feed locally. Older windows return nothing, and `notes` says so.
- **RAG corpus.** Documents must be indexed explicitly; there is no EDGAR fetcher. The
  bundled samples are extracts, not complete filings.
- **No analyst estimates.** The provider supplies none, so `eps_estimate` and
  `surprise_percent` are always null and the tool says so rather than guessing.
- **Fiscal-year alignment.** NVIDIA's FY2024 ends in January; AMD's ends in December.
  Comparisons compare `period_ending`, but a cross-company "FY2024" is still not
  apples-to-apples and the report should say so.
- **Single-tenant.** No authentication, no per-user isolation. Conversations are isolated
  by id, not by owner.
- **Cost is unbounded per query** beyond the tool-loop caps. There is no token budget.
