# Interview notes

Twenty-five questions this project should let you answer well, with the reasoning rather
than the slogan. Where the honest answer is "this is a weakness", it says so.

---

## MCP

### 1. What is MCP?

The Model Context Protocol is a JSON-RPC protocol that lets an application expose
**capabilities** — tools, resources and prompts — to an LLM client over a transport
(stdio, HTTP, or in-memory). The client discovers what exists at runtime by calling
`list_tools()`; it does not import anything.

The shift is from *tools as function references* to *tools as a protocol surface*. That
sounds academic until you notice what it buys: the agent in this project has never
imported `yfinance`, `chromadb` or any calculation function. It knows only what
`list_tools()` returned.

### 2. Why use MCP instead of ordinary function calling?

Function calling is fine until roughly the point you have twenty tools. Then:

| Problem with in-process tools | What MCP changes |
| --- | --- |
| One shared dependency set — `chromadb` drags an 80MB ONNX model into the process serving HTTP | Each server has its own dependencies |
| One tool's crash takes down the agent | A dead server degrades one capability |
| Cannot scale, restart or rate-limit tools separately | Each server is a separate process |
| Tools are locked inside your application | Any MCP client can use them |
| Adding a tool means editing the agent | Adding a tool means editing a server |

Honest counterpoint: for three tools in one process, MCP is overhead. The break-even is
somewhere around "tools with genuinely different dependencies or failure modes." This
project is past it — five servers with five different failure profiles.

### 3. MCP client vs MCP server

The **server** exposes capabilities and knows nothing about who is calling. The **client**
connects, discovers, and invokes.

In this codebase the split is enforced, not just intended: `agent/mcp_client.py` is the
only module that imports `fastmcp`, and `tests/test_mcp_registry.py` fails if any other
module under `agent/` reaches for it. The graph sees a list of LangChain tools and nothing
else.

### 4. Tools vs resources

**Tools** are actions the model invokes with arguments — everything here is a tool.
**Resources** are addressable content the client reads (`file://`, `db://…`), more like a
filesystem than a function.

A defensible alternative design would expose indexed filings as resources rather than
through `retrieve_document`. I kept them as tools because retrieval needs *arguments* —
semantic query, ticker filter, year filter — and a resource URI is a poor place to put a
similarity search.

### 5. Why FastMCP?

The raw MCP SDK requires hand-writing JSON schemas and result envelopes. FastMCP derives
both from Python type hints and docstrings:

```python
@mcp.tool
async def get_stock_price(ticker: TickerArg) -> StockQuote:
    """Get the current share price and today's move for one company.
    ...
    """
```

That single decorator produces the input schema (from `Annotated[..., Field(...)]`), the
output schema (from the Pydantic return type), and the tool description (from the
docstring). The description is what the LLM actually uses to choose, so keeping it next to
the signature means it cannot drift out of date.

It also provides an in-memory transport, which is why this project's tests can run the
real servers over the real protocol in milliseconds.

### 6. How does tool discovery work here?

At startup, `MCPToolkit.connect()` opens a client per server, calls `list_tools()`, and
stores each tool's name, description and JSON schemas. `langchain_tools()` then wraps each
one as a LangChain `StructuredTool`, **passing the MCP input schema through untouched** —
the model sees exactly the parameter names, types and descriptions the server declared.

A test asserts the round-trip: `"greater than zero"` from the calculator server's `Field`
description survives all the way into `tool.tool_call_schema`.

### 7. How do multiple MCP servers work together?

One client, N connections, one flat tool list handed to the model. Three details matter:

- **Independent failure.** A server that will not start is recorded in `failed_servers`
  and skipped. An outage in the news server must not prevent a purely numerical question
  from being answered.
- **Name collisions.** If two servers declare the same tool name, the second is exposed as
  `<server>__<tool>` and a warning is logged — shadowing would be silent and awful.
- **Attribution.** Every tool carries `metadata["mcp_server"]`, so the response can report
  which servers actually contributed.

### 8. Why did you write your own MCP-to-LangChain adapter?

Because `langchain-mcp-adapters` is currently incompatible with the MCP SDK v2 that
FastMCP 4 requires — it fails at import with
`cannot import name 'RequestContext' from 'mcp.shared.context'`.

That forced a decision, and the replacement turned out better for the specific thing this
project cares about: **how a tool failure is presented to the model.** My adapter returns
failures as JSON the model can read —

```json
{"error": "…", "guidance": "This tool call failed. Do not invent the value it would
 have returned. Either try a different tool or report the data as unavailable."}
```

— rather than raising, which would abort the graph, or returning empty, which invites the
model to fill the gap from imagination.

---

## LangGraph

### 9. Why LangGraph rather than a sequential chain?

A chain is a fixed pipeline. This workflow needs three things a chain does not give you:

1. **Loops with bounded iteration.** The collection stage calls tools repeatedly until the
   model stops, capped at 8 steps. A chain has no natural place for "repeat until done, but
   not forever".
2. **Persistent state across invocations.** The checkpointer is what makes follow-ups work.
   A chain has no state between calls.
3. **Inspectable structure.** `graph.get_graph().nodes` is a real object a test asserts
   against, and the stage a failure happened in is a first-class fact.

A chain would also make the anti-fabrication design awkward: the report generator needs
the *whole* accumulated message history, including failed tool calls, not just the
previous step's output.

### 10. How does state work?

`AnalystState` is a `TypedDict`. Most keys are last-write-wins; `messages` uses the
`add_messages` reducer so each node appends rather than overwrites.

The state is deliberately explicit about provenance — `evidence`, `metrics`, `analysis`,
`risks` and `report` are separate fields, not one blob of prose. That is what lets the
report generator label retrieved facts, calculated metrics and interpretation
differently, and lets the API return them separately.

### 11. How is tool selection LLM-driven, and how do you know?

All tools are bound to the model; the model emits `tool_calls`; the graph executes them.
No code inspects the query text.

"How do you know" is the interesting half. It is a test:

```python
def test_no_keyword_routing_exists_in_the_agent():
    source = Path("agent/graph.py").read_text()
    for banned in ("in query", 'in state["query"]', "query.lower()"):
        assert banned not in source.lower()
    for tool_name in ("get_stock_price", "get_income_statement", "calculate_revenue_growth"):
        assert tool_name not in source
```

The graph does not contain the name of a single tool. If someone later adds
`if "stock" in query`, the suite fails rather than the architecture quietly eroding.

### 12. Then why do stages have restricted tool sets? Isn't that routing?

No — it is a **permission boundary**. Within its set the model still chooses freely; the
restriction only says which sets exist.

It was added because of a real bug. The calculation stage was binding calculator tools but
executing lookups against the *full* tool table, so a model that asked for
`get_stock_price` during calculation silently got it — bypassing the audit trail the
collection stage builds. Now an out-of-stage request returns "not available during
calculation" plus the permitted list, and the model adapts.

### 13. How are failures handled inside the graph?

Four layers, because they fail differently:

| Failure | Handling |
| --- | --- |
| Tool raises | Captured by the MCP client, returned as JSON with "do not invent" guidance |
| Tool does not exist | ToolMessage naming the available tools; the run continues |
| Tool not permitted in this stage | Same, plus a warning |
| Server never started | Recorded at connect; its tools are simply absent |

Everything is also collected into `warnings`, which reach the API response and the UI. A
failure the user cannot see is a failure that becomes a hallucination.

### 14. How do you stop the agent looping forever?

Hard caps: 8 collection steps, 6 calculation steps. Hitting the cap adds a warning that
the analysis may rest on incomplete data — the user is told, rather than being handed a
truncated answer that looks complete. A test scripts a model that never stops calling
tools and asserts both the cap and the warning.

### 15. How is the agent testable without an API key?

The model is injected. Tests pass a `ScriptedChatModel` that returns pre-programmed
`AIMessage`s, including ones carrying `tool_calls`, and records which tools it was offered.

The important property: **only the model's choices are scripted.** The graph, the MCP
client, the FastMCP servers and the calculations are all real. When a test asserts
`revenue_growth == 125.85`, that number was computed by the actual calculator server over
the actual protocol.

---

## Architecture

### 16. Why a service layer between MCP tools and the provider?

```
FastMCP tool → service → provider adapter → external API
             ← domain model ← normalised payload ←
```

Three jobs that do not belong in a tool:

1. **Normalisation.** `Total Revenue`, `TotalRevenue` and `Revenues` all become `revenue`.
   A test asserts no response key contains a space, so vendor labels cannot leak.
2. **Error translation.** Every provider failure becomes one of five domain exceptions. No
   HTTP status code or vendor exception type escapes.
3. **Caching.** Applied at the service layer so what is cached is the *normalised domain
   object* — a provider swap does not invalidate the cache.

`services/providers/yfinance_provider.py` is the only file in the codebase that knows
yfinance exists.

### 17. Why structured outputs rather than text?

A tool returning `"NVIDIA's revenue was $60.9 billion"` forces the model to re-parse its
own tool output, which is exactly where numbers get mangled. A tool returning
`{"revenue": 60922000000.0}` does not.

Structured output is also what makes provenance possible:

```json
{"metric": "revenue_growth", "value": 125.85,
 "formula": "((current_revenue - previous_revenue) / previous_revenue) * 100",
 "inputs": {"previous_revenue": 26974000000.0, "current_revenue": 60922000000.0}}
```

The API can surface that table directly, and the guardrail layer can cross-check every
number in the report against the set of values tools actually produced.

### 18. How do you know the model didn't invent a number?

Two independent mechanisms.

**Prevention:** the model is told never to do arithmetic, and the calculator tools are
right there. Prompts state it; the tool descriptions repeat it.

**Detection:** at report time, every number in the text is extracted and compared against
the values returned by tools during that run. Anything unmatched is flagged — excluding
years and small counts, and accepting rounded restatements (125.9 for 125.85), because a
detector that fires on "the last 3 years" is a detector people turn off.

Flags are **appended visibly** rather than used to rewrite the text. A reader who can see
that a figure was questioned is better served than one shown a quietly edited report.

### 19. Why not let the LLM do the arithmetic?

Two reasons, one obvious and one that matters more.

Obvious: models make arithmetic errors, particularly on large numbers with many digits.

More important: **they are not consistently wrong.** The same question twice can yield
different numbers, which makes the system untestable and unauditable. A test asserts that
two identical runs produce byte-identical `model_dump()` output. That guarantee is
impossible if a model is doing the maths.

### 20. How would this scale?

| Concern | Now | Next |
| --- | --- | --- |
| MCP servers | one container each | replicas behind service DNS; all stateless except research |
| Provider rate limits | per-kind TTL cache | shared token bucket in Redis, per provider |
| Agent throughput | async, one process | more API replicas; state already in PostgreSQL |
| Vector store | embedded Chroma | pgvector or Qdrant once the corpus outgrows a volume |
| Cost | temperature 0, bounded loops | prompt caching; a cheaper model for understanding and planning |

The genuinely scalable decision was putting conversation state in PostgreSQL rather than
process memory. API replicas are interchangeable today.

### 21. How would you secure the MCP servers?

They currently trust their caller, which is right for a single-tenant private Compose
network and wrong for anything shared. In order:

1. Do not publish the MCP ports; internal network only.
2. Authentication middleware per server.
3. Per-caller rate limits **at the server**, not just the client — a compromised client
   should not be able to exhaust a provider quota.
4. Separate read from write. `index_financial_document` mutates the corpus and should
   require a stronger credential than the search tools.
5. Keep argument validation at the MCP boundary. That is the trust boundary between
   LLM-chosen arguments and a real provider, which is why tickers are regex-validated and
   `../../etc/passwd` is a test case.

### 22. How does the same code run as one process and as nine containers?

`MCPServerSpec` supports three transports for one protocol:

| Mode | When |
| --- | --- |
| stdio subprocess | local development, no orchestration needed |
| HTTP (`<NAME>_MCP_URL`) | Docker Compose, each server its own container |
| in-process | tests and single-process deployment |

Setting five environment variables switches the whole system. No code changes.

This also fixed a genuine bug: tests were mocking `yfinance` in the parent process while
the MCP servers ran as **subprocesses with their own import**, so those tests were quietly
hitting the live network. Moving tests to the in-process transport made mocking effective
and cut the suite from 17s to 1s.

---

## Production

### 23. What is your caching strategy, and where does it bite?

Per-kind TTLs, because "financial data" is not one thing: a quote is stale in 60 seconds,
a filed annual statement never changes.

Three decisions worth defending:

- **A cache failure is a miss, never an error.** A broken Redis makes the system slower,
  not wrong. Tested with a client where every operation raises.
- **Cache keys refuse credential-shaped parameters** with an exception rather than
  dropping them. A key containing an API key would sit in Redis in plaintext.
- **A stale *shape* is a miss.** After a deploy that changes a model, old entries must not
  break requests.

Where it bit: `from __future__ import annotations` makes return annotations **strings**, so
`hasattr(annotation, "model_validate")` was false and the cache returned raw dicts instead
of domain models. Caught by a test; fixed with `typing.get_type_hints`.

### 24. What does observability actually give you here?

One `request_id`, bound once at the API boundary to an async-aware context variable, and
present on every subsequent log line in that request — API, agent stage, MCP client, MCP
server, tool — without being threaded through a single function signature.

A test proves it end to end: it issues one request with `X-Request-ID: trace-e2e-001` and
asserts that id appears on `analysis_started`, `query_understood`, `mcp_tool_called`,
`tool_completed`, `analysis_completed` and `request_completed`.

On secrets: a redaction processor runs **last, before rendering**, dropping 15 sensitive
keys and scrubbing credential-shaped text. Not because we log keys deliberately, but
because the realistic leak is a key inside a third-party exception message nobody
anticipated. That exact bug existed here: the market service interpolated raw vendor
exceptions into the error the *LLM* saw.

### 25. What are the weakest parts of this implementation?

Being specific, because "no weaknesses" is not a credible answer:

- **The data provider.** yfinance is unofficial, rate-limited and reshapes payloads
  without notice. The adapter contains the blast radius, but the data quality ceiling is
  low. A real deployment pays for a licensed provider.
- **The RAG corpus is manual.** No EDGAR fetcher; the bundled filings are extracts. The
  retrieval machinery is production-shaped; the corpus is a demo.
- **Chunking is naive.** Fixed-size with overlap and a separator preference. A 10-K has
  real structure (Item 1A, Item 7) that a section-aware splitter should exploit, and
  tables are destroyed by character-level chunking.
- **No token budget.** Tool loops are capped by *steps*, not cost. A pathological query
  can still be expensive.
- **Single-tenant.** Conversations are isolated by id, not by owner. Anyone with an id can
  read a conversation.
- **The unsourced-number detector is heuristic.** It compares rendered strings and will
  miss a fabricated figure expressed in different units ("$60.9 billion" vs
  `60922000000`). It raises the cost of fabrication; it does not eliminate it.
- **Fiscal-year alignment.** NVIDIA's FY2024 ends in January, AMD's in December.
  Comparisons match `period_ending`, but a cross-company "FY2024" is still not
  apples-to-apples, and the report should say so more loudly than it does.
- **`query_understanding` and `research_planner` parse JSON out of free text.** Structured
  output would be more robust; the fallback is a permissive parser plus sane defaults.

---

## Bugs this project actually shipped and fixed

Worth having ready — "what went wrong" is usually a better answer than "what went right".

1. **Tests silently hitting the live network.** MCP servers ran as subprocesses with their
   own `yfinance` import, so mocking in the parent did nothing. Fixed by adding an
   in-process transport. Suite went from 17s to 1s and became genuinely offline.
2. **Cache returning raw dicts.** `from __future__ import annotations` turned return
   annotations into strings, defeating model revival.
3. **Stage tool permissions unenforced.** The calculation stage could execute data tools it
   was never offered.
4. **Credentials reaching the LLM.** The market service interpolated raw provider
   exceptions — which can echo a failing URL — into the error the model saw.
5. **Calculator overflow.** Inputs were checked for finiteness; results were not, so
   `1e308 / 1e-308` produced `inf`.
6. **Docker healthchecks marking healthy servers unhealthy.** `GET /mcp` returns
   `400 Missing session ID`; the check now performs a real MCP `initialize` handshake.
7. **`EphemeralClient` sharing state.** Chroma's in-memory client is process-wide, so
   "isolated" test stores were sharing documents.
8. **Streamlit `SyntaxError` at runtime.** A ternary-as-statement broke Streamlit's AST
   magic transform — invisible to both `ruff` and `ast.parse`, caught only by running the
   app.
