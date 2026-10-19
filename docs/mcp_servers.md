# MCP server reference

Five independent FastMCP servers, 29 tools. Each runs standalone over stdio by default,
or over HTTP with `MCP_TRANSPORT=http MCP_PORT=<port>`.

Every tool's description is written for an LLM to act on, and each documents four things:
**what it does**, **when to use it**, **what it returns**, and **what errors it can
raise** — plus a *Limitation* section where the data has a real caveat. Tests enforce the
presence of all four.

---

## 1. Financial Calculator — `financial-calculator`

`uv run calculator-mcp` · port 8001 · **no network, no database, no API key**

Deterministic arithmetic. This server exists because language models are unreliable at
maths and inconsistent across runs. Every number in a report must come from here.

| Tool | Inputs | Unit |
| --- | --- | --- |
| `calculate_revenue_growth` | previous_revenue, current_revenue | percent |
| `calculate_profit_margin` | net_income, revenue | percent |
| `calculate_gross_margin` | gross_profit, revenue | percent |
| `calculate_operating_margin` | operating_income, revenue | percent |
| `calculate_pe_ratio` | price_per_share, earnings_per_share | times |
| `calculate_debt_to_equity` | total_debt, total_equity | ratio |
| `calculate_roe` | net_income, shareholders_equity | percent |
| `calculate_roa` | net_income, total_assets | percent |
| `calculate_cagr` | beginning_value, ending_value, years | percent |
| `compare_financial_metrics` | metric, companies[], higher_is_better, unit | as supplied |

Every result carries `formula`, `inputs`, `interpretation` and `warnings`, which is what
makes a figure auditable rather than merely stated.

**Three distinct outcomes, never a silent zero:**

1. *Schema rejection* — wrong type or missing argument, before the function runs.
2. *`ToolError`* — mathematically undefined (zero denominator) or financially meaningless
   (growth from a negative base, negative share price, overflow to infinity).
3. *Result with `warnings`* — computable but easy to misread: negative equity, negative
   EPS, tiny-base growth effects, sub-year CAGR, ties in a comparison.

Full detail: [calculator_mcp.md](calculator_mcp.md).

---

## 2. Market Data — `market-data`

`uv run market-mcp` · port 8002 · provider: yfinance (no key required)

| Tool | Returns |
| --- | --- |
| `get_stock_price` | price, change, day range, volume, market state |
| `get_historical_prices` | OHLCV bars, period high/low, period return |
| `get_company_profile` | sector, industry, country, employees, business summary |
| `get_market_cap` | market cap, enterprise value, shares outstanding |
| `get_stock_statistics` | P/E (trailing and forward), P/B, P/S, EPS, beta, 52-week range |

Notable behaviours:

- **`missing_fields`** — `get_stock_statistics` names every statistic the provider did
  not supply, so the agent can say "not available" instead of inventing a number.
- **Down-sampling** — series longer than 400 bars are evenly sampled (latest bar always
  kept) and `truncated` is set, so a 10-year daily series does not swamp the context.
- **No vendor keys escape** — a test asserts the exact response key set and that
  `regularMarketPrice`, `trailingPE`, `marketCap` never appear.

---

## 3. Financial Statements — `financial-statements`

`uv run financial-mcp` · port 8003 · provider: yfinance

| Tool | Returns |
| --- | --- |
| `get_income_statement` | revenue, gross profit, operating income, net income, EPS, R&D |
| `get_balance_sheet` | assets, liabilities, debt, equity, cash, inventory |
| `get_cash_flow` | operating/investing/financing CF, capex, FCF, dividends, buybacks |
| `get_financial_ratios` | margins, ROE, ROA, D/E, current ratio + `inputs_used` |
| `get_earnings_data` | revenue, net income and EPS per period |

All accept `period`: `annual` or `quarterly` (also `ANNUAL`, `Q`, `yearly`…). Anything
else is rejected with the list of valid values.

Notable behaviours:

- **Missing is not zero.** Provider row labels vary by vendor and company; unrecognised
  line items go into `missing_items`, never default to `0.0`. `notes` says so explicitly.
- **Ratios are derived, not fetched.** Computed in Python from reported line items, with
  `inputs_used` showing exactly which figures produced them.
- **Period alignment.** Ratios match the balance sheet to the income statement's
  `period_ending`. Mixing FY2024 income with FY2023 equity would silently corrupt ROE.
- **Vendor labels do not leak.** `Total Revenue` becomes `revenue`; a test asserts no
  response key contains a space.

---

## 4. Financial News — `financial-news`

`uv run news-mcp` · port 8004 · provider: NewsAPI when `NEWSAPI_API_KEY` is set, else the
yfinance ticker feed

| Tool | Purpose |
| --- | --- |
| `search_company_news` | keyword search, optionally scoped to a ticker |
| `get_latest_company_news` | recent headlines for one company; needs no key |
| `search_news_by_date` | keyword search within an explicit date window |

Notable behaviours:

- **An article with no URL is dropped.** It cannot be cited, so it must not reach a report.
- **Empty results are explained.** `notes` says "no coverage was found, not that nothing
  happened" — absence of news is not evidence of absence of events.
- **The provider used is always disclosed.** Without a key the fallback filters recent
  headlines locally; `notes` says so and states the coverage limit.
- **The API key travels in a header, never the URL**, so it cannot land in a log line or
  an exception's request repr.

---

## 5. Research / RAG — `financial-research`

`uv run research-mcp` · port 8005 · Chroma vector store, sentence-transformer embeddings

| Tool | Purpose |
| --- | --- |
| `search_financial_documents` | semantic search with ticker/type/year filters |
| `retrieve_relevant_chunks` | passages for one company, larger default result count |
| `retrieve_document` | a whole document by id |
| `search_by_company` | list a company's indexed documents |
| `search_by_year` | list documents from a fiscal year |
| `index_financial_document` | add a filing, transcript or presentation |

Supported types: `10-K`, `10-Q`, `annual_report`, `earnings_report`,
`investor_presentation`, `earnings_transcript`.

Notable behaviours:

- **Chunk-level index, document-level metadata.** Every retrieved passage carries company,
  ticker, type, year, quarter and source, so it can be cited.
- **Filters run in the vector store**, not afterwards — post-filtering would return fewer
  results than requested while looking like a relevance failure.
- **Retrieved text is untrusted.** The server's instructions tell the agent to treat
  passages as quotations, never instructions; the agent scans them for injection.
- **Re-indexing replaces**, using a content-derived document id.
- **Empty results are explained**: "the corpus contains no such passage" ≠ "the company
  made no such statement".

Index the bundled sample filings with:

```bash
uv run python scripts_index_documents.py
# in Docker:
docker compose exec research-mcp python scripts_index_documents.py
```

---

## Running a server standalone

Each server is genuinely independent — a test starts each as a subprocess with nothing
else running:

```bash
uv run calculator-mcp                                   # stdio
MCP_TRANSPORT=http MCP_PORT=8001 uv run calculator-mcp  # HTTP
```

Verify one by hand:

```bash
curl -fsS -X POST http://localhost:8001/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"1"}}}'
```

A bare `GET /mcp` returns `400 Bad Request: Missing session ID` — that means the server is
healthy and speaking MCP, not broken. The Compose healthchecks perform the handshake above
for exactly this reason.
