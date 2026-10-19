# Financial Calculator MCP

**Server name:** `financial-calculator` · **Module:** `servers.calculator_mcp.server` ·
**Entrypoint:** `uv run calculator-mcp`

Deterministic financial mathematics exposed as MCP tools. No network, no database, no API
key, no LLM — the server runs in complete isolation from the rest of the system.

## Why it exists

Language models are unreliable at arithmetic and inconsistent across runs. Every number
that reaches a report must come from one of these tools, so that the same inputs always
produce the same figure and the figure can be audited afterwards.

## Design

```
MCP tool (server.py)          validates + logs, maps errors to ToolError
      │
      ▼
pure function (calculations.py)   the only place arithmetic happens
      │
      ▼
CalculationResult (models/)       value + formula + inputs + interpretation + warnings
```

Every result carries the formula and the exact inputs used. That is what makes
"calculated values can be traced back to tool results" true rather than aspirational.

## Tools

| Tool | Inputs | Unit | Notes |
| --- | --- | --- | --- |
| `calculate_revenue_growth` | `previous_revenue`, `current_revenue` | percent | Two periods only; use CAGR for longer spans |
| `calculate_profit_margin` | `net_income`, `revenue` | percent | Bottom-line margin |
| `calculate_gross_margin` | `gross_profit`, `revenue` | percent | Most peer-comparable margin |
| `calculate_operating_margin` | `operating_income`, `revenue` | percent | Excludes capital structure and tax |
| `calculate_pe_ratio` | `price_per_share`, `earnings_per_share` | times | Negative EPS ⇒ flagged not meaningful |
| `calculate_debt_to_equity` | `total_debt`, `total_equity` | ratio | Negative equity ⇒ flagged |
| `calculate_roe` | `net_income`, `shareholders_equity` | percent | Read alongside D/E — leverage inflates it |
| `calculate_roa` | `net_income`, `total_assets` | percent | Not inflated by leverage |
| `calculate_cagr` | `beginning_value`, `ending_value`, `years` | percent | Hides everything between the endpoints |
| `compare_financial_metrics` | `metric`, `companies[]`, `higher_is_better`, `unit` | as supplied | Explicit ranking direction |

## Error policy

Three distinct outcomes, never a silent zero:

1. **Schema rejection** — wrong type or missing argument, caught by Pydantic before the
   function runs.
2. **`ToolError`** — mathematically undefined (zero denominator) or financially
   meaningless (growth from a negative base, negative share price). The agent receives a
   readable reason; the server stays up and serves the next call.
3. **Result with `warnings`** — computable and returned, but unusual enough that ranking
   or comparing it would mislead: negative equity, negative EPS, tiny-base growth effects,
   sub-year CAGR, ties in a comparison.

NaN and infinity are rejected at the boundary so they can never propagate.

## Running and testing

```bash
uv run calculator-mcp                       # stdio (default)
MCP_TRANSPORT=http MCP_PORT=8001 uv run calculator-mcp

uv run pytest tests/test_calculations.py    # pure maths, 56 tests
uv run pytest tests/test_calculator_mcp.py  # MCP protocol surface, 23 tests
```
