# Volve Drilling Advisor — Strands Edition

**Well:** 15/9-F-15 | Volve Field | Norwegian Continental Shelf  
**Stack:** AWS Strands Agents SDK · Anthropic Claude · Pandas · Real Equinor Volve Data

A domain-expert AI drilling advisor rebuilt on the [AWS Strands Agents SDK](https://strandsagents.com) — porting the original raw Anthropic API implementation to demonstrate how an agentic framework collapses boilerplate without sacrificing domain depth.

---

## What It Does

Analyzes real drilling telemetry from Equinor's open-source Volve field dataset across 13 depth windows (3,305m–4,085m). For each window, the agent autonomously calls 4 diagnostic tools, reasons across formation context, drillability forecasts, ROP trends, and bit efficiency — then delivers a depth-referenced drilling advisory.

No simulated data. No toy examples. Real WITSML-sourced parameters from a Norwegian Continental Shelf well.

---

## Raw API vs Strands — The Core Contrast

This repo exists to make one architectural point concrete:

| Concern | Raw Anthropic API | AWS Strands SDK |
|---|---|---|
| Tool schema definition | Manual JSON schema per tool | `@tool` decorator — inferred from type hints + docstring |
| Tool dispatch | Hand-written `if/elif` dispatcher | Framework handles routing automatically |
| Agent loop | ~60 lines: `while`, stop reason checks, tool result injection | `Agent()` — one call |
| Conversation state | Manual message list construction | Managed internally |
| Total agent infrastructure | ~200 lines | ~10 lines |
| Domain logic touched? | No | No |

The `@tool` decorator is the sharpest illustration:

**Raw API — you write the schema:**
```python
tools = [{
    "name": "get_formation_context",
    "description": "Get formation context at current depth...",
    "input_schema": {
        "type": "object",
        "properties": {
            "depth_m": {"type": "number", "description": "Current depth in metres"}
        },
        "required": ["depth_m"]
    }
}]

# Then a dispatcher:
if tool_name == "get_formation_context":
    result = get_formation_context(tool_input["depth_m"])
elif tool_name == "get_drillability_forecast":
    ...
```

**Strands — the decorator does it:**
```python
@tool
def get_formation_context(depth_m: float) -> dict:
    """
    Get formation context from offset well Silver tables at current depth.
    Returns formation position (Draupne/Hugin/below), nearby HC flags,
    and reservoir quality information.
    """
    ...  # just the logic
```

Schema inferred. Dispatch handled. No boilerplate.

**The agent loop — same contrast:**

```python
# Raw API: ~60 lines of this
while True:
    response = client.messages.create(...)
    if response.stop_reason == "end_turn":
        break
    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch(block.name, block.input)
                tool_results.append({...})
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

# Strands: one line
response = agent(prompt)
```

---

## Architecture

```
advisor.py
│
├── Data Layer (CSV → Pandas)
│   ├── ROP data.csv                    ← Real F-15 drilling parameters
│   ├── silver_formation_tops.csv       ← From offset-well-intelligence-crew
│   ├── silver_reservoir_flags.csv      ← HC potential flags
│   └── silver_drillability_forecast.csv
│
├── 4 Tools (@tool decorator)
│   ├── get_formation_context           ← Formation position + HC flags
│   ├── get_drillability_forecast       ← HARD/MODERATE/SOFT from offset analogs
│   ├── check_rop_trend                 ← ROP vs 5-window rolling trend
│   └── check_mse_efficiency            ← Bit efficiency: WOB × RPM / ROP
│
├── Strands Agent (fresh per window)
│   └── Agent(model, system_prompt, tools)
│       └── Autonomously calls tools → synthesizes advisory
│
└── 13 Depth Windows (3,305m → 4,085m)
    └── One advisory per window → sample_output/drilling_advisory_strands.md
```

**Why fresh agent per window?**  
Each depth window is a stateless advisory — no context from the previous window is relevant. A single persistent agent accumulates conversation history across 13 iterations and hits `MaxTokensReachedException` by window 3. Resetting per window is both the correct architecture and the efficient one.

---

## Domain Intelligence

The system prompt encodes ~20 years of North Sea drilling expertise. Key field context baked in:

- Hugin reservoir entry at ~3,350m (offset well analog: F-11B)
- CRITICAL HC flags at 3,350m and 3,700m
- Best confirmed reservoir: 3,800–3,900m (RHOB 2.31–2.32, RT 911–3,322 ohm·m)
- CRITICAL anomaly at 4,000–4,050m (shale where offsets show HC sand)

The agent cross-references live drilling parameters against this context on every window — flagging formation transitions, diagnosing ROP drops (formation-driven vs bit-driven), and issuing severity-appropriate recommendations.

---

## Data Sources

Real open-source data from Equinor's Volve field release:

| File | Source | Description |
|---|---|---|
| `ROP data.csv` | Volve WITSML | F-15 real-time drilling parameters |
| `silver_formation_tops.csv` | [offset-well-intelligence-crew](https://github.com/venkatchittoor/offset-well-intelligence-crew) | Formation picks from multi-agent crew |
| `silver_reservoir_flags.csv` | offset-well-intelligence-crew | HC potential flags |
| `silver_drillability_forecast.csv` | offset-well-intelligence-crew | Offset analog drillability |

---

## How to Run

```bash
# Clone
git clone https://github.com/venkatchittoor/volve-drilling-advisor-strands.git
cd volve-drilling-advisor-strands

# Environment
python3 -m venv venv
source venv/bin/activate
pip install strands-agents strands-agents-tools pandas python-dotenv

# API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Run
python3 advisor.py
```

Output saved to `sample_output/drilling_advisory_strands.md`.

---

## Related Projects

| Repo | Description |
|---|---|
| [volve-drilling-advisor](https://github.com/venkatchittoor/volve-drilling-advisor) | Original — raw Anthropic API, constrained tool-calling |
| [offset-well-intelligence-crew](https://github.com/venkatchittoor/offset-well-intelligence-crew) | 5-agent crew that produced the Silver tables used here |
| [drilling-npt-agent](https://github.com/venkatchittoor/drilling-npt-agent) | NPT monitoring agent — Eyes/Brain/Hands pattern |

---

## The Spectrum

This project sits on a deliberate progression:

```
Prompt-based           Constrained              True                  Strands
structured output  →   tool-calling         →   tool-calling      →   SDK
(no tools)             (queries pre-written)    (Claude drives all)   (framework)

data-incident-agent    volve-drilling-advisor   tool-calling-dq-agent  THIS REPO
```

Each step adds autonomy. Strands doesn't change where this sits on the autonomy spectrum — it changes how much infrastructure you write to get there.
## License & Attribution

This project was independently developed by **Venkat Chittoor** on personal time,
using personal resources, and is not affiliated with or owned by any employer
or client organization.

© 2025 Venkat Chittoor. Licensed under
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:** view, learn from, share, and adapt this work for
non-commercial purposes with attribution.

**Commercial use** — including use in client demonstrations, sales engagements,
consulting deliverables, or any revenue-generating activity — requires explicit
written permission from the author.

For commercial licensing inquiries: venkat.chittoor24@gmail.com
