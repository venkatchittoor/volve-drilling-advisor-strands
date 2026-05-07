"""
Volve Drilling Advisor — Strands Edition
Well 15/9-F-15 | Volve Field | Norwegian Continental Shelf

Rebuilt on AWS Strands Agents SDK — same O&G domain expertise,
same 4 tools, dramatically less boilerplate than the raw Anthropic API version.

Architecture comparison:
  Raw API version : ~200 lines of agent loop, dispatcher, schema definitions
  Strands version : @tool decorator + Agent() — framework handles the rest

Data sources (local CSV — no Databricks required):
  - data/ROP data.csv                   : F-15 real drilling parameters
  - data/silver_formation_tops.csv      : Formation tops from offset well crew
  - data/silver_reservoir_flags.csv     : HC potential flags from offset well crew
  - data/silver_drillability_forecast.csv: Drillability forecast from offset well crew
"""

import os
import pandas as pd
from strands import Agent, tool
from strands.models.anthropic import AnthropicModel
from dotenv import load_dotenv
load_dotenv()


WINDOW_SIZE  = 5    # rows per depth window (25m)
STEP_SIZE    = 10   # rows to advance between windows (~50m)
MAX_WINDOWS  = 13   # maximum windows to process

# Key depth intervals — same as original project
FOCUS_DEPTHS = [3305, 3350, 3400, 3500, 3600, 3650, 3700,
                3750, 3800, 3850, 3900, 3950, 4000, 4050, 4085]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading Volve field data...")

df_drilling = pd.read_csv("data/ROP data.csv")
df_drilling["ROP_mhr"]      = df_drilling["ROP_AVG"] * 3600
df_drilling["MSE_proxy"]    = df_drilling["WOB"] * df_drilling["SURF_RPM"] / df_drilling["ROP_mhr"].clip(lower=0.001)
df_drilling["Torque_est"]   = df_drilling["WOB"] * 0.3 / df_drilling["SURF_RPM"].clip(lower=0.001)
df_drilling["ROP_rolling"]  = df_drilling["ROP_mhr"].rolling(window=5, min_periods=1).mean()
df_drilling["ROP_drop_flag"]= df_drilling["ROP_mhr"] < df_drilling["ROP_rolling"] * 0.7
df_drilling = df_drilling.sort_values("Depth").reset_index(drop=True)

df_tops      = pd.read_csv("data/silver_formation_tops.csv")
df_flags     = pd.read_csv("data/silver_reservoir_flags.csv")
df_drill_fct = pd.read_csv("data/silver_drillability_forecast.csv")

print(f"F-15 drilling data: {len(df_drilling)} rows | Depth {df_drilling['Depth'].min():.0f}–{df_drilling['Depth'].max():.0f}m")
print(f"Formation tops: {len(df_tops)} rows")
print(f"Reservoir flags: {len(df_flags)} rows")
print(f"Drillability forecast: {len(df_drill_fct)} rows")

# ── Streaming window helper ───────────────────────────────────────────────────
def get_window(window_index: int) -> dict | None:
    """Return drilling parameter window at given index."""
    start = window_index * STEP_SIZE
    end   = start + WINDOW_SIZE
    if start >= len(df_drilling):
        return None
    w = df_drilling.iloc[start:min(end, len(df_drilling))]
    return {
        "window_index":  window_index,
        "current_depth": float(w["Depth"].iloc[-1]),
        "depth_from":    float(w["Depth"].iloc[0]),
        "depth_to":      float(w["Depth"].iloc[-1]),
        "WOB_mean_N":    round(float(w["WOB"].mean()), 2),
        "WOB_std_N":     round(float(w["WOB"].std()), 2),
        "RPM_mean":      round(float(w["SURF_RPM"].mean()), 3),
        "ROP_mhr_mean":  round(float(w["ROP_mhr"].mean()), 2),
        "ROP_mhr_min":   round(float(w["ROP_mhr"].min()), 2),
        "MSE_proxy_mean":round(float(w["MSE_proxy"].mean()), 2),
        "PHIF_mean":     round(float(w["PHIF"].mean()), 4),
        "VSH_mean":      round(float(w["VSH"].mean()), 4),
        "SW_mean":       round(float(w["SW"].mean()), 4),
        "ROP_drop_flag": bool(w["ROP_drop_flag"].any()),
    }

# ── Tools — @tool decorator replaces JSON schema + dispatcher ─────────────────
@tool
def get_formation_context(depth_m: float) -> dict:
    """
    Get formation context from offset well Silver tables at current depth.
    Returns formation position (Draupne/Hugin/below), nearby HC flags,
    and reservoir quality information. Use this to understand what formation
    the bit is drilling through and what HC potential exists nearby.
    """
    # Formation position
    position = "ABOVE_DRAUPNE"
    for _, row in df_tops.iterrows():
        if row["formation"] == "DRAUPNE" and depth_m >= row["picked_depth_m"]:
            position = "IN_DRAUPNE"
        if row["formation"] == "HUGIN_TOP" and depth_m >= row["picked_depth_m"]:
            position = "IN_HUGIN_RESERVOIR"
        if row["formation"] == "HUGIN_BASE" and depth_m >= row["picked_depth_m"]:
            position = "BELOW_HUGIN"

    # Nearby reservoir flags (±50m)
    nearby = df_flags[
        (df_flags["depth_from_m"] <= depth_m + 50) &
        (df_flags["depth_to_m"]   >= depth_m - 50)
    ].sort_values(by="depth_from_m").head(3)

    return {
        "depth_m":            depth_m,
        "formation_position": position,
        "formation_tops":     df_tops.to_dict(orient="records"),
        "nearby_flags":       nearby[["depth_from_m","depth_to_m","flag_type",
                                      "severity","recommendation"]].to_dict(orient="records"),
    }


@tool
def get_drillability_forecast(depth_m: float) -> dict:
    """
    Get expected drillability at current depth from offset well analog data.
    Returns HARD/MODERATE/SOFT forecast with geological basis.
    Use this to compare actual ROP against what offset wells experienced.
    """
    nearby = df_drill_fct[
        (df_drill_fct["depth_from_m"] <= depth_m + 50) &
        (df_drill_fct["depth_to_m"]   >= depth_m - 50)
    ].sort_values(by="depth_from_m").head(2)

    return {
        "depth_m":  depth_m,
        "forecast": nearby[["depth_from_m","depth_to_m",
                             "expected_drillability","basis"]].to_dict(orient="records"),
        "available": len(nearby) > 0,
    }


@tool
def check_rop_trend(window_index: int, current_depth: float) -> dict:
    """
    Compare current ROP against the 5-window rolling trend.
    Detects ROP drops (>20% below trend) or improvements.
    Use this to determine if an ROP change is formation-driven or bit-related.
    """
    start   = max(0, window_index - 4) * STEP_SIZE
    end     = window_index * STEP_SIZE + WINDOW_SIZE
    recent  = df_drilling.iloc[start:end]
    current = df_drilling.iloc[window_index * STEP_SIZE:
                               window_index * STEP_SIZE + WINDOW_SIZE]

    trend_rop   = round(float(recent["ROP_mhr"].mean()), 2)
    current_rop = round(float(current["ROP_mhr"].mean()), 2)
    pct_change  = round((current_rop - trend_rop) / trend_rop * 100, 1) \
                  if trend_rop > 0 else 0.0

    return {
        "current_depth":   current_depth,
        "current_rop_mhr": current_rop,
        "trend_rop_mhr":   trend_rop,
        "pct_change":      pct_change,
        "assessment":      "ROP_DROP"     if pct_change < -20 else
                           "ROP_INCREASE" if pct_change > 20  else "STABLE",
    }


@tool
def check_mse_efficiency(wob_n: float, rpm: float, rop_mhr: float) -> dict:
    """
    Assess bit efficiency using Mechanical Specific Energy (MSE) proxy.
    MSE_proxy = WOB × RPM / ROP_mhr
    Higher MSE = bit working harder for less penetration = inefficiency.
    Use this to determine if low ROP is a bit issue or a formation issue.
    """
    mse = round(wob_n * rpm / rop_mhr, 2) if rop_mhr > 0 else 999999

    if mse > 50000:
        assessment     = "INEFFICIENT"
        recommendation = "Consider reducing WOB or optimizing RPM"
    elif mse > 30000:
        assessment     = "MODERATE"
        recommendation = "Monitor for trend — acceptable but watch closely"
    else:
        assessment     = "EFFICIENT"
        recommendation = "Parameters within acceptable range"

    return {
        "MSE_proxy":      mse,
        "WOB_N":          wob_n,
        "RPM":            rpm,
        "ROP_mhr":        rop_mhr,
        "assessment":     assessment,
        "recommendation": recommendation,
    }


# ── Strands Agent — replaces the entire manual agent loop ─────────────────────
model = AnthropicModel(
    model_id="claude-sonnet-4-6",
    max_tokens=4096,
)

SYSTEM_PROMPT = """You are an expert drilling advisor with 20 years of North Sea experience
monitoring well 15/9-F-15 in the Volve field, Norwegian Continental Shelf.

You receive real-time drilling parameters window by window and provide
specific, actionable drilling recommendations.

Key field context:
- Hugin reservoir entry at ~3,350m (offset well analog)
- CRITICAL HC potential at 3,350m and 3,700m
- Best confirmed reservoir: 3,800–3,900m (RHOB 2.31–2.32, RT 911–3,322 ohm.m)
- CRITICAL anomaly at 4,000–4,050m (shale where offsets show HC sand)

Use your tools to investigate then provide a concise, depth-referenced advisory.
Focus on: what is happening, why, and what the driller should do."""


# ── Run advisor across focus depth windows ────────────────────────────────────
def run_advisor():
    """Run the drilling advisor across all selected depth windows."""

    # Select windows at focus depths
    selected = []
    for depth in FOCUS_DEPTHS:
        diff    = abs(df_drilling["Depth"] - depth)
        closest = diff.idxmin()
        win_idx = max(0, closest - WINDOW_SIZE // 2) // STEP_SIZE
        if win_idx not in selected:
            selected.append(win_idx)
    selected = sorted(set(selected))[:MAX_WINDOWS]

    print(f"\n{'='*60}")
    print(f"VOLVE DRILLING ADVISOR — Strands Edition")
    print(f"Well: 15/9-F-15 | {len(selected)} depth windows")
    print(f"{'='*60}\n")

    advisories = []

    for i, win_idx in enumerate(selected):
        window = get_window(win_idx)
        if not window:
            continue

        depth = window["current_depth"]
        print(f"[{i+1:02d}/{len(selected)}] Analyzing depth {depth}m...")

        # Single agent call per window — Strands manages the tool loop
        prompt = f"""Analyze drilling conditions at depth {depth}m.

Current parameters:
- Depth interval: {window['depth_from']}–{window['depth_to']}m
- WOB: {window['WOB_mean_N']:,.0f} N (std: {window['WOB_std_N']:,.0f} N)
- RPM: {window['RPM_mean']}
- ROP: {window['ROP_mhr_mean']} m/hr (min: {window['ROP_mhr_min']} m/hr)
- MSE proxy: {window['MSE_proxy_mean']:,.0f}
- PHIF: {window['PHIF_mean']} | VSH: {window['VSH_mean']} | SW: {window['SW_mean']}
- ROP drop flag: {window['ROP_drop_flag']}
- Window index: {win_idx}

Investigate using your tools and provide a specific drilling advisory."""
        agent = Agent(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            tools=[get_formation_context, get_drillability_forecast,
                   check_rop_trend, check_mse_efficiency],
        )
        response = agent(prompt)
        advisory_text = str(response)

        advisories.append({
            "depth_m":  depth,
            "advisory": advisory_text,
            "window":   window,
        })

        print(f"  ✅ Advisory generated\n")

    return advisories


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    advisories = run_advisor()

    print(f"\n{'='*60}")
    print(f"FULL ADVISORIES — Well 15/9-F-15")
    print(f"{'='*60}")

    for adv in advisories:
        print(f"\n{'─'*60}")
        print(f"DEPTH: {adv['depth_m']}m")
        print(f"{'─'*60}")
        print(adv["advisory"])

    # Save to sample output
    os.makedirs("sample_output", exist_ok=True)
    output_path = "sample_output/drilling_advisory_strands.md"
    with open(output_path, "w") as f:
        f.write("# Volve Drilling Advisor — Strands Edition\n")
        f.write(f"**Well:** 15/9-F-15 | **Windows:** {len(advisories)}\n\n---\n\n")
        for adv in advisories:
            f.write(f"## Depth: {adv['depth_m']}m\n\n")
            f.write(adv["advisory"])
            f.write("\n\n---\n\n")

    print(f"\n✅ Report saved to {output_path}")
    print(f"✅ {len(advisories)} advisories generated")
