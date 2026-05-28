# Dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from database.connection import SessionLocal
from tracking.metrics_tracker import get_full_metrics
from database.models import Query, Evaluation

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────

st.set_page_config(
    page_title="Adaptive AI Orchestration Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────
# MODEL DISPLAY NAMES
# ──────────────────────────────────────────

MODEL_DISPLAY = {
    "nova-micro": "Nova Micro",
    "llama3-8b":  "Llama 3.1 8B",
    "haiku":      "Llama 3.3 70B",
    "mistral":    "Mistral (legacy)",
    "pending":    "Pending (legacy)"
}

MODEL_COLORS = {
    "Nova Micro":    "#00C9A7",
    "Llama 3.1 8B":  "#845EC2",
    "Llama 3.3 70B": "#FF6F91",
}

def display(name):
    return MODEL_DISPLAY.get(name, name)


# ──────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────

@st.cache_data(ttl=10)
def load_metrics():
    db = SessionLocal()
    try:
        return get_full_metrics(db)
    finally:
        db.close()

@st.cache_data(ttl=10)
def load_recent_queries(limit=20):
    db = SessionLocal()
    try:
        rows = (
            db.query(Query)
            .order_by(Query.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":            r.id,
                "query":         r.query_text[:60] + "..." if len(r.query_text) > 60 else r.query_text,
                "model":         r.model_used,
                "complexity":    r.complexity,
                "strategy":      r.strategy,
                "latency_ms":    r.latency_ms,
                "total_tokens":  r.total_tokens,
                "cost_usd":      round(r.estimated_cost, 6) if r.estimated_cost else None,
                "fallback":      r.fallback_used,
                "created_at":    r.created_at,
            }
            for r in rows
        ]
    finally:
        db.close()

@st.cache_data(ttl=10)
def load_quality_over_time():
    db = SessionLocal()
    try:
        rows = (
            db.query(Query.model_used, Query.created_at, Evaluation.quality_score)
            .join(Evaluation, Evaluation.query_id == Query.id)
            .order_by(Query.created_at.asc())
            .all()
        )
        return [
            {
                "model":         display(r[0]),
                "created_at":    r[1],
                "quality_score": r[2]
            }
            for r in rows
        ]
    finally:
        db.close()


# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────

with st.sidebar:
    st.title("🧠 AI Orchestrator")
    st.caption("Adaptive LLM Routing System")
    st.divider()

    auto_refresh = st.toggle("Auto Refresh (10s)", value=False)
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**Models**")
    st.markdown("🟢 `Nova Micro` — fast, low")
    st.markdown("🟡 `Llama 3.1 8B` — reasoning, medium")
    st.markdown("🔴 `Llama 3.3 70B` — complex, high")
    st.divider()
    st.caption("Data refreshes every 10s when auto-refresh is on.")

if auto_refresh:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()


# ──────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────

metrics = load_metrics()
queries = load_recent_queries()
quality = load_quality_over_time()

qpm   = metrics["queries_per_model"]
aqpm  = metrics["avg_quality_per_model"]
alpm  = metrics["avg_latency_per_model"]
strat = metrics["strategy_breakdown"]
comp  = metrics["complexity_breakdown"]
prob  = metrics["probability_table"]
tok_g = metrics["token_stats_global"]       # NEW
tok_m = metrics["token_stats_per_model"]    # NEW

total_queries = sum(qpm.values())
avg_quality   = (
    sum(v["avg_quality"] * v["eval_count"] for v in aqpm.values())
    / sum(v["eval_count"] for v in aqpm.values())
    if aqpm else 0
)
best_model  = max(qpm, key=qpm.get) if qpm else "—"
avg_latency = (
    sum(v["avg_ms"] for v in alpm.values()) / len(alpm)
    if alpm else 0
)


# ──────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────

st.title("🧠 Adaptive AI Orchestration Dashboard")
st.caption("Live system metrics — routing decisions, model performance, quality trends")
st.divider()


# ──────────────────────────────────────────
# ROW 1 — KPI CARDS (performance)
# ──────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Queries",     total_queries)
c2.metric("Avg Quality Score", f"{avg_quality:.3f}" if avg_quality else "—")
c3.metric("Most Used Model",   display(best_model))
c4.metric("Avg Latency",       f"{avg_latency:.0f} ms" if avg_latency else "—")


# ──────────────────────────────────────────
# ROW 2 — KPI CARDS (token / cost)  NEW
# ──────────────────────────────────────────

st.caption("Token & cost summary")
t1, t2, t3, t4 = st.columns(4)

t1.metric(
    "Total Tokens Used",
    f"{tok_g['total_tokens']:,}" if tok_g["total_tokens"] else "—"
)
t2.metric(
    "Avg Tokens / Query",
    f"{tok_g['avg_tokens_per_query']:.0f}" if tok_g["avg_tokens_per_query"] else "—"
)
t3.metric(
    "Total Estimated Cost",
    f"${tok_g['total_estimated_cost']:.4f}" if tok_g["total_estimated_cost"] else "—"
)

# Cost efficiency: cost per quality point (lower = better)
if tok_g["total_estimated_cost"] and avg_quality:
    cost_per_quality = tok_g["total_estimated_cost"] / avg_quality
    t4.metric("Cost / Quality Point", f"${cost_per_quality:.4f}")
else:
    t4.metric("Cost / Quality Point", "—")

st.divider()


# ──────────────────────────────────────────
# ROW 3 — ROUTING + COMPLEXITY
# ──────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Queries per Model")
    if qpm:
        clean_qpm = {k: v for k, v in qpm.items() if k not in ("pending", "mistral")}
        display_keys = [display(k) for k in clean_qpm.keys()]
        fig = px.bar(
            x=display_keys, y=list(clean_qpm.values()),
            labels={"x": "Model", "y": "Queries"},
            color=display_keys,
            color_discrete_map=MODEL_COLORS,
            text=list(clean_qpm.values())
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No query data yet.")

with col2:
    st.subheader("🎯 Complexity Breakdown")
    if comp:
        clean_comp = {k: v for k, v in comp.items() if k != "unknown"}
        fig = px.pie(
            names=list(clean_comp.keys()),
            values=list(clean_comp.values()),
            color=list(clean_comp.keys()),
            color_discrete_map={"low": "#00C9A7", "medium": "#FFC75F", "high": "#FF6F91"},
            hole=0.4
        )
        fig.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No complexity data yet.")


# ──────────────────────────────────────────
# ROW 4 — QUALITY + LATENCY
# ──────────────────────────────────────────

col3, col4 = st.columns(2)

with col3:
    st.subheader("⭐ Avg Quality per Model")
    if aqpm:
        raw_models = list(aqpm.keys())
        models     = [display(m) for m in raw_models]
        scores     = [aqpm[m]["avg_quality"] for m in raw_models]
        counts     = [aqpm[m]["eval_count"]  for m in raw_models]
        fig = go.Figure(go.Bar(
            x=models, y=scores,
            text=[f"{s:.3f} ({c} evals)" for s, c in zip(scores, counts)],
            textposition="outside",
            marker_color=[MODEL_COLORS.get(m, "#aaaaaa") for m in models]
        ))
        fig.update_layout(yaxis=dict(range=[0, 1.1]), height=300,
                          margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No evaluation data yet.")

with col4:
    st.subheader("⚡ Avg Latency per Model (ms)")
    if alpm:
        clean_alpm = {k: v for k, v in alpm.items() if k not in ("pending", "mistral")}
        models = [display(k) for k in clean_alpm.keys()]
        avgs   = [clean_alpm[k]["avg_ms"] for k in clean_alpm.keys()]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Avg", x=models, y=avgs,
            marker_color=[MODEL_COLORS.get(m, "#aaaaaa") for m in models],
            text=[f"{v:.0f}" for v in avgs],
            textposition="outside"
        ))
        fig.update_layout(height=300, margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No latency data yet.")


# ──────────────────────────────────────────
# ROW 5 — TOKEN USAGE + COST PER MODEL  NEW
# ──────────────────────────────────────────

col5, col6 = st.columns(2)

with col5:
    st.subheader("🪙 Token Usage per Model")
    if tok_m:
        df_tok = pd.DataFrame(tok_m)
        df_tok["model"] = df_tok["model"].apply(display)
        df_tok = df_tok[df_tok["model"] != "Pending (legacy)"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Input tokens",
            x=df_tok["model"],
            y=df_tok["input_tokens"],
            marker_color="#93C5FD",
            text=df_tok["input_tokens"],
            textposition="inside"
        ))
        fig.add_trace(go.Bar(
            name="Output tokens",
            x=df_tok["model"],
            y=df_tok["output_tokens"],
            marker_color="#6EE7B7",
            text=df_tok["output_tokens"],
            textposition="inside"
        ))
        fig.update_layout(
            barmode="stack",
            height=300,
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No token data yet.")

with col6:
    st.subheader("💰 Estimated Cost per Model (USD)")
    if tok_m:
        df_cost = pd.DataFrame(tok_m)
        df_cost["model"] = df_cost["model"].apply(display)
        df_cost = df_cost[df_cost["model"] != "Pending (legacy)"]
        fig = px.bar(
            df_cost, x="model", y="total_cost",
            color="model",
            color_discrete_map=MODEL_COLORS,
            text=df_cost["total_cost"].apply(lambda x: f"${x:.5f}"),
            labels={"model": "Model", "total_cost": "Cost (USD)"}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cost data yet.")


# ──────────────────────────────────────────
# ROW 6 — QUALITY TREND OVER TIME
# ──────────────────────────────────────────

st.subheader("📈 Quality Score Over Time")

if quality:
    df_q = pd.DataFrame(quality)
    df_q["created_at"] = pd.to_datetime(df_q["created_at"])
    fig = px.line(
        df_q, x="created_at", y="quality_score", color="model",
        markers=True,
        labels={"created_at": "Time", "quality_score": "Quality Score", "model": "Model"},
        color_discrete_map=MODEL_COLORS
    )
    fig.update_layout(yaxis=dict(range=[0, 1.1]), height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No evaluation history yet.")

st.divider()


# ──────────────────────────────────────────
# ROW 7 — STRATEGY BREAKDOWN + PROBABILITY TABLE
# ──────────────────────────────────────────

col7, col8 = st.columns(2)

with col7:
    st.subheader("🗺️ Strategy Usage")
    if strat:
        clean_strat = {k: v for k, v in strat.items() if k != "pending"}
        df_s = pd.DataFrame([
            {"strategy": k, "count": v["count"], "percent": v["percent"]}
            for k, v in clean_strat.items()
        ])
        fig = px.bar(
            df_s, x="strategy", y="count",
            text=[f"{v['percent']}%" for v in clean_strat.values()],
            labels={"strategy": "Strategy", "count": "Count"},
            color="strategy"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

with col8:
    st.subheader("🎲 Probability Routing Table")
    if prob:
        df_p = pd.DataFrame(prob)
        df_p["model"] = df_p["model"].apply(display)
        st.dataframe(
            df_p.style.background_gradient(
                subset=["p_quality", "p_latency"],
                cmap="RdYlGn"
            ).format({
                "p_quality": "{:.4f}",
                "p_latency": "{:.4f}",
                "p_cost":    "{:.4f}"
            }),
            use_container_width=True,
            height=300
        )

st.divider()


# ──────────────────────────────────────────
# ROW 8 — LIVE RECENT QUERIES TABLE
# ──────────────────────────────────────────

st.subheader("🔴 Live Routing Decisions")
st.caption("Last 20 queries — most recent first. Token and cost columns show per-query usage.")

if queries:
    df = pd.DataFrame(queries)
    df["model"]      = df["model"].apply(display)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%H:%M:%S")
    df["fallback"]   = df["fallback"].map({True: "⚠️ Yes", False: "✅ No"})
    df["cost_usd"]   = df["cost_usd"].apply(
        lambda x: f"${x:.5f}" if x is not None else "—"
    )
    df["total_tokens"] = df["total_tokens"].apply(
        lambda x: f"{x:,}" if x is not None else "—"
    )

    def colour_model(val):
        colours = {
            "Nova Micro":    "background-color: #d4f7ef",
            "Llama 3.1 8B":  "background-color: #e8d4f7",
            "Llama 3.3 70B": "background-color: #f7d4df",
        }
        return colours.get(val, "")

    st.dataframe(
        df[[
            "id", "query", "model", "complexity", "strategy",
            "latency_ms", "total_tokens", "cost_usd",   # token + cost columns (NEW)
            "fallback", "created_at"
        ]].style.map(colour_model, subset=["model"]),
        use_container_width=True,
        height=420
    )
else:
    st.info("No queries yet. Send a request to /api/chat to see live routing.")
