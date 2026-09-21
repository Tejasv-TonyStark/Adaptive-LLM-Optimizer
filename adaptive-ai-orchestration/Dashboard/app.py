"""Admin-only dashboard through the API; no direct database credentials."""
import os
import httpx
import pandas as pd
import streamlit as st
API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="LLM Router Metrics", layout="wide")
st.title("LLM Router Metrics")
st.caption("Administrative aggregates. Evaluation quality reflects sampled, judged answers.")
if "token" not in st.session_state:
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")
    if submit:
        try:
            response = httpx.post(API+"/api/auth/login", json=dict(username=username,password=password), timeout=10)
            response.raise_for_status()
            st.session_state.token = response.json()["access_token"]
            st.rerun()
        except httpx.HTTPError:
            st.error("Sign-in failed. Check credentials and API availability.")
    st.stop()
if st.sidebar.button("Sign out"):
    del st.session_state.token
    st.rerun()
@st.fragment(run_every="10s")
def metrics_panel():
    headers = {"Authorization": "Bearer "+st.session_state.token}
    try:
        response = httpx.get(API+"/api/metrics", headers=headers, timeout=10)
        if response.status_code in (401,403):
            st.error("An active administrator account is required. Configure ADMIN_USERNAMES on the API.")
            return
        response.raise_for_status()
        metrics = response.json()
        probs = httpx.get(API+"/api/probabilities", headers=headers, timeout=10)
        probs.raise_for_status()
    except httpx.HTTPError:
        st.error("Metrics unavailable.")
        return
    a,b,c = st.columns(3)
    a.metric("Completed / abstained queries", metrics["total_queries"])
    b.metric("Mean request processing latency", f'{metrics["average_latency_ms"]:.0f} ms')
    c.metric("Mean judged quality", f'{metrics["average_quality_score"]:.3f}')
    st.subheader("Estimated usage cost by stage")
    st.bar_chart(metrics["cost_by_stage"])
    st.caption(f'Unknown-cost provider attempts: {metrics["unknown_cost_attempts"]}. '
               'Totals exclude ingestion and unrecorded calls interrupted by process crashes.')
    st.subheader("Responses by model")
    st.bar_chart(metrics["model_breakdown"])
    st.subheader("Online routing metrics")
    st.dataframe(pd.DataFrame(probs.json()), hide_index=True, use_container_width=True)
    st.caption("p_cost is retained for legacy compatibility; routing uses estimated request token cost.")
metrics_panel()
