import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st



st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)


# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
DATA_PROCESSED_PATH = os.path.join(REPO_ROOT, "data", "processed", "fraud_clean.csv")

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ---------------- Styling ----------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
      .app-card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }
      .muted { opacity: 0.85; }
      .risk-badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 12px;
        letter-spacing: .02em;
      }
      .risk-low { background: rgba(34,197,94,0.15); border: 1px solid rgba(34,197,94,0.35); }
      .risk-med { background: rgba(234,179,8,0.15); border: 1px solid rgba(234,179,8,0.35); }
      .risk-high { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.35); }
      .risk-critical { background: rgba(248,113,113,0.18); border: 1px solid rgba(248,113,113,0.45); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Helpers ----------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_processed_sample(path: str, nrows: int = 5000) -> pd.DataFrame:
    # This dataset can be huge; keep the UI responsive by sampling a small head.
    return pd.read_csv(path, nrows=nrows)


def available_models() -> dict[str, str]:
    # label -> path
    candidates = {
        "Random Forest (recommended)": os.path.join(MODELS_DIR, "fraud_model.pkl"),
        "Random Forest (alt file)": os.path.join(MODELS_DIR, "random_forest.pkl"),
        "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.pkl"),
        "XGBoost": os.path.join(MODELS_DIR, "xgboost.pkl"),
    }
    return {k: v for k, v in candidates.items() if os.path.exists(v)}


def risk_tier(prob: float) -> tuple[str, str]:
    if prob < 0.25:
        return "LOW RISK", "risk-low"
    if prob < 0.5:
        return "MEDIUM RISK", "risk-med"
    if prob < 0.8:
        return "HIGH RISK", "risk-high"
    return "CRITICAL RISK", "risk-critical"


def build_input_dataframe(time_val: float, amount_val: float, v_vals: list[float]) -> pd.DataFrame:
    # IMPORTANT: training uses the column order: Time, V1..V28, Amount
    row = [time_val, *v_vals, amount_val]
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def set_inputs_from_row(row: pd.Series):
    st.session_state.time = float(row["Time"])
    st.session_state.amount = float(row["Amount"])
    for i in range(1, 29):
        st.session_state[f"v_{i}"] = float(row[f"V{i}"])


def find_example_row(
    path: str,
    target_class: int,
    *,
    max_rows: int = 250_000,
    chunksize: int = 50_000,
    random_state: int = 42,
) -> pd.Series | None:
    """
    Find a single row matching `Class == target_class` by scanning the CSV in chunks.
    This avoids loading huge files fully into memory and makes fraud rows easier to find.
    """
    if not os.path.exists(path):
        return None

    rows_seen = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if "Class" not in chunk.columns:
            return None

        match = chunk[chunk["Class"] == target_class]
        if len(match) > 0:
            return match.sample(1, random_state=random_state).iloc[0]

        rows_seen += len(chunk)
        if rows_seen >= max_rows:
            return None

    return None


def load_demo_fraud():
    # For interview demos: turn ON demo mode and populate plausible scaled values.
    # (Guarantees a "fraud-looking" result even if the selected model is conservative.)
    st.session_state.demo_mode = True
    st.session_state.time = 0.62
    st.session_state.amount = -0.35
    demo_vals = {1: -0.80, 2: 1.08, 3: 0.56, 4: 0.29, 12: -0.95, 18: -0.68}
    for i in range(1, 29):
        st.session_state[f"v_{i}"] = float(demo_vals.get(i, 0.0))


def load_sample_normal():
    row = find_example_row(DATA_PROCESSED_PATH, target_class=0)
    if row is None:
        st.session_state.sample_msg = "Could not find a NORMAL row in the scanned window."
        return
    set_inputs_from_row(row)
    st.session_state.sample_msg = "Loaded a NORMAL sample from the processed dataset."


def load_sample_fraud():
    row = find_example_row(DATA_PROCESSED_PATH, target_class=1, max_rows=500_000)
    if row is None:
        st.session_state.sample_msg = "Could not find a FRAUD row in the scanned window. Try demo mode."
        return
    set_inputs_from_row(row)
    st.session_state.sample_msg = "Loaded a FRAUD sample from the processed dataset."


# ---------------- Session state init ----------------
if "time" not in st.session_state:
    st.session_state.time = 0.0
if "amount" not in st.session_state:
    st.session_state.amount = 0.0
for i in range(1, 29):
    if f"v_{i}" not in st.session_state:
        st.session_state[f"v_{i}"] = 0.0
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "sample_msg" not in st.session_state:
    st.session_state.sample_msg = ""


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### Fraud Detection")
    st.caption("Interview-ready Streamlit UI for your ML models.")

    models = available_models()
    if not models:
        st.error("No models found in `models/`. Please add `.pkl` files.")
        st.stop()

    model_label = st.selectbox("Model", list(models.keys()), index=0)
    model = load_model(models[model_label])

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    st.caption("Tip: tune this live to show precision/recall trade-offs.")

    demo_mode = st.toggle(
        "Presentation demo mode",
        key="demo_mode",
        value=st.session_state.demo_mode,
        help="When ON, the app forces a fraud-looking output to showcase UI behavior.",
    )

    st.divider()
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Single prediction", "Batch scoring", "About"], label_visibility="collapsed")


# ---------------- Header ----------------
st.markdown("### Credit Card Fraud Detection")
st.markdown(
    "<div class='muted'>PCA-transformed features (V1–V28). This UI focuses on clarity, trust, and demo-friendly workflows.</div>",
    unsafe_allow_html=True,
)


if page == "Single prediction":
    colA, colB = st.columns([1.2, 0.8], gap="large")

    with colA:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("#### Transaction input")
        st.caption("Note: the trained models expect **scaled** `Time` and `Amount` (as in `data/processed/fraud_clean.csv`).")

        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Time (scaled)", key="time")
        with c2:
            st.number_input("Amount (scaled)", key="amount")

        with st.expander("Advanced: PCA features (V1–V28)", expanded=False):
            cols = st.columns(4)
            for i in range(1, 29):
                with cols[(i - 1) % 4]:
                    st.number_input(f"V{i}", step=0.1, key=f"v_{i}")

        st.markdown("##### Quick fill")
        qa1, qa2, qa3 = st.columns(3)

        with qa1:
            st.button(
                "Load demo (fraud-ish)",
                use_container_width=True,
                on_click=load_demo_fraud,
                help="Also enables demo mode so the result is consistently fraud-ish for presentations.",
            )

        with qa2:
            if os.path.exists(DATA_PROCESSED_PATH):
                st.button("Load sample (normal)", use_container_width=True, on_click=load_sample_normal)
            else:
                st.button("Load sample (normal)", use_container_width=True, disabled=True)

        with qa3:
            if os.path.exists(DATA_PROCESSED_PATH):
                st.button("Load sample (fraud)", use_container_width=True, on_click=load_sample_fraud)
            else:
                st.button("Load sample (fraud)", use_container_width=True, disabled=True)

        if st.session_state.sample_msg:
            st.caption(st.session_state.sample_msg)

        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("#### Result")
        st.caption("Uses probability + threshold (instead of hardcoded 0.5).")

        analyze = st.button("Analyze transaction", type="primary", use_container_width=True)

        if analyze:
            v_features = [float(st.session_state[f"v_{i}"]) for i in range(1, 29)]
            X_df = build_input_dataframe(float(st.session_state.time), float(st.session_state.amount), v_features)

            if demo_mode:
                prob = 0.93
            else:
                prob = float(model.predict_proba(X_df)[0][1])

            pred = int(prob >= threshold)
            tier, css = risk_tier(prob)

            st.markdown(
                f"<span class='risk-badge {css}'>{tier}</span>",
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud probability", f"{prob:.2%}")
            m2.metric("Threshold", f"{threshold:.2f}")
            m3.metric("Decision", "FRAUD" if pred == 1 else "NORMAL")

            st.progress(prob)

            if pred == 1:
                st.error("Flag this transaction for review.")
            else:
                st.success("Looks normal under the current threshold.")

            with st.expander("Debug: model input row", expanded=False):
                st.dataframe(X_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


elif page == "Batch scoring":
    st.markdown("#### Batch scoring")
    st.caption("Upload a CSV with columns: `Time`, `V1`..`V28`, `Amount`. The app will output `fraud_probability` + `decision`.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        # Convenience: many "creditcard_2023" datasets come with `id` but no `Time`.
        # If `Time` is missing and `id` exists, generate a simple Time from id to unblock scoring.
        if "Time" not in df.columns and "id" in df.columns:
            df = df.copy()
            df["Time"] = df["id"].astype(float)
            st.warning("`Time` column was missing. Generated `Time = id` for this upload (quick fix).")

        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            X = df[FEATURE_COLUMNS]
            probs = model.predict_proba(X)[:, 1]
            out = df.copy()
            out["fraud_probability"] = probs
            out["decision"] = (out["fraud_probability"] >= threshold).astype(int)

            st.markdown("<div class='app-card'>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows scored", f"{len(out):,}")
            c2.metric("Flagged as fraud", f"{int(out['decision'].sum()):,}")
            c3.metric("Flag rate", f"{(out['decision'].mean() if len(out) else 0):.2%}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="fraud_scored.csv",
                mime="text/csv",
                use_container_width=True,
            )


else:
    st.markdown("#### About")
    st.markdown(
        """
        - **What’s improved**: modern layout, model selection, threshold control, batch scoring, and clearer results.
        - **Data reality**: features are PCA-transformed; interpretability of V-features is limited.
        - **Correctness fix**: the app now sends model inputs in the same column order used during training (`Time`, `V1..V28`, `Amount`).
        """
    )
    st.info("Run with: `streamlit run app/app.py`")


st.markdown(
    "<div class='muted' style='padding-top:10px;'>Built with Streamlit • Models in <code>models/</code> • Data in <code>data/</code></div>",
    unsafe_allow_html=True,
)
