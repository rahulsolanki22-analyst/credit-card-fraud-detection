import streamlit as st
import joblib
import numpy as np
import os

# -------- Load model safely --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

# -------- Initialize session state --------
if "time" not in st.session_state:
    st.session_state.time = 10000.0

if "amount" not in st.session_state:
    st.session_state.amount = 100.0

for i in range(28):
    if f"v_{i}" not in st.session_state:
        st.session_state[f"v_{i}"] = 0.0

# -------- Callback for demo fraud --------
def load_fraud_sample():
    st.session_state.time = 50000.0
    st.session_state.amount = 2500.0

    fraud_vals = {
        0: -3.5,   # V1
        2: -4.2,   # V3
        3: 3.1,    # V4
        9: -2.8,   # V10
        11: -3.0,  # V12
        13: -4.1,  # V14
        16: -2.6   # V17
    }

    for i in range(28):
        st.session_state[f"v_{i}"] = fraud_vals.get(i, 0.0)

# -------- Page config --------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -------- Header --------
st.markdown(
    """
    <h1 style="text-align:center;">💳 Credit Card Fraud Detection</h1>
    <p style="text-align:center; color: gray;">
    Machine Learning powered fraud detection system
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

st.info(
    "This is a demo application. Transaction features are anonymized using PCA, "
    "so values are for demonstration purposes only."
)

# -------- Demo Mode Toggle --------
st.subheader("Demo Mode")

demo_mode = st.toggle(
    "Enable Demo Fraud Mode",
    value=True,
    help="For demonstration purposes only"
)

# -------- Inputs --------
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    st.number_input(
        "Transaction Time (seconds)",
        min_value=0.0,
        key="time"
    )

with col2:
    st.number_input(
        "Transaction Amount",
        min_value=0.0,
        key="amount"
    )

# -------- Quick Demo --------
st.subheader("Quick Demo")

st.button(
    "⚡ Load Sample Fraud Transaction",
    on_click=load_fraud_sample
)

# -------- Advanced PCA --------
with st.expander("Advanced: Transaction Behavior (PCA Features)"):
    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            st.number_input(
                f"V{i+1}",
                step=0.1,
                key=f"v_{i}"
            )

# -------- Prediction --------
st.divider()
st.subheader("Run Prediction")

if st.button("🔍 Analyze Transaction", use_container_width=True):

    v_features = [st.session_state[f"v_{i}"] for i in range(28)]

    input_data = np.array([[
        st.session_state.time,
        st.session_state.amount,
        *v_features
    ]])

    # -------- DEMO MODE (FORCED FRAUD) --------
    if demo_mode:
        prediction = 1
        prob = 0.93

        st.subheader("Prediction Result")
        st.error("🚨 Fraudulent Transaction Detected")
        st.progress(prob)
        st.metric("Fraud Probability", f"{prob:.2%}")
        st.caption("Demo Mode enabled: result shown for demonstration purposes.")

    # -------- REAL MODEL MODE --------
    else:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction Detected")
            st.progress(min(prob, 1.0))
            st.metric("Fraud Probability", f"{prob:.2%}")
        else:
            st.success("✅ Normal Transaction")
            st.progress(min(1 - prob, 1.0))
            st.metric("Confidence", f"{(1 - prob):.2%}")

# -------- Footer --------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size: 12px; color: gray;">
    Built using Machine Learning · Random Forest Model
    </p>
    """,
    unsafe_allow_html=True
)
