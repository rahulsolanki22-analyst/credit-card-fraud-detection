import streamlit as st
import joblib
import numpy as np
import os

# ---------------- Load model ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fraud_model.pkl")
model = joblib.load(MODEL_PATH)

# ---------------- Session state init ----------------
if "time" not in st.session_state:
    st.session_state.time = 10000.0

if "amount" not in st.session_state:
    st.session_state.amount = 100.0

for i in range(28):
    if f"v_{i}" not in st.session_state:
        st.session_state[f"v_{i}"] = 0.0

# ---------------- Demo fraud callback ----------------
def load_demo_fraud():
    st.session_state.time = 50000.0
    st.session_state.amount = 2500.0

    demo_vals = {
        0: -3.5,
        2: -4.2,
        3: 3.1,
        9: -2.8,
        11: -3.0,
        13: -4.1,
        16: -2.6
    }

    for i in range(28):
        st.session_state[f"v_{i}"] = demo_vals.get(i, 0.0)

# ---------------- REAL fraud callback (from dataset) ----------------
def load_real_fraud():
    # Time and Amount from real fraud row
    st.session_state.time = 0.06594888799996523
    st.session_state.amount = -0.3488333439567424

    real_vals = [
        -0.807297894052063,
        1.0855890339441092,
        0.5654028914496401,
        0.29184261292639796,
        1.2828301340994699,
        0.11425205553215398,
        -0.06178675922282317,
        -0.10181135853785062,
        -0.6277644799303883,
        -0.30031714515073826,
        0.025665748824022228,
        -0.9528308629362598,
        0.3893629791211495,
        1.0466089845185371,
        0.19056123564725924,
        1.0900463695826024,
        0.35398511758247453,
        -0.6787546815036027,
        -0.06430188522175172,
        0.011048972105640875,
        0.09997929701024322,
        -0.3450727086399884,
        -0.14158779402106844,
        -0.09889536471698598,
        -0.5121521756689156,
        0.6196967155974751,
        -0.2794539985068841,
        0.05245093821680197
    ]

    for i in range(28):
        st.session_state[f"v_{i}"] = real_vals[i]

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="centered"
)

# ---------------- UI ----------------
st.markdown(
    "<h1 style='text-align:center;'>Credit Card Fraud Detection</h1>",
    unsafe_allow_html=True
)

st.info(
    "This is a demo application. Features are PCA-transformed, "
    "so inputs are for demonstration and learning purposes."
)

# ---------------- Demo mode ----------------
st.subheader("Demo Mode")

demo_mode = st.toggle(
    "Enable Demo Fraud Mode",
    value=True,
    help="When ON, fraud result is forced for demo"
)

# ---------------- Transaction inputs ----------------
st.subheader("Transaction Details")

col1, col2 = st.columns(2)

with col1:
    st.number_input(
        "Transaction Time",
        min_value=0.0,
        key="time"
    )

with col2:
    st.number_input(
        "Transaction Amount",
        key="amount"
    )

# ---------------- Quick actions ----------------
st.subheader("Quick Actions")


# ---------------- PCA inputs ----------------
with st.expander("Advanced: PCA Features (V1 to V28)"):
    cols = st.columns(4)
    for i in range(28):
        with cols[i % 4]:
            st.number_input(
                f"V{i+1}",
                step=0.1,
                key=f"v_{i}"
            )

# ---------------- Prediction ----------------
st.divider()
st.subheader("Prediction")

if st.button("Analyze Transaction", use_container_width=True):

    v_features = [st.session_state[f"v_{i}"] for i in range(28)]

    input_data = np.array([[
        st.session_state.time,
        st.session_state.amount,
        *v_features
    ]])

    if demo_mode:
        prob = 0.93
        st.error("Fraudulent Transaction Detected")
        st.progress(prob)
        st.metric("Fraud Probability", f"{prob:.2%}")
        st.caption("Demo Mode is ON (forced result).")

    else:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("Fraudulent Transaction Detected")
            st.progress(min(prob, 1.0))
            st.metric("Fraud Probability", f"{prob:.2%}")
        else:
            st.success("Normal Transaction")
            st.progress(min(1 - prob, 1.0))
            st.metric("Confidence", f"{(1 - prob):.2%}")

# ---------------- Footer ----------------
st.markdown(
    "<hr><p style='text-align:center;'>Random Forest Model · ML Demo App</p>",
    unsafe_allow_html=True
)
