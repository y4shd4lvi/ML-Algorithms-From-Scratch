"""
app.py
─────────────────────────────────────────────────────────────
Streamlit UI – wires together data_utils, models, and visualization.

Run with:
    streamlit run app.py
"""

import streamlit as st

from data_utils import generate_dataset
from models import Perceptron, train_logistic_regression
from visualization import (
    plot_dataset,
    plot_final_boundaries,
    plot_perceptron_epoch,
    plot_accuracy_comparison,
)

# ──────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Perceptron vs Logistic Regression",
    page_icon="🧠",
    layout="wide",
)

# ──────────────────────────────────────────────────────────
# Custom CSS – dark, polished look
# ──────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0F1117;
        color: #E2E4EF;
    }

    /* ── Header ── */
    .hero {
        background: linear-gradient(135deg, #1A1D27 0%, #111322 100%);
        border: 1px solid #2A2D3A;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, #C77DFF22, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 0.4rem;
        letter-spacing: -1px;
    }
    .hero p {
        font-size: 1rem;
        color: #A8AABD;
        margin: 0;
        max-width: 640px;
        line-height: 1.6;
    }

    /* ── Metric cards ── */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 160px;
        background: #1A1D27;
        border: 1px solid #2A2D3A;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    .metric-card .label { font-size: 0.75rem; color: #A8AABD; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.25rem; }
    .metric-card .value { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #FFFFFF; }
    .metric-card .value.yellow { color: #FFD93D; }
    .metric-card .value.green  { color: #6BCB77; }
    .metric-card .value.purple { color: #C77DFF; }
    .metric-card .value.teal   { color: #4ECDC4; }

    /* ── Section headings ── */
    .section-heading {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        color: #C77DFF;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border-left: 3px solid #C77DFF;
        padding-left: 0.75rem;
        margin: 1.8rem 0 1rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111322 !important;
        border-right: 1px solid #2A2D3A;
    }
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] p {
        color: #A8AABD !important;
        font-size: 0.85rem;
    }

    /* ── Streamlit elements cleanup ── */
    .stButton > button {
        background: #C77DFF;
        color: #0F1117;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.45rem 1.2rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    div[data-testid="stHorizontalBlock"] { gap: 1.2rem; }

    /* make expander dark */
    details { background: #1A1D27; border-radius: 10px; border: 1px solid #2A2D3A; padding: 0.5rem; }
    summary { color: #A8AABD; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>🧠 Perceptron vs Logistic Regression</h1>
        <p>
            Explore how two classic linear classifiers learn on the same 2-D synthetic dataset.
            Adjust the parameters in the sidebar, then step through the Perceptron's
            training history epoch-by-epoch with the interactive slider.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# Sidebar – controls
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**Dataset**")
    n_samples = st.slider("Number of samples", min_value=100, max_value=1000, value=300, step=50)
    class_sep = st.slider("Class separation", min_value=0.3, max_value=3.0, value=1.0, step=0.1,
                          help="Higher = easier to separate classes")
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    st.markdown("---")
    st.markdown("**Perceptron**")
    n_epochs = st.slider("Training epochs", min_value=5, max_value=200, value=30, step=5)
    learning_rate = st.select_slider(
        "Learning rate",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        value=0.1,
    )

    st.markdown("---")
    retrain = st.button("🔄  Retrain Models")

    st.markdown(
        """
        <br>
        <div style='font-size:0.72rem;color:#5A5C7A;line-height:1.6'>
        <b>Algorithms</b><br>
        <b style='color:#FFD93D'>Perceptron</b> – custom from scratch<br>
        <b style='color:#6BCB77'>Logistic Reg.</b> – sklearn
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────
# State management – cache key changes trigger a full retrain
# ──────────────────────────────────────────────────────────

cache_key = (n_samples, class_sep, int(random_state), n_epochs, learning_rate)

if "cache_key" not in st.session_state or st.session_state.cache_key != cache_key or retrain:
    st.session_state.cache_key = cache_key

    with st.spinner("Generating data & training models…"):
        data = generate_dataset(
            n_samples=n_samples,
            class_sep=class_sep,
            random_state=int(random_state),
        )
        perceptron = Perceptron(learning_rate=learning_rate, n_epochs=n_epochs)
        perceptron.fit(data["X_train"], data["y_train"])

        lr_model = train_logistic_regression(data["X_train"], data["y_train"])

    st.session_state.data = data
    st.session_state.perceptron = perceptron
    st.session_state.lr_model = lr_model

data = st.session_state.data
perceptron = st.session_state.perceptron
lr_model = st.session_state.lr_model

# ──────────────────────────────────────────────────────────
# Metric cards
# ──────────────────────────────────────────────────────────

perc_train_acc = perceptron.score(data["X_train"], data["y_train"])
perc_test_acc  = perceptron.score(data["X_test"],  data["y_test"])
lr_train_acc   = lr_model.score(data["X_train"], data["y_train"])
lr_test_acc    = lr_model.score(data["X_test"],  data["y_test"])
final_errors   = perceptron.errors_per_epoch[-1]

st.markdown(
    f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Perceptron Train Acc</div>
            <div class="value yellow">{perc_train_acc:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="label">Perceptron Test Acc</div>
            <div class="value yellow">{perc_test_acc:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="label">LogReg Train Acc</div>
            <div class="value green">{lr_train_acc:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="label">LogReg Test Acc</div>
            <div class="value green">{lr_test_acc:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="label">Train samples</div>
            <div class="value teal">{len(data["X_train"])}</div>
        </div>
        <div class="metric-card">
            <div class="label">Errors @ last epoch</div>
            <div class="value purple">{final_errors}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# Section 1 – Dataset & accuracy comparison
# ──────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">Dataset Overview</div>', unsafe_allow_html=True)

col_data, col_acc = st.columns([1.6, 1])

with col_data:
    fig_data = plot_dataset(data["X_all"], data["y_all"], title="Full Scaled Dataset")
    st.pyplot(fig_data, use_container_width=True)

with col_acc:
    fig_acc = plot_accuracy_comparison(perceptron, lr_model, data["X_test"], data["y_test"])
    st.pyplot(fig_acc, use_container_width=True)

# ──────────────────────────────────────────────────────────
# Section 2 – Final decision boundaries side-by-side
# ──────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">Final Decision Boundaries</div>', unsafe_allow_html=True)

fig_boundaries = plot_final_boundaries(
    data["X_all"], data["y_all"],
    perceptron, lr_model,
    data["feature_range"],
)
st.pyplot(fig_boundaries, use_container_width=True)

# ──────────────────────────────────────────────────────────
# Section 3 – Perceptron learning progression
# ──────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">Perceptron Learning Progression</div>', unsafe_allow_html=True)

st.markdown(
    "<p style='color:#A8AABD;font-size:0.88rem;margin-bottom:0.8rem'>"
    "Drag the slider to replay how the Perceptron's decision boundary evolved "
    "epoch-by-epoch. Faint dashed lines show all prior boundaries."
    "</p>",
    unsafe_allow_html=True,
)

epoch_idx = st.slider(
    "Epoch",
    min_value=1,
    max_value=n_epochs,
    value=1,
    step=1,
    format="Epoch %d",
    key="epoch_slider",
)

fig_epoch = plot_perceptron_epoch(
    data["X_all"], data["y_all"],
    perceptron,
    epoch=epoch_idx - 1,          # 0-indexed internally
    feature_range=data["feature_range"],
)
st.pyplot(fig_epoch, use_container_width=True)

# ──────────────────────────────────────────────────────────
# Section 4 – Expandable info cards
# ──────────────────────────────────────────────────────────

st.markdown('<div class="section-heading">How It Works</div>', unsafe_allow_html=True)

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    with st.expander("🟡  Perceptron Algorithm"):
        st.markdown(
            """
**Origin:** Frank Rosenblatt, 1958

**Core idea:** Iteratively adjust weights whenever a sample is misclassified.

**Update rule:**
```
if ŷ ≠ y:
    w ← w + η · y · x
    b ← b + η · y
```

**Activation:** Step function — output is either **+1** or **−1**.

**Guarantee:** Converges in finite steps *if* the data is linearly separable.

**Limitation:** No convergence guarantee on non-separable data; boundary may oscillate.
            """
        )

with col_exp2:
    with st.expander("🟢  Logistic Regression"):
        st.markdown(
            """
**Core idea:** Model P(y=1|x) via the sigmoid function; optimise log-likelihood.

**Sigmoid:**
```
σ(z) = 1 / (1 + exp(−z))
```

**Loss (binary cross-entropy):**
```
L = −[y log(σ) + (1−y) log(1−σ)]
```

**Optimisation:** Gradient descent (sklearn uses L-BFGS by default).

**Advantage:** Probabilistic output; handles non-separable data gracefully; numerically stable.
            """
        )

# ──────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────

st.markdown(
    """
    <hr style='border:none;border-top:1px solid #2A2D3A;margin:2.5rem 0 1rem'>
    <p style='text-align:center;color:#3A3D55;font-size:0.75rem'>
        Built with Streamlit · Custom Perceptron from scratch · sklearn Logistic Regression
    </p>
    """,
    unsafe_allow_html=True,
)