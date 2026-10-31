
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TruthLens AI | Fake News Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/fake_news_model.joblib")

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5edf8;
}
[data-testid="stHeader"] {
    background: rgba(11, 17, 32, 0.95);
}
[data-testid="stSidebar"] {
    background: #111b2e;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
h1, h2, h3, p, label, li {
    color: #e5edf8 !important;
}
.hero {
    background: linear-gradient(120deg, #172d4d, #173e5e);
    padding: 30px;
    border: 1px solid #315477;
    border-radius: 20px;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 35px;
    margin-bottom: 8px;
}
.hero p {
    color: #c4d7ee !important;
    font-size: 16px;
}
.metric-card {
    background: #141f33;
    border: 1px solid #293b56;
    border-radius: 15px;
    padding: 20px;
    min-height: 105px;
}
.result-card {
    background: #142239;
    border: 1px solid #38516f;
    border-radius: 16px;
    padding: 22px;
    margin: 15px 0;
}
div.stButton > button,
div.stDownloadButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 600;
}
.stTextArea textarea {
    background: #111c30 !important;
    color: #ffffff !important;
    border: 1px solid #425a79 !important;
    border-radius: 12px !important;
}
div[data-testid="stMetric"] {
    background: #141f33;
    border: 1px solid #293b56;
    padding: 15px;
    border-radius: 13px;
}
div[data-testid="stMetricLabel"] p {
    color: #b8c9df !important;
}
hr {
    border-color: #293b56;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🔎 TruthLens AI")
    st.caption("Intelligent News Analysis")
    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Choose a page",
        ["🏠 Dashboard", "📋 History", "ℹ️ About Project"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Technology")
    st.write("🐍 Python")
    st.write("🧠 Scikit-learn")
    st.write("📝 TF-IDF + Logistic Regression")
    st.write("🖥️ Streamlit")

    st.info(
        "Educational prototype. The model does not independently "
        "verify news against trusted sources."
    )

# ---------------- LOAD MODEL ----------------
if not MODEL_PATH.exists():
    st.error("Trained model not found.")
    st.code("python train_model.py")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🔎 TruthLens AI</h1>
        <p>Fake News Detection & Text Classification Dashboard</p>
        <p>Analyze news text with a Machine Learning model.</p>
    </div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)
    fake_count = sum(
        item["Prediction"] == "FAKE"
        for item in st.session_state.history
    )
    real_count = sum(
        item["Prediction"] == "REAL"
        for item in st.session_state.history
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("📰 Articles Analyzed", total)

    with c2:
        st.metric("⚠️ Likely Fake", fake_count)

    with c3:
        st.metric("✅ Likely Real", real_count)

    st.divider()

    left, right = st.columns([1.7, 1], gap="large")

    with left:
        st.markdown("### 📝 Analyze a News Article")
        st.write("Paste a headline or article into the box below.")

        with st.form("news_analysis_form"):
            news_text = st.text_area(
                "News content",
                value=st.session_state.news_input,
                height=230,
                placeholder=(
                    "Example: The university published its "
                    "official admission schedule..."
                ),
            )

            submitted = st.form_submit_button(
                "🔍 Analyze News",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            st.session_state.news_input = news_text

            if len(news_text.strip()) < 20:
                st.warning("Please enter at least 20 characters.")
            else:
                prediction = model.predict([news_text.strip()])[0]
                probabilities = model.predict_proba(
                    [news_text.strip()]
                )[0]

                classes = list(
                    model.named_steps["classifier"].classes_
                )
                confidence = float(
                    probabilities[classes.index(prediction)]
                )

                result = {
                    "News": news_text.strip(),
                    "Prediction": str(prediction),
                    "Model score": confidence,
                }

                st.session_state.last_result = result
                st.session_state.history.insert(0, result)
                st.session_state.history = (
                    st.session_state.history[:100]
                )

    with right:
        st.markdown("### ⚙️ How It Works")

        st.markdown("""
        <div class="result-card">
            <h3>01 · Input</h3>
            <p>Enter news text for analysis.</p>
            <hr>
            <h3>02 · NLP Processing</h3>
            <p>TF-IDF converts words into numerical features.</p>
            <hr>
            <h3>03 · Prediction</h3>
            <p>Logistic Regression predicts a news category.</p>
            <hr>
            <h3>04 · Review</h3>
            <p>Check the output and verify it independently.</p>
        </div>
        """, unsafe_allow_html=True)

    # Show latest result
    result = st.session_state.last_result

    if result:
        st.divider()
        st.markdown("## 📊 Analysis Result")

        result_left, result_right = st.columns([1.4, 1])

        with result_left:
            if result["Prediction"] == "FAKE":
                st.error("⚠️ Model Prediction: LIKELY FAKE")
            else:
                st.success("✅ Model Prediction: LIKELY REAL")

            st.markdown(
                f'<div class="result-card">'
                f'<p><b>Analyzed text</b></p>'
                f'<p>{result["News"][:500]}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with result_right:
            st.metric(
                "Model confidence score",
                f'{result["Model score"]:.1%}',
            )
            st.progress(result["Model score"])
            st.caption(
                "This is the model's estimated class probability, "
                "not the probability that the news is actually true."
            )

        st.warning(
            "Always verify important claims through reliable "
            "independent sources. This model may misclassify news."
        )

    # History chart
    if st.session_state.history:
        st.divider()
        st.markdown("### 📈 Prediction Overview")

        chart_data = pd.DataFrame(st.session_state.history)
        counts = chart_data["Prediction"].value_counts()
        st.bar_chart(counts)

        export_df = chart_data.copy()
        export_df["Model score"] = export_df["Model score"].map(
            lambda score: f"{score:.1%}"
        )

        st.download_button(
            "⬇️ Download Prediction Report",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="truthlens_prediction_report.csv",
            mime="text/csv",
        )

# ---------------- HISTORY PAGE ----------------
elif page == "📋 History":

    st.markdown("## 📋 Analysis History")
    st.write("Review predictions made during this browser session.")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history).copy()
        history_df["News"] = history_df["News"].str.slice(0, 150)
        history_df["Model score"] = history_df["Model score"].map(
            lambda score: f"{score:.1%}"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Export History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="truthlens_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
    else:
        st.info("No predictions yet. Analyze some news first.")

# ---------------- ABOUT PAGE ----------------
elif page == "ℹ️ About Project":

    st.markdown("## ℹ️ About TruthLens AI")

    st.markdown("""
    **Project objective**

    Build an educational application that classifies news text
    using Natural Language Processing and Machine Learning.

    **Key features**

    - News text input and classification
    - TF-IDF text vectorization
    - Logistic Regression classifier
    - Estimated class probability
    - Session-based prediction history
    - CSV report download
    - Interactive dashboard

    **Limitations**

    - The model depends on the quality of its training dataset.
    - The current sample dataset is very small and synthetic.
    - It does not retrieve or verify live news articles.
    - Model confidence is not proof of factual accuracy.
    - History is stored only in the current Streamlit session.
    """)

    st.info(
        "For a meaningful evaluation, train and test the model "
        "on a large, representative, correctly labelled dataset."
    )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "TruthLens AI · Python + Machine Learning · Educational Prototype"
)
