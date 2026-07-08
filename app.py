import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Global Market AI Investment Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #111827 45%, #0f172a 100%);
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

.hero-box {
    background: linear-gradient(135deg, rgba(59,130,246,0.25), rgba(147,51,234,0.22));
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 30px;
    padding: 35px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 10px;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
    max-width: 950px;
}

.glass-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 16px 45px rgba(0,0,0,0.28);
    margin-bottom: 18px;
}

.metric-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.85));
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 14px 35px rgba(0,0,0,0.28);
    text-align: center;
}

.metric-value {
    font-size: 33px;
    font-weight: 800;
    color: #38bdf8;
}

.metric-label {
    font-size: 14px;
    color: #cbd5e1;
    margin-top: 5px;
}

.prediction-invest {
    background: linear-gradient(135deg, #064e3b, #059669);
    border-radius: 25px;
    padding: 30px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 50px rgba(5,150,105,0.3);
}

.prediction-avoid {
    background: linear-gradient(135deg, #7f1d1d, #dc2626);
    border-radius: 25px;
    padding: 30px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 50px rgba(220,38,38,0.3);
}

.prediction-monitor {
    background: linear-gradient(135deg, #78350f, #f59e0b);
    border-radius: 25px;
    padding: 30px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 50px rgba(245,158,11,0.3);
}

.prediction-title {
    font-size: 18px;
    font-weight: 600;
    opacity: 0.9;
}

.prediction-result {
    font-size: 48px;
    font-weight: 800;
    margin-top: 5px;
}

.prediction-confidence {
    font-size: 17px;
    margin-top: 8px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.12);
}

div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1.2rem;
    font-weight: 700;
    box-shadow: 0 12px 25px rgba(37,99,235,0.25);
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #6d28d9);
    color: white;
    border: none;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 14px 22px;
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.10);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
}

h1, h2, h3 {
    color: #ffffff;
}

p, label, span {
    color: #e2e8f0;
}

[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADING & CACHING
# =========================================================

@st.cache_data
def load_default_csv():
    """Checks repository fallback paths, favoring the automatic compressed file."""
    possible_files = [
        "compressed_data.csv.gz",
        "clean_global_market_ai_dataset_for_ml.csv.gz",
        "global_market_ai_dataset.csv",
        "global_market_ai_dataset(3).csv",
        "/content/clean_global_market_ai_dataset_for_ml.csv",
        "/content/global_market_ai_dataset(3).csv"
    ]

    for file in possible_files:
        if os.path.exists(file):
            return pd.read_csv(file), file

    return None, None


def create_ai_target(data):
    df = data.copy()

    required_cols = [
        "Expected_Return",
        "Risk_Level",
        "Technical_Signal",
        "News_Sentiment"
    ]

    if "AI_Investment_Decision" not in df.columns:
        if all(col in df.columns for col in required_cols):

            risk = df["Risk_Level"].astype(str).str.lower()
            signal = df["Technical_Signal"].astype(str).str.lower()

            invest_condition = (
                (df["Expected_Return"] >= 10) &
                (risk == "low") &
                (signal == "bullish") &
                (df["News_Sentiment"] > 0)
            )

            avoid_condition = (
                (df["Expected_Return"] < 3) |
                (risk == "high") |
                (signal == "bearish") |
                (df["News_Sentiment"] < -0.3)
            )

            df["AI_Investment_Decision"] = np.select(
                [invest_condition, avoid_condition],
                ["Invest", "Avoid"],
                default="Monitor"
            )
        else:
            st.error("Target column is missing and required heuristic columns are not available.")
            st.stop()

    return df


def prepare_dataset(data):
    df = data.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day

    df = create_ai_target(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    target = "AI_Investment_Decision"
    df = df.dropna(subset=[target])

    return df


# =========================================================
# EXPERT MACHINE LEARNING MODEL TRAINING
# =========================================================

@st.cache_resource(show_spinner=False)
def train_ml_model(data):
    df = data.copy()
    target = "AI_Investment_Decision"

    # 1. Separate the classes to handle the 81.5% imbalance
    df_avoid = df[df[target] == "Avoid"]
    df_monitor = df[df[target] == "Monitor"]
    df_invest = df[df[target] == "Invest"]
    
    # Downsample 'Avoid' and 'Monitor' to match the size of 'Invest' (approx 2,800 rows each)
    sample_size = len(df_invest)
    
    df_avoid_downsampled = df_avoid.sample(n=sample_size, random_state=42)
    df_monitor_downsampled = df_monitor.sample(n=sample_size, random_state=42)
    
    # Recombine into a perfectly balanced training dataset
    df_balanced = pd.concat([df_invest, df_monitor_downsampled, df_avoid_downsampled])
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. DROP LEAKAGE COLUMNS (Crucial for fixing fake 100% accuracy)
    leakage_columns = ["Expected_Return", "Risk_Level", "Technical_Signal", "News_Sentiment"]
    metadata_columns = [target, "Investment_Recommendation", "Record_ID", "Date", "AI_Model_Version"]
    drop_columns = list(set(leakage_columns + metadata_columns))
    drop_columns = [col for col in drop_columns if col in df_balanced.columns]

    X = df_balanced.drop(columns=drop_columns)
    y = df_balanced[target]

    # 3. Setup standard preprocessors
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns)
        ]
    )

    # 4. Train on the perfectly balanced subset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    return {
        "model": model,
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "classes": model.classes_,
        "feature_columns": X.columns.tolist(),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }


def get_feature_importance(model_info):
    model = model_info["model"]
    numeric_columns = model_info["numeric_columns"]
    categorical_columns = model_info["categorical_columns"]

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        feature_names = []
        feature_names.extend(numeric_columns)

        if len(categorical_columns) > 0:
            encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_features = encoder.get_feature_names_out(categorical_columns)
            feature_names.extend(cat_features)

        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)

        return importance_df

    except Exception:
        return pd.DataFrame()


# =========================================================
# DATA AUTOMATION CONTROLS (SIDEBAR)
# =========================================================

st.sidebar.markdown("## 📂 Dataset Control")

# Automatically searches and picks up files relative to your repo root path
raw_df, file_source = load_default_csv()

if raw_df is None:
    st.sidebar.error("Error: 'compressed_data.csv.gz' not found in repo directory.")
    st.stop()

df = prepare_dataset(raw_df)
model_info = train_ml_model(df)

st.sidebar.success("Dataset connected automatically!")
st.sidebar.write("**Source Local Path:**", file_source)
st.sidebar.write("**Observations:**", df.shape[0])
st.sidebar.write("**Evaluated Dimensions:**", df.shape[1])
st.sidebar.write("**Honest Model Accuracy:**", f"{model_info['accuracy'] * 100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Evaluation Vector Target")
st.sidebar.info("AI_Investment_Decision")


# =========================================================
# HERO LAYOUT HEADER
# =========================================================

st.markdown("""
<div class="hero-box">
    <div class="hero-title">Global Market AI Investment Predictor</div>
    <div class="hero-subtitle">
        An advanced Machine Learning production matrix running non-biased feature analysis over structured international market indices, 
        liquidity ratios, and geopolitical impact metrics to yield stable alpha decision strategies.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD CARD AGGREGATIONS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[0]:,}</div>
        <div class="metric-label">Total Valid Row-Entries</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[1]}</div>
        <div class="metric-label">Operational Feature Nodes</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{model_info['accuracy'] * 100:.2f}%</div>
        <div class="metric-label">Generalization Accuracy Score</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['AI_Investment_Decision'].nunique()}</div>
        <div class="metric-label">Active Prediction Target Branches</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# TABBED CONTAINER CONFIGURATION
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📊 Complete Dataset",
    "📈 Interactive Analytics",
    "🤖 Live Predictor Form",
    "🧠 Core Model Parameters"
])


# =========================================================
# TAB 1: OVERVIEW
# =========================================================

with tab1:
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("""
        <div class="glass-card">
            <h2>Internship Project Executive Summary</h2>
            <p>
            This deployment engine implements a clean Random Forest Pipeline over the updated global indicators framework. 
            By isolating artificial data indicators from the feature spaces, the predictive engine uncovers real market structural weight distributions.
            </p>
            <p>
            Use the corresponding application tabs to explore underlying target balances, check matrix data types, evaluate classification reports, or execute simulated inputs.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <h3>Technical Capabilities Checklist</h3>
            <p>✅ Automated background repository extraction via Gzip stream</p>
            <p>✅ Hardened leakage handling pipeline blocks</p>
            <p>✅ Dynamic multi-type imputer and standardizer transformers</p>
            <p>✅ Real-time interactive model inference testing framework</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        target_counts = df["AI_Investment_Decision"].value_counts().reset_index()
        target_counts.columns = ["Decision", "Count"]

        fig = px.pie(
            target_counts,
            names="Decision",
            values="Count",
            hole=0.55,
            title="Class Target Label Balancing Split"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 2: DATASET EXPLORER
# =========================================================

with tab2:
    st.markdown("## 📊 Active Dataset Explorer Matrix")
    st.dataframe(df.head(100), use_container_width=True, height=420)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows Processed", f"{df.shape[0]:,}")
    with c2:
        st.metric("Total Schema Headers", df.shape[1])
    with c3:
        st.metric("Detected Null Matrix Spaces", int(df.isnull().sum().sum()))

    st.markdown("### Structural Column Configurations")
    info_df = pd.DataFrame({
        "Column Matrix Tag": df.columns,
        "Primitive Class Type": df.dtypes.astype(str),
        "Null Entries": df.isnull().sum().values,
        "Unique Entity Count": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True, height=350)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Active Memory Dataset Preview (.CSV)",
        data=csv_data,
        file_name="clean_global_market_ai_dataset_for_ml.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 3: VISUAL INTERACTIVE ANALYTICS
# =========================================================

with tab3:
    st.markdown("## 📈 Deep Diagnostic Visualization Analytics")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        decision_counts = df["AI_Investment_Decision"].value_counts().reset_index()
        decision_counts.columns = ["Decision", "Count"]

        fig1 = px.bar(decision_counts, x="Decision", y="Count", title="Raw Category Class Tallies", text="Count")
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        if "Market_Trend" in df.columns:
            trend_counts = df.groupby(["Market_Trend", "AI_Investment_Decision"]).size().reset_index(name="Count")
            fig2 = px.bar(trend_counts, x="Market_Trend", y="Count", color="AI_Investment_Decision", title="Macro Trend Vector vs Generated Strategy Target", barmode="group")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
            st.plotly_chart(fig2, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        if "Volatility_Index" in df.columns:
            fig3 = px.box(df, x="AI_Investment_Decision", y="Volatility_Index", title="Volatility Spread Metrics Across Decisions")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
            st.plotly_chart(fig3, use_container_width=True)

    with chart_col4:
        if "Liquidity_Score" in df.columns:
            fig4 = px.histogram(df, x="Liquidity_Score", color="AI_Investment_Decision", title="Asset Liquidity Volume Density Curve", nbins=40)
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=420)
            st.plotly_chart(fig4, use_container_width=True)


# =========================================================
# TAB 4: SIMULATED INFERENCE PREDICTOR
# =========================================================

with tab4:
    st.markdown("## 🤖 Real-Time Production Simulation Prediction")
    left, right = st.columns([1.2, 1])

    feature_columns = model_info["feature_columns"]
    categorical_columns = model_info["categorical_columns"]
    numeric_columns = model_info["numeric_columns"]

    with left:
        st.markdown("""
        <div class="glass-card">
            <h3>Simulated Ingestion Parameters</h3>
            <p>Input new production-level values below to execute an evaluation call to the Random Forest core structure.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("production_prediction_form"):
            input_data = {}
            input_cols = st.columns(2)

            for index, column in enumerate(feature_columns):
                current_col = input_cols[index % 2]

                with current_col:
                    if column in categorical_columns:
                        options = sorted(df[column].dropna().astype(str).unique().tolist())
                        if len(options) == 0:
                            options = ["Unknown"]

                        input_data[column] = st.selectbox(
                            label=column.replace("_", " "),
                            options=options,
                            key=f"prod_cat_{column}"
                        )
                    else:
                        col_data = pd.to_numeric(df[column], errors="coerce")
                        min_value = float(col_data.min()) if pd.notnull(col_data.min()) else 0.0
                        max_value = float(col_data.max()) if pd.notnull(col_data.max()) else 100.0
                        mean_value = float(col_data.mean()) if pd.notnull(col_data.mean()) else 0.0

                        input_data[column] = st.number_input(
                            label=column.replace("_", " "),
                            min_value=min_value,
                            max_value=max_value,
                            value=mean_value,
                            key=f"prod_num_{column}"
                        )

            submitted = st.form_submit_button("🚀 Execute Live Evaluation Inference Pipeline")

    with right:
        st.markdown("""
        <div class="glass-card">
            <h3>Calculated Structural Strategy Target Output</h3>
            <p>The matrix reads remaining structural variables to classify allocation risk weighting choices.</p>
        </div>
        """, unsafe_allow_html=True)

        if submitted:
            input_df = pd.DataFrame([input_data])
            prediction = model_info["model"].predict(input_df)[0]

            try:
                prediction_proba = model_info["model"].predict_proba(input_df)[0]
                classes = model_info["model"].classes_
                probability_df = pd.DataFrame({
                    "Decision Structure": classes,
                    "Probability Ratio": prediction_proba
                }).sort_values(by="Probability Ratio", ascending=False)

                confidence = probability_df.iloc[0]["Probability Ratio"] * 100
            except Exception:
                probability_df = pd.DataFrame()
                confidence = 0

            prediction_lower = str(prediction).lower()

            if prediction_lower == "invest":
                css_class = "prediction-invest"
                message = "The localized underlying market features project high structural strength configurations."
            elif prediction_lower == "avoid":
                css_class = "prediction-avoid"
                message = "System parameters project heightened exposure limits. Mitigation routes requested."
            else:
                css_class = "prediction-monitor"
                message = "Neutral metric state. Maintain existing liquidity constraints."

            st.markdown(f"""
            <div class="{css_class}">
                <div class="prediction-title">Allocated Matrix Result Strategy</div>
                <div class="prediction-result">{prediction}</div>
                <div class="prediction-confidence">Statistical Evaluation Confidence: {confidence:.2f}%</div>
                <p>{message}</p>
            </div>
            """, unsafe_allow_html=True)

            if not probability_df.empty:
                fig_prob = px.bar(probability_df, x="Decision Structure", y="Probability Ratio", title="Probabilistic Class Distribution Ratios")
                fig_prob.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=320)
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("Awaiting evaluation configurations from the parameters panel.")


# =========================================================
# TAB 5: UNDERLYING PIPELINE DETAILS
# =========================================================

with tab5:
    st.markdown("## 🧠 Production Pipeline Metadata Architecture")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Active Estimator Struct", "Random Forest Ensemble")
    with m2:
        st.metric("Leakage-Protected Accuracy", f"{model_info['accuracy'] * 100:.2f}%")
    with m3:
        st.metric("Isolated Labels Space Type", "Multiclass Target Mapping")

    st.markdown("### Test Split Classification Report Summary")
    report_df = pd.DataFrame(model_info["report"]).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.markdown("### Pipeline Confusion Matrix Coordinates")
    cm = model_info["confusion_matrix"]
    classes = model_info["classes"]
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    fig_cm = px.imshow(cm_df, text_auto=True, title="Model Evaluation Cross-Classification Tally", labels=dict(x="Predicted Labels", y="True Labels", color="Instances Count"))
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=450)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### Generalized Feature Importance Weight Weights")
    importance_df = get_feature_importance(model_info)

    if not importance_df.empty:
        fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="Top Core Informational Gain Contributors")
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("No dynamic numeric attributes evaluated for informational gain mapping displays.")
