#libraries
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io

# -------------------------
# Page config & styles
# -------------------------
st.set_page_config(page_title="Big Data Analytics Dashboard",
                   page_icon="📊",
                   layout="wide")

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

st.markdown(
    """
    <style>
    .stButton>button { background-color: #4B8BBE; color: white; }
    .big-font {font-size:20px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar 
# -------------------------
with st.sidebar:
    st.markdown("## 📂 Navigation")
    selected = option_menu(
        menu_title=None,
        options=["Data Overview", "Visualizations", "Analytics"],
        icons=["table", "bar-chart", "cpu"],
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#2296bf", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "padding": "10px 15px",
                "text-align": "left",
                "margin": "2px",
            },
            "nav-link-selected": {"background-color": "#2296bf"},
        }
    )

# -------------------------
# Helper UI: Card
# -------------------------
def card(title, value, icon):
    st.markdown(
        f"""
        <div style="
            background-color: #f7faff;
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #e6e9ef;
            display: flex;
            align-items: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            ">
            <div style="font-size: 34px; margin-right: 14px;">{icon}</div>
            <div>
                <h4 style="margin:0; padding:0;">{title}</h4>
                <p style="font-size: 18px; margin:0; font-weight:600;">{value}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Caching loaders
# -------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


# -------------------------
# Utility functions
# -------------------------
NUMERIC_EXCLUDE = {"traffic_id", "weather_id"}
CATEGORICAL_EXCLUDE = {"date_time"}
def safe_get_df():
    """Return dataframe if loaded in session_state, else None."""
    return st.session_state.get("data", None)

def display_basic_overview(df):
    st.subheader("Preview")
    with st.expander("Head / Tail / Sample", expanded=True):
        st.write("Head:")
        st.dataframe(df.head(), use_container_width=True)
        st.write("Tail:")
        st.dataframe(df.tail(), use_container_width=True)
        st.write("Random sample (5 rows):")
        st.dataframe(df.sample(5), use_container_width=True)

    st.subheader("Shape & counts")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    col1, col2 = st.columns(2)
    with col1:
        st.write("Duplicated rows:", int(df.duplicated().sum()))
    with col2:
        st.write("Missing values (total):", int(df.isna().sum().sum()))

    # Well-formatted info()
    st.subheader("Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue())


    st.subheader("Unique values per column")
    st.dataframe(df.nunique().sort_values(ascending=False).rename("unique_counts").to_frame(), use_container_width=True)

    st.subheader("Null counts by column")
    nulls = df.isnull().sum().sort_values(ascending=False)
    st.dataframe(pd.DataFrame({"null_count": nulls}), use_container_width=True)

    st.subheader("Column summary")
    column_info = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Data Type": df.dtypes.astype(str).values,
        "Unique": df.nunique().values
    })
    st.dataframe(column_info, use_container_width=True)

def categorical_multiplots(df, column):
    """
    Show multiple plots for a single categorical feature:
    - Pie (plotly)
    - Donut (plotly)
    - Bar (plotly)
    - Treemap (plotly)
    - Optional Sunburst (plotly) if secondary categorical chosen
    """
    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "count"]

    st.markdown(f"### Visualizations for **{column}**")

    # Pie
    fig_pie = px.pie(counts, names=column, values="count",
                     title=f"Pie — {column}",
                     color=column, color_discrete_sequence=px.colors.qualitative.Set3)
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Donut
    fig_donut = px.pie(counts, names=column, values="count",
                       title=f"Donut — {column}",
                       hole=0.45, color=column, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_donut.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_donut, use_container_width=True)

    # Bar
    fig_bar = px.bar(counts, x=column, y="count", title=f"Bar — {column}",
                     text="count", color=column, color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)


def display_factor_interpretation():
    st.subheader("🧠 Factor Interpretation")

    st.markdown("""
        **Factor 1: Accident Risk Factor**
        This factor is strongly dominated by accident_count, with an extremely high loading,
        indicating that it represents the overall risk of traffic accidents.
        Other weather-related variables contribute only marginally, suggesting that this factor
        captures accident occurrence as an independent latent construct.

        **Factor 2: Atmospheric Conditions Factor**
        This factor loads primarily on air_pressure_hpa, with secondary contributions from
        visibility and temperature. It reflects general atmospheric and weather conditions
        that describe the state of the environment rather than direct traffic dynamics.
                
        **Factor 3: Weather Severity Factor**
        This factor is mainly influenced by wind_speed_kmh and visibility, representing
        weather severity and variability that can affect driving conditions.
        Although its direct influence on traffic variables is limited, it captures
        environmental stress caused by adverse weather conditions.
                
        **Weather–Traffic Relationship:**
        The analysis suggests that accident risk emerges as a distinct latent factor,
        largely independent of traffic flow variables.
        Among weather variables, air_pressure_hpa shows the strongest contribution within
        the atmospheric conditions factor, indicating it as the most influential weather
        feature in the dataset. 
        """)
def compute_thresholds(df):
    return {
        "Heavy rain": df['rain_mm'].quantile(0.90),
        "Temperature low extreme": df['temperature_c'].quantile(0.10),
        "Temperature high extreme": df['temperature_c'].quantile(0.90),
        "Low visibility": df['visibility_m_weather'].quantile(0.10),
        "High humidity": df['humidity'].quantile(0.90),
        "Strong winds": df['wind_speed_kmh'].quantile(0.90),
        "Congestion": df['avg_speed_kmh'].quantile(0.10)
    }


def accident_indicator(row):
    return row['accident_count'] > 0


def congestion_indicator(row, thresholds):
    return (
        row['avg_speed_kmh'] < thresholds["Congestion"]
        or str(row.get('congestion_level', '')).strip().title() in ['High', 'Severe']
    )
def scenario_functions(th):
    return {
        "Heavy rain": lambda r: r['rain_mm'] > th["Heavy rain"],
        "Low visibility": lambda r: r['visibility_m_weather'] < th["Low visibility"],
        "High humidity": lambda r: r['humidity'] > th["High humidity"],
        "Strong winds": lambda r: r['wind_speed_kmh'] > th["Strong winds"],
        "Cold temperature": lambda r: r['temperature_c'] < th["Temperature low extreme"],
        "Hot temperature": lambda r: r['temperature_c'] > th["Temperature high extreme"],
        "Congestion (low speed)": lambda r: r['avg_speed_kmh'] < th["Congestion"],
    }
def monte_carlo(df, scenario_func, thresholds, iterations=2000, sample_frac=0.1):
    accident_probs, congestion_probs = [], []
    n = len(df)
    sample_size = max(1, int(sample_frac * n))
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        sample = df.sample(sample_size, replace=True, random_state=rng.integers(0, 1e9))
        scenario_data = sample[sample.apply(scenario_func, axis=1)]

        if scenario_data.empty:
            continue

        acc_prob = scenario_data.apply(accident_indicator, axis=1).mean()
        cong_prob = scenario_data.apply(
            lambda r: congestion_indicator(r, thresholds), axis=1
        ).mean()

        accident_probs.append(acc_prob)
        congestion_probs.append(cong_prob)

    return np.array(accident_probs), np.array(congestion_probs)


# -------------------------
# Pages
# -------------------------

# DATA OVERVIEW (contains uploader here)
if selected == "Data Overview":
    st.title("📁 Dataset Overview")

    st.markdown("Upload or replace dataset here (CSV ). This page shows dataset preview & statistics.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = load_csv(uploaded_file)
            else:
                df = load_parquet(uploaded_file)
            st.session_state["data"] = df
            st.success(f"Loaded '{uploaded_file.name}'")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    df = safe_get_df()
    if df is None:
        st.info("No dataset loaded yet. Upload above to display overview.")
        st.stop()

    # Overview content
    display_basic_overview(df)

# VISUALIZATIONS
elif selected == "Visualizations":
    st.title("📈 Visualizations Dashboard")
    df = safe_get_df()
    if df is None:
        st.warning("No dataset loaded. Please upload a dataset on the 'Data Overview' page.")
        st.stop()

    st.markdown("### 🔍 Choose visualization category")
    viz_option = st.selectbox(
        "Select",
        ["Numerical Analysis", "Categorical Analysis", "Time-Based Plots"]
    )

    # detect columns
    num_cols = [ c for c in df.select_dtypes(include=["int64", "float64"]).columns if c not in NUMERIC_EXCLUDE]
    cat_cols = [ c for c in df.select_dtypes(include=["object", "category", "bool"]).columns if c not in CATEGORICAL_EXCLUDE]

    # ----------------------------
    # Numerical Analysis (histogram + boxplot)
    # ----------------------------
    if viz_option == "Numerical Analysis":
        st.header("📊 Numerical Analysis")
        if not num_cols:
            st.info("No numerical columns in dataset.")
        else:
            st.subheader("Histogram")
            col = st.selectbox("Choose numeric column (histogram)", num_cols, index=0)
            bins = st.slider("Bins", 10, 100, 20)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True, bins=bins, ax=ax, color="royalblue")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

            st.subheader("Boxplot")
            col_box = st.selectbox("Choose numeric column (boxplot)", num_cols, index=0, key="num_box_col")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[col_box].dropna(), ax=ax2, color="#2296bf")
            ax2.set_title(f"Boxplot of {col_box}")
            st.pyplot(fig2)

    # ----------------------------
    # Categorical Analysis 
    # ----------------------------
    elif viz_option == "Categorical Analysis":
        st.header("Categorical Analysis")
        if not cat_cols:
            st.info("No categorical columns in dataset.")
        else:
            sel_cat = st.selectbox("Select categorical column", cat_cols)
            categorical_multiplots(df, sel_cat)

            # Optional sunburst: choose second categorical column
            other_cats = [c for c in cat_cols if c != sel_cat]
            if other_cats:
                st.subheader("Optional nested view")
                st.write("Select a second categorical column to nest inside the first.")
                nest_col = st.selectbox("Select nest column (or choose None)", ["None"] + other_cats)
                if nest_col and nest_col != "None":
                    # build counts for sunburst
                    sun = df.groupby([sel_cat, nest_col]).size().reset_index(name="count")
                    if not sun.empty:
                        fig_sb = px.sunburst(sun, path=[sel_cat, nest_col], values="count",
                                             title=f"{sel_cat} → {nest_col}")
                        st.plotly_chart(fig_sb, use_container_width=True)
                    else:
                        st.info("No counts available for sunburst.")
            else:
                st.info("No additional categorical columns available for nested sunburst.")


    # ----------------------------
    # Time-Based Plots
    # ----------------------------
    elif viz_option == "Time-Based Plots":
        st.header("⏱️ Time-Based Patterns")
        time_cols = [c for c in ["hour", "day", "month", "year"] if c in df.columns]
        if not time_cols:
            st.info("No time-like columns found (hour/day/month/year).")
        else:
            tcol = st.selectbox("Choose time column", time_cols)
            target = st.selectbox("Numeric target to aggregate", num_cols)
            agg = st.selectbox("Aggregation", ["mean", "median", "sum", "count"])
            grouped = df.groupby(tcol)[target].agg(agg).reset_index()
            fig = px.line(grouped, x=tcol, y=target, markers=True, title=f"{agg.title()} {target} by {tcol}")
            st.plotly_chart(fig, use_container_width=True)

            # day_of_week x hour heatmap (plotly) if available
            if "day_of_week" in df.columns and "hour" in df.columns and "vehicle_count" in df.columns:
                st.markdown("**Heatmap: Day of Week × Hour (avg vehicle_count) — Plotly**")
                pivot = df.pivot_table(values="vehicle_count", index="day_of_week", columns="hour", aggfunc="mean")
                fig_h = px.imshow(pivot, title="Traffic Heatmap: Day of Week × Hour (Plotly)")
                st.plotly_chart(fig_h, use_container_width=True)


# ANALYTICS
elif selected == "Analytics":
    st.title("Analytics Page")
    df = safe_get_df()
    if df is None:
        st.warning("No dataset loaded. Please upload a dataset on the 'Data Overview' page.")
        st.stop()

    st.markdown("Choose an analysis module.")
    analysis_type = st.radio("Analysis Options", ["Monte Carlo", "Factor Analysis"])
    if analysis_type == "Factor Analysis":
        st.subheader("Factor Analysis")
        st.markdown("Upload Factor Loadings file (CSV).")
        loadings_file = st.file_uploader("Factor Loadings CSV",type=["csv"],key="fa_loadings")
        if loadings_file is not None:
                loadings_df = pd.read_csv(loadings_file)
                # Display loadings table
                st.subheader("📌 Factor Loadings")
                st.dataframe(loadings_df, use_container_width=True)
                # Heatmap
                st.markdown("Factor Loadings Heatmap")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(loadings_df.set_index(loadings_df.columns[0]),annot=True,cmap="coolwarm",ax=ax )
                st.pyplot(fig)
                # Interpretation text
                display_factor_interpretation()
        else:
                st.info("⬆️ Please upload a Factor Loadings CSV file to display results.")

    elif analysis_type == "Monte Carlo":
        st.subheader("🎲 Monte Carlo Traffic Risk Simulation")

        thresholds = compute_thresholds(df)
        scenarios = scenario_functions(thresholds)

        scenario_name = st.selectbox("Select scenario", list(scenarios.keys()))
        iterations = st.slider("Iterations", 500, 5000, 2000, step=500)
        sample_frac = st.slider("Sample fraction", 0.05, 0.5, 0.1)

        if st.button("Run Monte Carlo Simulation"):
            with st.spinner("Running simulation..."):
                acc_probs, cong_probs = monte_carlo(
                    df,
                    scenarios[scenario_name],
                    thresholds,
                    iterations=iterations,
                    sample_frac=sample_frac
                )

            if cong_probs.size == 0:
                st.warning("No matching data for this scenario.")
            else:
                st.success("Simulation completed")

                # Stats cards
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Congestion Prob", f"{cong_probs.mean():.3f}")
                col2.metric("Max Congestion Prob", f"{cong_probs.max():.3f}")
                col3.metric("Std Dev", f"{cong_probs.std():.3f}")

                # Histogram (same style as notebook)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(
                    cong_probs,
                    bins=30,
                    color="#2296bf",
                    edgecolor="black",
                    alpha=0.8
                )
                ax.set_title(f"Congestion Probability — {scenario_name}")
                ax.set_xlabel("Congestion Probability")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
    

# End of app
