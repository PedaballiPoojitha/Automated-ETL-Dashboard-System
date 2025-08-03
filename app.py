import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# PAGE SETTINGS
st.set_page_config(page_title="ETL Dashboard", layout="wide")

# PAGE HEADER
st.markdown('<h1 style="color:#00acee;font-size:40px;">⚙️ Automated ETL + Dashboard System</h1>', unsafe_allow_html=True)
st.write("Upload, clean, analyze and visualize your datasets—all in one place.")

# 1. UPLOAD
st.markdown('---')
st.markdown('### 📂 1. Upload CSV/Excel File')
uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # 2. MISSING VALUE HANDLING
    st.markdown('---')
    st.markdown('### 🧹 2. Handle Missing Values')
    strategy = st.selectbox("Select a strategy for handling missing values:", ["None", "Drop", "Fill Mean", "Fill Median", "Fill Mode"])
    df_clean = df.copy()
    if strategy == "Drop":
        df_clean = df_clean.dropna()
    elif strategy == "Fill Mean":
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
    elif strategy == "Fill Median":
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    elif strategy == "Fill Mode":
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])
    st.info(f"🔧 Missing values handled using: **{strategy}**")
    st.dataframe(df_clean.head())

    # 3. OUTLIER DETECTION
    st.markdown('---')
    st.markdown('### 🚨 3. Outlier Detection & Removal')
    method = st.radio("Choose a method to detect/remove outliers:", ["None", "Z-Score", "IQR"])
    if method == "Z-Score":
        z_scores = np.abs(stats.zscore(df_clean.select_dtypes(include=np.number)))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]
        st.success("✅ Outliers removed using Z-Score")
    elif method == "IQR":
        Q1 = df_clean.quantile(0.25)
        Q3 = df_clean.quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[~((df_clean < (Q1 - 1.5 * IQR)) | (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.success("✅ Outliers removed using IQR")
    st.dataframe(df_clean.head())

    # 4. SUMMARY STATS
    st.markdown('---')
    st.markdown('### 📊 4. Summary Statistics')
    with st.expander("📈 Numerical Summary"):
        st.dataframe(df_clean.describe())
    with st.expander("📋 Categorical Value Counts"):
        for col in df_clean.select_dtypes(include='object'):
            st.write(f"**{col}**")
            st.write(df_clean[col].value_counts())

    # 5. VISUALIZATION
    st.markdown('---')
    st.markdown('### 📉 5. Visualization Dashboard')
    chart_type = st.selectbox("Choose a chart type:", ["Box", "Histogram", "Bar", "Line", "Scatter", "Pie"])
    cols = df_clean.columns.tolist()

    try:
        if chart_type in ["Box", "Histogram"]:
            col = st.selectbox("Select a column:", cols)
            fig = px.box(df_clean, y=col, template="plotly_dark") if chart_type == "Box" else px.histogram(df_clean, x=col, template="plotly_dark")
        elif chart_type in ["Bar", "Line", "Scatter"]:
            col1 = st.selectbox("X-axis:", cols)
            col2 = st.selectbox("Y-axis:", cols)
            if chart_type == "Bar":
                fig = px.bar(df_clean, x=col1, y=col2, template="plotly_dark")
            elif chart_type == "Line":
                fig = px.line(df_clean, x=col1, y=col2, template="plotly_dark")
            else:
                fig = px.scatter(df_clean, x=col1, y=col2, template="plotly_dark")
        elif chart_type == "Pie":
            cat_col = st.selectbox("Category column:", df_clean.select_dtypes(include='object').columns)
            num_col = st.selectbox("Numeric column:", df_clean.select_dtypes(include=np.number).columns)
            fig = px.pie(df_clean, names=cat_col, values=num_col, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Error rendering chart: {e}")

    # 6. EXPORT
    st.markdown('---')
    st.markdown('### 📁 6. Export Cleaned Dataset')
    st.download_button("📥 Download Cleaned CSV", df_clean.to_csv(index=False), file_name="cleaned_data.csv")