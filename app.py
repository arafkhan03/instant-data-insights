import streamlit as st
import pandas as pd
import plotly.express as px
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Instant Data Insights", layout="wide")

st.title("Instant Data Insights")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read first 50 rows
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file).head(50)
    else:
        df = pd.read_excel(uploaded_file).head(50)

    st.subheader("Preview of your data")
    st.dataframe(df)

    # -------------------------------
    # Column Fixing + Stats (AI prep)
    # -------------------------------
    def get_column_overview(df):
        overview = []
        for col in df.columns:
            col_data = df[col].dropna()
            dtype = str(df[col].dtype)
            unique_vals = col_data.nunique()
            sample_vals = col_data.sample(min(len(col_data), 3)).tolist() if not col_data.empty else []
            overview.append({
                "name": col,
                "dtype": dtype,
                "unique_vals": unique_vals,
                "sample_vals": sample_vals
            })
        return overview

    column_overview = get_column_overview(df)
    st.subheader("Column Overview (for AI)")
    st.write(column_overview)

    # -------------------------------
    # AI Analyst (Summary + Charts)
    # -------------------------------
    st.subheader("AI Suggested Summary & Charts")

    # Initialize Hugging Face InferenceClient (replace with your token if needed)
    client = InferenceClient()

    # Create a single string prompt for AI with sample data
    sample_data_str = df.head(5).to_csv(index=False)
    prompt = f"""
You are a data analyst AI.
Given the dataset with columns and first 5 rows:
{sample_data_str}
Suggest:
1. A brief summary highlighting key points.
2. Which charts would be useful for analysis (max 3).
Return as JSON with keys 'summary' and 'charts' where 'charts' is a list of chart instructions (type, x, y if applicable).
"""

    try:
        response = client.text_generation(model="google/flan-t5-large", inputs=prompt, max_new_tokens=300)
        ai_output = response.generated_text
        st.text_area("AI Output (raw)", ai_output, height=200)
    except Exception as e:
        st.error(f"AI inference error: {e}")
        ai_output = None

    # -------------------------------
    # Placeholder: Generate charts based on AI suggestions
    # -------------------------------
    st.subheader("Charts (Auto-generated based on AI suggestions)")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        st.plotly_chart(fig)
    else:
        st.info("No numeric columns detected for charting.")

    # -------------------------------
    # Download cleaned file
    # -------------------------------
    st.subheader("Download Processed Data")
    df.to_csv("processed_data.csv", index=False)
    st.download_button("Download CSV", "processed_data.csv")

    # -------------------------------
    # Feedback
    # -------------------------------
    st.subheader("Feedback")
    col1, col2 = st.columns(2)
    feedback = None
    with col1:
        if st.button("👍 Good"):
            feedback = "Good"
    with col2:
        if st.button("👎 Bad"):
            feedback = "Bad"

    email = st.text_input("Optional: leave your email for follow-up")

    if feedback:
        st.success(f"Thanks for your feedback: {feedback}")
        with open("feedback_log.csv", "a") as f:
            f.write(f"{feedback},{email}\n")
