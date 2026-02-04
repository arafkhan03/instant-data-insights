import io
import json
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from openai import OpenAI
import os
import traceback

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Instant Data Insights", layout="wide")

HF_TOKEN = st.secrets.get("HF_TOKEN")  # Add your Hugging Face token in Streamlit Secrets
os.environ["HF_TOKEN"] = HF_TOKEN

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

MAX_ROWS = 50  # Only analyze first 50 rows

# -------------------------------
# Upload
# -------------------------------
st.title("📊 Instant Data Insights")
uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file).head(MAX_ROWS)
        else:
            df = pd.read_excel(uploaded_file).head(MAX_ROWS)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"File loaded! Shape: {df.shape}")
    st.dataframe(df)

    # -------------------------------
    # Column Analysis (summary only)
    # -------------------------------
    st.subheader("🔧 Column Analysis")
    col_types = {"numeric": [], "categorical": []}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types["numeric"].append(col)
        else:
            col_types["categorical"].append(col)
    st.markdown(f"**Numeric columns:** {', '.join(col_types['numeric']) or 'None'}")
    st.markdown(f"**Categorical columns:** {', '.join(col_types['categorical']) or 'None'}")

    # -------------------------------
    # Prepare JSON for AI
    # -------------------------------
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Convert rows to JSON-safe types
    rows = []
    for _, row in df.iterrows():
        py_row = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (np.integer, np.int64, np.int32)):
                py_row[col] = int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                py_row[col] = float(val)
            elif isinstance(val, (np.bool_)):
                py_row[col] = bool(val)
            elif pd.isna(val):
                py_row[col] = None
            else:
                py_row[col] = str(val)
        rows.append(py_row)

    # Column statistics / summary
    column_summary = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_summary[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std())
            }
        else:
            column_summary[col] = {
                "unique_values": df[col].dropna().unique().tolist(),
                "num_unique": int(df[col].nunique())
            }

    # Metadata
    metadata = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "source_file": uploaded_file.name,
        "column_summary": column_summary
    }

    metadata_safe = json.loads(json.dumps(metadata, default=str))
    ai_json_obj = {
        "columns": column_types,
        "rows": rows,
        "metadata": metadata_safe
    }
    ai_json_str = json.dumps(ai_json_obj, ensure_ascii=False, indent=2)

    # -------------------------------
    # Build prompt for AI
    # -------------------------------
    prompt = f"""
    You are a data analyst. Here is a dataset (first {MAX_ROWS} rows):
    
    {ai_json_str}
    
    1. Suggest a brief summary of the dataset (2-3 sentences)
    2. Suggest 2-3 meaningful charts to visualize the data
    Return a JSON with 'summary' and 'charts' fields. 
    Charts should include 'column', 'chart_type' ('bar', 'line', 'histogram', 'scatter'), 'title', and 'description'.
    """

    # -------------------------------
    # AI Analysis
    # -------------------------------
    st.subheader("🧠 AI Summary & Suggested Charts")
    ai_json = {"summary": "AI analysis not available", "charts": []}  # default
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
        )

        ai_output_text = completion.choices[0].message.content
        ai_json = json.loads(ai_output_text[ai_output_text.find("{"):])
        
        st.write("Raw AI output:", repr(ai_output_text))

        if ai_output_text and ai_output_text.strip():
            ai_json = json.loads(ai_output_text)

    except Exception:
        st.warning("AI analysis failed!")
        st.text(traceback.format_exc())

    # Show summary
    st.markdown(f"**Summary:** {ai_json.get('summary')}")

    # -------------------------------
    # Generate charts
    # -------------------------------
    charts = ai_json.get("charts", [])
    for c in charts:
        col1 = c.get("column")
        chart_type = c.get("chart_type", "bar")
        title = c.get("title", "")
        description = c.get("description", "")

        if col1 in df.columns:
            # If y is numeric, pick the first numeric column that's not col1
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != col1]
            y_col = numeric_cols[0] if numeric_cols else None

            if chart_type == "bar":
                fig = px.bar(df, x=col1, y=y_col)
            elif chart_type == "line":
                fig = px.line(df, x=col1, y=y_col)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=col1, y=y_col)
            elif chart_type == "histogram":
                fig = px.histogram(df, x=col1)
            else:
                continue

            if title:
                st.markdown(f"**{title}**")
            st.plotly_chart(fig, use_container_width=True)
            if description:
                st.markdown(f"*{description}*")

    # -------------------------------
    # Feedback
    # -------------------------------
    st.subheader("👍 Feedback")
    feedback_col1, feedback_col2 = st.columns([1, 3])
    with feedback_col1:
        feedback = st.radio("Was this analysis helpful?", ("👍 Yes", "👎 No"))
    with feedback_col2:
        email = st.text_input("Optional: your email for follow-up")

    if st.button("Submit Feedback"):
        feedback_data = {
            "feedback": feedback,
            "email": email,
            "file": uploaded_file.name
        }
        with open("feedback_log.jsonl", "a") as f:
            f.write(json.dumps(feedback_data) + "\n")
        st.success("Thanks for your feedback!")
