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
st.caption("Upload a CSV or Excel file to instantly understand your data. You’ll get a quick summary, column insights, and automatically generated charts. No files or analyses stored.")
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
    
    1. Suggest a brief summary of the dataset (2-3 sentences).
    2. Suggest 2-3 meaningful charts to visualize the data. Each chart should include **at most 2 columns**: one for the x-axis and one optional for the y-axis. Always choose the most significant column(s) for each chart based on the data.
    
    Return a JSON with 'summary' and 'charts' fields. Each chart should include:
    - 'x_axis': name of the column for the x-axis
    - 'y_axis' (optional): name of the column for the y-axis if relevant
    - 'chart_type': one of 'bar', 'line', 'histogram', 'scatter'
    - 'title': chart title
    - 'description': short description of the chart
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
        #st.write("Raw AI output:", repr(ai_output_text))
    
        # Clean AI output and parse JSON in one line
        ai_json = json.loads(ai_output_text[ai_output_text.find("{") : ai_output_text.rfind("}") + 1])

    except Exception:
        st.warning("AI analysis failed!")
        st.text(traceback.format_exc())
        ai_json = {"summary": "AI analysis not available", "charts": []}

    # Show summary
    st.markdown(f"**Summary:** {ai_json.get('summary')}")

    # -------------------------------
    # Generate charts
    # -------------------------------
    charts = ai_json.get("charts", [])
    for c in charts:
        x_col = c.get("x_axis") or c.get("column")  # fallback to old 'column' if x_axis missing
        y_col = c.get("y_axis")  # optional
        chart_type = c.get("chart_type", "bar")
        title = c.get("title", "")
        description = c.get("description", "")
    
        if not x_col or x_col not in df.columns:
            st.text(f"Skipping chart '{title}': x-axis column '{x_col}' not found in DataFrame.")
            continue
    
        # Pick y column if needed and not provided
        if y_col and y_col not in df.columns:
            y_col = None
        if chart_type in ["bar", "line", "scatter"] and not y_col:
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != x_col]
            y_col = numeric_cols[0] if numeric_cols else None
    
        # -------------------------------
        # Data cleaning
        # -------------------------------
        df_plot = df.copy()
    
        # Convert dates if x_col looks like datetime
        if pd.api.types.is_object_dtype(df_plot[x_col]):
            try:
                df_plot[x_col] = pd.to_datetime(df_plot[x_col], errors="ignore")
            except Exception:
                pass
    
        # Sort by x_col if line chart
        if chart_type == "line" and pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
            df_plot = df_plot.sort_values(by=x_col)
    
        # Ensure y_col is numeric
        if y_col:
            df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors="coerce")
    
        # -------------------------------
        # Plot chart
        # -------------------------------
        try:
            if chart_type == "bar":
                fig = px.bar(df_plot, x=x_col, y=y_col)
            elif chart_type == "line":
                fig = px.line(df_plot, x=x_col, y=y_col)
            elif chart_type == "scatter":
                fig = px.scatter(df_plot, x=x_col, y=y_col)
            elif chart_type == "histogram":
                fig = px.histogram(df_plot, x=x_col)
            else:
                st.text(f"Skipping chart '{title}': unknown chart type '{chart_type}'.")
                continue
    
            # Display title and description
            if title:
                st.markdown(f"**{title}**")
            st.plotly_chart(fig, use_container_width=True)
            if description:
                st.markdown(f"*{description}*")
    
        except Exception:
            st.text(f"Error generating chart '{title}':")
            st.text(traceback.format_exc())

    # -------------------------------
    # Feedback (lightweight + high-signal)
    # -------------------------------
    st.markdown("---")
    
    feedback_text = st.text_area(
        "Something broke or felt off? (optional)",
        placeholder="e.g. summary was wrong, chart looked weird, upload failed, slow, confusing…",
    )
    
    if st.button("Submit feedback"):
        feedback_payload = {
            "feedback_text": feedback_text,
            "file_name": uploaded_file.name if uploaded_file else None,
            "num_rows": df.shape[0] if uploaded_file else None,
            "num_columns": df.shape[1] if uploaded_file else None,
            "columns": list(df.columns) if uploaded_file else None,
            "ai_summary_present": bool(ai_json.get("summary")),
            "num_charts_returned": len(ai_json.get("charts", [])),
        }
    
        with open("feedback_log.jsonl", "a") as f:
            f.write(json.dumps(feedback_payload) + "\n")
    
        st.success("Thanks — this helps improve the app 🙏")
    
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 0.9em; margin-top: 1rem;">
        Developed by <a href="https://www.linkedin.com/in/arafkhan03/" target="_blank">Araf</a>
    </div>
    """,
    unsafe_allow_html=True,
)

