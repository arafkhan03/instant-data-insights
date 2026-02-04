import streamlit as st
import pandas as pd
import plotly.express as px
from huggingface_hub import InferenceClient
import json
import io

# ===== Hugging Face client =====
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
client = InferenceClient(HF_TOKEN)

st.set_page_config(page_title="Instant Data Insights", layout="wide")
st.title("Instant Data Insights (AI + Charts)")

# ===== File Upload =====
uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df_first50 = df.head(50)
    st.write("Preview first 50 rows")
    st.dataframe(df_first50)

    # ===== Column AI =====
    st.subheader("Column Analysis")
    col_prompt = f"""
    Dataset columns: {list(df_first50.columns)}
    Show fixed/clean column names, column type (numeric, categorical, date), 
    and basic stats (mean, min, max, unique, missing values). 
    Respond in JSON with keys: column_name, fixed_name, type, stats.
    Use first 50 rows sample: {df_first50.to_dict(orient='records')}
    """
    col_response = client.text_generation(model="google/flan-t5-small", inputs=col_prompt, parameters={"max_new_tokens":500})
    try:
        col_json = json.loads(col_response.generated_text)
        st.json(col_json)
    except:
        st.warning("Column AI response could not be parsed, showing raw text.")
        st.text(col_response.generated_text)
        col_json = None

    # ===== Analyst AI =====
    st.subheader("Dataset Summary & Chart Suggestions")
    if col_json:
        analyst_prompt = f"""
        Columns info: {col_json}
        Based on the first 50 rows, generate:
        1. A quick textual summary of dataset insights
        2. Suggested charts: type, x, y, color, title
        Respond in JSON: {{'summary': str, 'charts': [{{'chart_type','x','y','color','title'}}]}}
        """
        analyst_response = client.text_generation(model="google/flan-t5-small", inputs=analyst_prompt, parameters={"max_new_tokens":500})
        try:
            analyst_json = json.loads(analyst_response.generated_text)
            st.json(analyst_json)
        except:
            st.warning("Analyst AI response could not be parsed, showing raw text.")
            st.text(analyst_response.generated_text)
            analyst_json = None

        # ===== Chart Generation =====
        if analyst_json and "charts" in analyst_json:
            st.subheader("Generated Charts")
            for c in analyst_json["charts"]:
                chart_type = c.get("chart_type")
                x = c.get("x")
                y = c.get("y")
                color = c.get("color")
                title = c.get("title", f"{chart_type} of {y} vs {x}")

                if chart_type.lower() == "bar":
                    fig = px.bar(df_first50, x=x, y=y, color=color, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type.lower() == "line":
                    fig = px.line(df_first50, x=x, y=y, color=color, title=title)
                    st.plotly_chart(fig, use_container_width=True)
                elif chart_type.lower() == "histogram":
                    fig = px.histogram(df_first50, x=x, color=color, title=title)
                    st.plotly_chart(fig, use_container_width=True)

    # ===== Feedback =====
    st.subheader("Feedback")
    feedback_col1, feedback_col2 = st.columns([1,3])
    with feedback_col1:
        thumbs = st.radio("Was this useful?", ["👍 Yes", "👎 No"])
    with feedback_col2:
        email = st.text_input("Optional email for follow-up")

    if st.button("Submit Feedback"):
        feedback_record = pd.DataFrame([{
            "feedback": thumbs,
            "email": email
        }])
        feedback_record.to_csv("feedback_log.csv", mode="a", index=False, header=st.session_state.get("feedback_file_created", True))
        st.session_state["feedback_file_created"] = False
        st.success("Thanks for your feedback!")

    # ===== Download cleaned file =====
    st.subheader("Download Cleaned Data")
    buf = io.BytesIO()
    df_first50.to_csv(buf, index=False)
    st.download_button("Download CSV", data=buf.getvalue(), file_name="cleaned_data.csv", mime="text/csv")
