import streamlit as st
import pandas as pd
from src.rules import analyze_text

st.set_page_config(
    page_title="Cognitive Bias Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Cognitive Bias Analyzer")
st.write("Analyze text in real time to detect System 1 heuristics, overgeneralizations, and cognitive biases.")

default_sample = "I recently saw a project fail online, so changing our plan now will obviously lead to a complete disaster!"
input_text = st.text_area("Enter input text below:", value=default_sample, height=120)

if st.button("Run Diagnostic Report", type="primary"):
    if input_text.strip():
        report = analyze_text(input_text)

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Total Characters", len(input_text))
        col2.metric("Biases Detected", report.total_biases_found)

        if report.total_biases_found == 0:
            st.success("🎉 No explicit System 1 cognitive biases detected!")
        else:
            st.warning(f"Found {report.total_biases_found} bias trigger(s) in the text.")

            table_data = []
            for item in report.detected_biases:
                table_data.append({
                    "Bias Detected": item.bias_name,
                    "Trigger Lemma": item.trigger_lemma,
                    "Category": item.category,
                    "System 2 Reframing Prompt": item.reframe_prompt
                })

            df = pd.DataFrame(table_data)
            st.dataframe(df, width="stretch")

            with st.expander("🔍 View Raw Pydantic Model Output (JSON)"):
                st.json(report.model_dump_json())
    else:
        st.error("Please provide non-empty text to run the analysis.")
