import os
import json
import streamlit as st
from pydantic import BaseModel, Field
from typing import List
from google import genai

# Page Configuration
st.set_page_config(
    page_title="Cognitive Bias Analyzer",
    page_icon="🧠",
    layout="wide"
)

# Safe spaCy model loading
nlp = None
try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


class BiasDetection(BaseModel):
    bias_name: str = Field(description="Name of the cognitive bias or System 1 heuristic")
    trigger_lemma: str = Field(description="Key word, phrase, or concept triggering the bias")
    category: str = Field(description="General psychological category of the bias")
    reframe_prompt: str = Field(description="A System 2 reframing question to mitigate the bias")


class DiagnosticReport(BaseModel):
    original_text: str
    total_biases_found: int
    detected_biases: List[BiasDetection] = Field(default_factory=list)


BIAS_RULES = [
    {
        "name": "Availability Heuristic",
        "trigger_lemmas": ["recently", "saw", "heard", "remember"],
        "category": "Recall Bias",
        "reframe": "Are you relying solely on recent or memorable examples rather than statistical baseline data?"
    },
    {
        "name": "Catastrophizing",
        "trigger_lemmas": ["disaster", "ruin", "terrible", "worst"],
        "category": "Emotional Magnification",
        "reframe": "What is the actual most realistic outcome versus this worst-case scenario?"
    },
    {
        "name": "Overgeneralization",
        "trigger_lemmas": ["always", "never", "everyone", "nobody", "obviously"],
        "category": "Absolute Thinking",
        "reframe": "Are there counterexamples or exceptions that contradict this absolute statement?"
    }
]


def _get_api_key() -> str:
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY", "")


def _analyze_with_llm(text: str) -> List[BiasDetection]:
    api_key = _get_api_key()
    if not api_key:
        st.warning("⚠️ GEMINI_API_KEY not configured in Secrets or environment.")
        return []

    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert cognitive psychology system analyzing text for System 1 cognitive biases.
    Analyze the following text and identify implicit cognitive biases (e.g., Halo Effect, Appeal to Authority, Affect Heuristic, Confirmation Bias).
    Return a structured JSON list of detected biases with keys: bias_name, trigger_lemma, category, reframe_prompt.

    Text to analyze: "{text}"
    """

    # Priority order: If one model gives a 503 overload error, it immediately moves to the next
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-3.6-flash",
        "gemini-1.5-flash"
    ]

    last_error = ""
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )

            data = json.loads(response.text)
            return [BiasDetection(**item) for item in data]
        except Exception as e:
            last_error = f"{model_name}: {e}"
            continue  # Automatically failover to the next model in the list

    st.error(f"❌ Gemini API Error Details: {last_error}")
    return []


def analyze_text(text: str) -> DiagnosticReport:
    found_biases = []

    if nlp is not None:
        doc = nlp(text)
        lemmas_in_text = [token.lemma_.lower() for token in doc]
    else:
        lemmas_in_text = text.lower().split()

    for rule in BIAS_RULES:
        for trigger in rule["trigger_lemmas"]:
            if trigger in lemmas_in_text:
                found_biases.append(
                    BiasDetection(
                        bias_name=rule["name"],
                        trigger_lemma=trigger,
                        category=rule["category"],
                        reframe_prompt=rule["reframe"]
                    )
                )

    if not found_biases:
        found_biases = _analyze_with_llm(text)

    return DiagnosticReport(
        original_text=text,
        total_biases_found=len(found_biases),
        detected_biases=found_biases
    )


# --- STREAMLIT UI RENDER ---
st.title("🧠 Cognitive Bias Analyzer")
st.markdown("Analyze text for System 1 heuristics, emotional magnification, and absolute thinking.")

input_text = st.text_area(
    "Input Text to Analyze:",
    value="She is so kind and well-spoken, so there is no doubt her software architecture will be reliable.",
    height=150
)

if st.button("Run Diagnostic Report", type="primary"):
    if not input_text.strip():
        st.warning("Please enter text to analyze.")
    else:
        with st.spinner("Analyzing text for cognitive biases..."):
            report = analyze_text(input_text)

        st.subheader(f"Results: {report.total_biases_found} Bias(es) Detected")

        if report.detected_biases:
            for idx, bias in enumerate(report.detected_biases, 1):
                with st.expander(f"{idx}. {bias.bias_name} ({bias.category})", expanded=True):
                    st.write(f"**Trigger Word / Concept:** `{bias.trigger_lemma}`")
                    st.info(f"**System 2 Reframe Question:** {bias.reframe_prompt}")
        else:
            st.success("No obvious cognitive biases detected in the provided text.")