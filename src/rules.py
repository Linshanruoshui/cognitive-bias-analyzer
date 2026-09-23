import os
import spacy
import time
import streamlit as st
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

nlp = spacy.load("en_core_web_sm")


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
    """Retrieve API key from Streamlit Cloud Secrets or local Environment Variables."""
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY", "")


def _analyze_with_llm(text: str) -> List[BiasDetection]:
    """Fallback LLM analysis using explicit model path formatting."""
    api_key = _get_api_key()
    if not api_key:
        st.warning("⚠️ GEMINI_API_KEY not configured in Secrets or environment.")
        return []

    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are an expert cognitive psychology system analyzing text for System 1 cognitive biases.
    Analyze the following text and identify implicit cognitive biases (e.g., Halo Effect, Appeal to Authority, Affect Heuristic, Confirmation Bias).
    Return a structured list of detected biases.

    Text to analyze: "{text}"
    """

    # Fully qualified model identifiers required by google-genai SDK
    models_to_try = [
        "models/gemini-2.5-flash",
        "models/gemini-1.5-flash",
    ]

    last_error = ""
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[BiasDetection],
                    temperature=0.1,
                ),
            )

            if response.parsed:
                return response.parsed
            elif response.text:
                import json
                data = json.loads(response.text)
                return [BiasDetection(**item) for item in data]
            return []
        except Exception as e:
            last_error = f"{model_name}: {e}"
            continue

    st.error(f"❌ Could not reach Gemini API. Last error: {last_error}")
    return []

def analyze_text(text: str) -> DiagnosticReport:
    doc = nlp(text)
    found_biases = []
    lemmas_in_text = [token.lemma_.lower() for token in doc]

    # 1. Fast local spaCy rule matching
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

    # 2. LLM Fallback if no explicit local rules triggered
    if not found_biases:
        found_biases = _analyze_with_llm(text)

    return DiagnosticReport(
        original_text=text,
        total_biases_found=len(found_biases),
        detected_biases=found_biases
    )