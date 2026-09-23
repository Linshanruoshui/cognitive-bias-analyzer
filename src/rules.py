import os
import spacy
import time
import streamlit as st
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai  # 旧安定版SDKインポート

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
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.environ.get("GEMINI_API_KEY", "")

def _analyze_with_llm(text: str) -> List[BiasDetection]:
    """Fallback LLM analysis using standard google-generativeai SDK."""
    api_key = _get_api_key()
    if not api_key:
        st.warning("⚠️ GEMINI_API_KEY not configured in Secrets or environment.")
        return []

    # 旧SDKの設定方法
    genai.configure(api_key=api_key)

    prompt = f"""
    You are an expert cognitive psychology system analyzing text for System 1 cognitive biases.
    Analyze the following text and identify implicit cognitive biases (e.g., Halo Effect, Appeal to Authority, Affect Heuristic, Confirmation Bias).
    Return a structured JSON list of detected biases with keys: bias_name, trigger_lemma, category, reframe_prompt.

    Text to analyze: "{text}"
    """

    try:
        # 安定版モデル呼び出し
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        import json
        data = json.loads(response.text)
        return [BiasDetection(**item) for item in data]
    except Exception as e:
        st.error(f"❌ Gemini API Error: {e}")
        return []

def _analyze_with_llm(text: str) -> List[BiasDetection]:
    """Fallback LLM analysis using standard google-generativeai SDK."""
    api_key = _get_api_key()
    if not api_key:
        st.warning("⚠️ GEMINI_API_KEY not configured in Secrets or environment.")
        return []

    genai.configure(api_key=api_key)

    prompt = f"""
    You are an expert cognitive psychology system analyzing text for System 1 cognitive biases.
    Analyze the following text and identify implicit cognitive biases (e.g., Halo Effect, Appeal to Authority, Affect Heuristic, Confirmation Bias).
    Return a structured JSON list of detected biases with keys: bias_name, trigger_lemma, category, reframe_prompt.

    Text to analyze: "{text}"
    """

    # 旧SDKでは 'models/' プレフィックスを付けずに 'gemini-1.5-flash' と直接指定します
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    last_error = ""
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            import json
            data = json.loads(response.text)
            return [BiasDetection(**item) for item in data]
        except Exception as e:
            last_error = f"{model_name}: {e}"
            continue

    st.error(f"❌ Gemini API Error: {last_error}")
    return []