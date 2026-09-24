import os
import re
import time
from typing import List, Optional

import streamlit as st
from pydantic import BaseModel, Field
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cognitive Bias Analyzer",
    page_icon="🧠",
    layout="centered",
)


# ============================================================
# CONFIGURATION
# ============================================================

# You can override this in Streamlit Secrets or environment variables.
#
# Recommended:
# GEMINI_MODEL = "gemini-3.8-flash"
#
# If your account has access problems with that model, you can
# temporarily try another currently supported model.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# If one model is temporarily overloaded, automatically
# try another current Flash model.
GEMINI_FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

MAX_RETRIES_PER_MODEL = 2
INITIAL_RETRY_DELAY = 1.5


# ============================================================
# DATA MODELS
# ============================================================

class BiasDetection(BaseModel):
    bias_name: str = Field(
        description="Name of the cognitive bias or System 1 heuristic."
    )
    trigger_lemma: str = Field(
        description="The exact word, phrase, or idea in the input that triggers the bias."
    )
    category: str = Field(
        description="General psychological category of the bias."
    )
    reframe_prompt: str = Field(
        description="A System 2 question that could help the person reconsider the bias."
    )


class BiasDetectionResponse(BaseModel):
    detected_biases: List[BiasDetection] = Field(
        default_factory=list,
        description="List of cognitive biases detected in the text."
    )


class DiagnosticReport(BaseModel):
    original_text: str
    total_biases_found: int
    detected_biases: List[BiasDetection] = Field(default_factory=list)
    analysis_method: str = "Unknown"
    error_message: Optional[str] = None


# ============================================================
# RULE-BASED BIAS DETECTION
# ============================================================

# These rules are deliberately conservative.
# They are intended to catch obvious linguistic signals before
# sending the text to Gemini.
BIAS_RULES = [
    {
        "name": "Availability Heuristic",
        "trigger_patterns": [
            r"\brecently\b",
            r"\bjust saw\b",
            r"\bjust heard\b",
            r"\bremember\b",
            r"\bthe first thing that comes to mind\b",
            r"\bi keep hearing\b",
        ],
        "category": "Recall Bias",
        "reframe": (
            "Am I relying on a memorable or recent example rather than "
            "considering how common this outcome actually is?"
        ),
    },
    {
        "name": "Catastrophizing",
        "trigger_patterns": [
            r"\bdisaster\b",
            r"\bruined?\b",
            r"\bruin\b",
            r"\bterrible\b",
            r"\bworst\b",
            r"\beverything is over\b",
            r"\bmy life is over\b",
            r"\bthis will destroy\b",
        ],
        "category": "Emotional Magnification",
        "reframe": (
            "What is the most realistic outcome, and how does it differ "
            "from the worst-case scenario?"
        ),
    },
    {
        "name": "Overgeneralization",
        "trigger_patterns": [
            r"\balways\b",
            r"\bnever\b",
            r"\beveryone\b",
            r"\bnobody\b",
            r"\bno one\b",
            r"\bobviously\b",
            r"\beverything\b",
            r"\bnothing\b",
        ],
        "category": "Absolute Thinking",
        "reframe": (
            "Are there counterexamples or exceptions that make this "
            "statement less absolute?"
        ),
    },
    {
        "name": "All-or-Nothing Thinking",
        "trigger_patterns": [
            r"\bcompletely useless\b",
            r"\btotal failure\b",
            r"\babsolute failure\b",
            r"\bperfect or useless\b",
            r"\beither .* or nothing\b",
            r"\bif .* then .* failure\b",
        ],
        "category": "Absolute Thinking",
        "reframe": (
            "What are the intermediate possibilities between complete "
            "success and complete failure?"
        ),
    },
    {
        "name": "Emotional Reasoning",
        "trigger_patterns": [
            r"\bi feel like .* therefore",
            r"\bi feel .* so it must be",
            r"\bbecause it feels\b",
            r"\bfeels true\b",
            r"\bfeels wrong\b",
            r"\bfeels right\b",
        ],
        "category": "Emotion-Based Reasoning",
        "reframe": (
            "What evidence supports this conclusion independently of "
            "how I currently feel?"
        ),
    },
]


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase text for rule matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_with_rules(text: str) -> List[BiasDetection]:
    """
    Run conservative local rules first.

    We intentionally don't require spaCy here. This makes the app
    easier to deploy and avoids failures caused by a missing
    en_core_web_sm model.
    """
    normalized = normalize_text(text)
    found_biases = []

    for rule in BIAS_RULES:
        matched_trigger = None

        for pattern in rule["trigger_patterns"]:
            match = re.search(pattern, normalized)

            if match:
                matched_trigger = match.group(0)
                break

        if matched_trigger:
            found_biases.append(
                BiasDetection(
                    bias_name=rule["name"],
                    trigger_lemma=matched_trigger,
                    category=rule["category"],
                    reframe_prompt=rule["reframe"],
                )
            )

    return found_biases


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

def get_secret_or_env(name: str, default: str = "") -> str:
    """
    Read a value from Streamlit Secrets first, then environment.
    """
    try:
        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value)
    except Exception:
        pass

    return os.environ.get(name, default)


def get_api_key() -> str:
    return get_secret_or_env("GEMINI_API_KEY", "")


def get_model_name() -> str:
    return get_secret_or_env(
        "GEMINI_MODEL",
        DEFAULT_GEMINI_MODEL,
    )


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def _is_retryable_error(error: Exception) -> bool:
    """
    Identify temporary errors such as 503/unavailable and
    rate-limit errors.

    We intentionally inspect the error text because the exact
    exception classes can differ between google-genai versions.
    """
    message = str(error).lower()

    retryable_terms = [
        "503",
        "unavailable",
        "service unavailable",
        "high demand",
        "temporarily unavailable",
        "429",
        "resource exhausted",
        "rate limit",
        "too many requests",
        "deadline exceeded",
        "timeout",
    ]

    return any(term in message for term in retryable_terms)


def _analyze_with_gemini(
    text: str,
) -> tuple[Optional[List[BiasDetection]], Optional[str]]:

    """
    Analyze text using Gemini.

    If the selected model is temporarily unavailable, the function
    retries it and then automatically falls back to other supported
    Flash models.

    Returns:

        (bias_list, None)
            = successful analysis

        (None, error_message)
            = analysis failed
    """

    api_key = get_api_key()

    if not api_key:
        return (
            None,
            "GEMINI_API_KEY is not configured. "
            "Add it to Streamlit Secrets or your environment variables.",
        )

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return (
            None,
            f"Could not initialize Gemini client: {e}",
        )

    prompt = f"""
You are an expert cognitive psychology system.

Analyze the following text for cognitive biases and System 1
heuristics.

Look for both explicit and implicit reasoning errors.

Possible examples include:

- Halo Effect
- Affect Heuristic
- Appeal to Authority
- Confirmation Bias
- Availability Heuristic
- Anchoring
- Overgeneralization
- Catastrophizing
- Emotional Reasoning
- Fundamental Attribution Error
- Stereotyping
- False Dichotomy
- Bandwagon Effect
- Outcome Bias
- Optimism Bias
- Negativity Bias
- Loss Aversion
- Self-Serving Bias
- Hindsight Bias
- Planning Fallacy
- Status Quo Bias
- Mere Exposure Effect

Important instructions:

1. Only identify a bias when there is meaningful evidence.
2. Do not invent a bias merely because one is theoretically possible.
3. Distinguish descriptions from actual reasoning errors.
4. Identify the specific word, phrase, or reasoning pattern
   associated with each detected bias.
5. Give a concise System 2 reframing question.
6. If there is no meaningful cognitive bias, return an empty list.
7. Return only the structured response requested by the schema.

Text to analyze:

{text}
"""

    # --------------------------------------------------------
    # Build model list.
    #
    # If GEMINI_MODEL is explicitly configured, try it first.
    # Then use the fallback models.
    # --------------------------------------------------------

    configured_model = get_model_name()

    models_to_try = [configured_model]

    for model in GEMINI_FALLBACK_MODELS:
        if model not in models_to_try:
            models_to_try.append(model)

    errors = []

    # --------------------------------------------------------
    # Try each model
    # --------------------------------------------------------

    for model_name in models_to_try:

        for attempt in range(MAX_RETRIES_PER_MODEL):

            try:

                st.caption(
                    f"Using Gemini model: `{model_name}`"
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": BiasDetectionResponse,
                    },
                )

                # ------------------------------------------------
                # Structured response
                # ------------------------------------------------

                parsed = getattr(response, "parsed", None)

                if parsed is not None:

                    if isinstance(
                        parsed,
                        BiasDetectionResponse,
                    ):
                        return (
                            parsed.detected_biases,
                            None,
                        )

                    try:

                        parsed_report = (
                            BiasDetectionResponse.model_validate(
                                parsed
                            )
                        )

                        return (
                            parsed_report.detected_biases,
                            None,
                        )

                    except Exception:
                        pass

                # ------------------------------------------------
                # Fallback to raw JSON
                # ------------------------------------------------

                response_text = getattr(
                    response,
                    "text",
                    None,
                )

                if not response_text:

                    errors.append(
                        f"{model_name}: empty response"
                    )

                    break

                try:

                    parsed_report = (
                        BiasDetectionResponse.model_validate_json(
                            response_text
                        )
                    )

                    return (
                        parsed_report.detected_biases,
                        None,
                    )

                except Exception as parse_error:

                    errors.append(
                        f"{model_name}: invalid response format: "
                        f"{parse_error}"
                    )

                    break

            except Exception as e:

                error_text = str(e)

                errors.append(
                    f"{model_name}: {error_text}"
                )

                # --------------------------------------------
                # Determine whether retry makes sense
                # --------------------------------------------

                if _is_retryable_error(e):

                    if attempt < MAX_RETRIES_PER_MODEL - 1:

                        delay = (
                            INITIAL_RETRY_DELAY
                            * (2 ** attempt)
                        )

                        time.sleep(delay)

                        continue

                    # This model appears temporarily unavailable.
                    # Move to the next model.
                    break

                else:

                    # Non-transient error.
                    # For example, malformed request or authentication.
                    #
                    # Trying another model won't normally fix that,
                    # so stop immediately.
                    return (
                        None,
                        f"Gemini API error using "
                        f"`{model_name}`: {error_text}",
                    )

    # --------------------------------------------------------
    # Every model failed
    # --------------------------------------------------------

    error_summary = "\n\n".join(errors)

    return (
        None,
        "Gemini analysis failed after trying multiple models.\n\n"
        + error_summary,
    )

# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def analyze_text(text: str) -> DiagnosticReport:
    """
    Main analysis entrypoint.

    Pipeline:

        1. Validate input
        2. Run local rules
        3. If local rules find something, return those results
        4. Otherwise ask Gemini for deeper semantic analysis
        5. Clearly distinguish success from API failure
    """

    cleaned_text = text.strip()

    if not cleaned_text:
        return DiagnosticReport(
            original_text=text,
            total_biases_found=0,
            detected_biases=[],
            analysis_method="Input validation",
            error_message="Please enter some text to analyze.",
        )

    # --------------------------------------------------------
    # Stage 1: Local rule-based analysis
    # --------------------------------------------------------

    rule_results = analyze_with_rules(cleaned_text)

    if rule_results:
        return DiagnosticReport(
            original_text=cleaned_text,
            total_biases_found=len(rule_results),
            detected_biases=rule_results,
            analysis_method="Local rule-based analysis",
            error_message=None,
        )

    # --------------------------------------------------------
    # Stage 2: Gemini semantic analysis
    # --------------------------------------------------------

    gemini_results, gemini_error = _analyze_with_gemini(
        cleaned_text
    )

    # IMPORTANT:
    #
    # None = analysis FAILED
    #
    # [] = analysis SUCCEEDED and found no biases
    #
    # This is the key bug fixed from the old version.
    if gemini_results is None:
        return DiagnosticReport(
            original_text=cleaned_text,
            total_biases_found=0,
            detected_biases=[],
            analysis_method="Gemini AI analysis",
            error_message=gemini_error,
        )

    return DiagnosticReport(
        original_text=cleaned_text,
        total_biases_found=len(gemini_results),
        detected_biases=gemini_results,
        analysis_method="Gemini AI semantic analysis",
        error_message=None,
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🧠 Cognitive Bias Analyzer")

st.write(
    "Analyze text for System 1 heuristics, emotional magnification, "
    "absolute thinking, and other cognitive biases."
)

st.markdown("---")


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

st.subheader("Input Text to Analyze")

default_text = (
    "She is so kind and well-spoken, so there is no doubt "
    "her software architecture will be reliable."
)

text_input = st.text_area(
    "Enter a sentence, paragraph, argument, or claim:",
    value=default_text,
    height=180,
    placeholder="Type or paste text here...",
)


# ------------------------------------------------------------
# Optional configuration display
# ------------------------------------------------------------

with st.expander("⚙️ Configuration"):
    model_name = get_model_name()

    st.write(f"**Gemini model:** `{model_name}`")

    api_key_present = bool(get_api_key())

    if api_key_present:
        st.success("Gemini API key detected.")
    else:
        st.warning(
            "Gemini API key not detected. "
            "Add GEMINI_API_KEY to Streamlit Secrets."
        )

    st.caption(
        "You can override the model with the GEMINI_MODEL environment "
        "variable or Streamlit Secret."
    )


# ------------------------------------------------------------
# Run button
# ------------------------------------------------------------

if st.button(
    "Run Diagnostic Report",
    type="primary",
    use_container_width=True,
):

    if not text_input.strip():
        st.warning("Please enter some text first.")

    else:
        with st.spinner("Analyzing your text..."):

            report = analyze_text(text_input)

        # ----------------------------------------------------
        # ERROR STATE
        # ----------------------------------------------------

        if report.error_message:

            st.error(
                f"❌ Analysis could not be completed\n\n"
                f"{report.error_message}"
            )

            st.info(
                "This is an analysis error, not a conclusion that "
                "the text contains zero cognitive biases."
            )

        # ----------------------------------------------------
        # SUCCESS STATE
        # ----------------------------------------------------

        else:

            st.success(
                f"Analysis completed using "
                f"**{report.analysis_method}**."
            )

            st.markdown("---")

            # ------------------------------------------------
            # Results heading
            # ------------------------------------------------

            if report.total_biases_found == 0:

                st.subheader("Results: 0 Bias(es) Detected")

                st.success(
                    "No obvious cognitive biases were detected "
                    "in the provided text."
                )

            else:

                st.subheader(
                    f"Results: {report.total_biases_found} "
                    f"Bias(es) Detected"
                )

                # --------------------------------------------
                # Individual bias cards
                # --------------------------------------------

                for index, bias in enumerate(
                    report.detected_biases,
                    start=1,
                ):

                    with st.container(border=True):

                        st.markdown(
                            f"### {index}. {bias.bias_name}"
                        )

                        st.markdown(
                            f"**Category:** "
                            f"{bias.category}"
                        )

                        st.markdown(
                            f"**Trigger:** "
                            f"`{bias.trigger_lemma}`"
                        )

                        st.markdown(
                            "**System 2 Reframe:**"
                        )

                        st.info(
                            bias.reframe_prompt
                        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Cognitive Bias Analyzer • Rule-based detection + Gemini semantic analysis"
)