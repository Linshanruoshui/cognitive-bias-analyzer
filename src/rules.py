import spacy
from typing import List
from src.schemas import BiasDetectionItem, DiagnosticReport

# Load lightweight spaCy model
nlp = spacy.load("en_core_web_sm")

# Expanded Cognitive Bias Taxonomy
BIAS_TAXONOMY = [
    {
        "bias_name": "Sunk Cost Fallacy",
        "category": "Decision-Making Heuristic",
        "keywords": ["already", "invested", "spent", "waste", "put"],
        "mechanism": "Focusing on past unrecoverable resources rather than future utility.",
        "system_2_reframe": "If you had $0 and zero time invested as of today, would you still choose to pursue this?"
    },
    {
        "bias_name": "Absolutism / Confirmation Bias",
        "category": "Perceptual Shortcut",
        "keywords": ["always", "never", "everyone", "nobody", "obviously", "clearly", "impossible"],
        "mechanism": "Categorical overgeneralization that bypasses nuanced analysis of counter-evidence.",
        "system_2_reframe": "What is one plausible scenario where the exact opposite of this assumption holds true?"
    },
    {
        "bias_name": "Availability Heuristic",
        "category": "Probability Judgment",
        "keywords": ["recently", "saw", "heard", "friend", "yesterday", "lately", "anecdote"],
        "mechanism": "Overestimating likelihood based on how easily a recent or vivid memory comes to mind.",
        "system_2_reframe": "Is this pattern supported by broader statistical data, or primarily by a single recent memory?"
    },
    {
        "bias_name": "Negativity Bias",
        "category": "Affective Evaluation",
        "keywords": ["disaster", "terrible", "worst", "ruin", "fail", "catastrophe", "dangerous"],
        "mechanism": "Asymmetrically weighting negative risks over equivalent positive or neutral possibilities.",
        "system_2_reframe": "If the worst-case scenario occurs, what is your concrete mitigation plan, and what is the best-case outcome?"
    },
    {
        "bias_name": "Anchoring Bias",
        "category": "Numerical/Baseline Judgment",
        "keywords": ["originally", "initially", "starting", "baseline", "first", "quoted"],
        "mechanism": "Fixating on an initial piece of information to judge subsequent estimates or value.",
        "system_2_reframe": "If you ignored the original estimate completely, what would a fresh valuation look like from scratch?"
    }
]


def analyze_text(text: str) -> DiagnosticReport:
    """
    Parses input text using spaCy, evaluates linguistic lemmas against rules,
    and returns a structured Pydantic DiagnosticReport.
    """
    doc = nlp(text)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_punct]

    detected_items: List[BiasDetectionItem] = []

    for rule in BIAS_TAXONOMY:
        for keyword in rule["keywords"]:
            if keyword in lemmas:
                detected_items.append(
                    BiasDetectionItem(
                        bias_name=rule["bias_name"],
                        trigger_lemma=keyword,
                        category=rule["category"],
                        reframe_prompt=rule["system_2_reframe"]
                    )
                )
                break

    return DiagnosticReport(
        input_text=text,
        total_biases_found=len(detected_items),
        detected_biases=detected_items
    )