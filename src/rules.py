import spacy
from pydantic import BaseModel, Field
from typing import List

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class BiasDetection(BaseModel):
    bias_name: str
    trigger_lemma: str
    category: str
    reframe_prompt: str

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

def analyze_text(text: str) -> DiagnosticReport:
    doc = nlp(text)
    found_biases = []
    
    lemmas_in_text = [token.lemma_.lower() for token in doc]
    
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
    
    return DiagnosticReport(
        original_text=text,
        total_biases_found=len(found_biases),
        detected_biases=found_biases
    )
