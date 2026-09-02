from pydantic import BaseModel, Field
from typing import List

class BiasDetectionItem(BaseModel):
    bias_name: str = Field(description="Name of the detected cognitive bias")
    trigger_lemma: str = Field(description="Linguistic root word that triggered the rule")
    category: str = Field(description="Psychological classification")
    reframe_prompt: str = Field(description="System 2 reflective prompt for metacognition")

class DiagnosticReport(BaseModel):
    input_text: str = Field(description="Original input string evaluated")
    total_biases_found: int = Field(description="Number of biases detected")
    detected_biases: List[BiasDetectionItem] = Field(default_factory=list)