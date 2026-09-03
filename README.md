[![Python CI](https://github.com/Linshanruoshui/cognitive-bias-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/Linshanruoshui/cognitive-bias-analyzer/actions)

# Cognitive Bias Analyzer 🧠
![Python CI](https://github.com/Linshanruoshui/cognitive-bias-analyzer/actions/workflows/ci.yml/badge.svg)

A lightweight, local NLP pipeline built with **spaCy**, **Pydantic**, and **Rich** to detect System 1 cognitive biases and generate metacognitive System 2 reframing prompts.

## Features
- **Deterministic Heuristic Detection**: Rule-based NLP parsing leveraging spaCy lemmatization.
- **Structured Pydantic Schemas**: Provides typed, validated outputs easily exportable to raw JSON.
- **Rich Terminal UI**: Displays interactive diagnostic summary tables directly in your CLI.

## Quick Start

### 1. Clone & Set Up Virtual Environment
```bash
git clone [https://github.com/YOUR_USERNAME/cognitive-bias-analyzer.git](https://github.com/YOUR_USERNAME/cognitive-bias-analyzer.git)
cd cognitive-bias-analyzer
python -m venv venv
```

### 2. Activate Environment
- **Windows (PowerShell)**:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
  .\venv\Scripts\Activate.ps1
  ```
- **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies & spaCy Model
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run Analysis
```bash
python run_analysis.py
```
