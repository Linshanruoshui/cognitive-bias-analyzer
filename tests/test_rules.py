from src.rules import analyze_text

def test_sunk_cost_fallacy():
    text = "We have already spent so much money, we cannot quit now."
    report = analyze_text(text)
    biases = [b.bias_name for b in report.detected_biases]
    assert "Sunk Cost Fallacy" in biases

def test_absolutism_confirmation_bias():
    text = "Everyone knows this idea is obviously impossible."
    report = analyze_text(text)
    biases = [b.bias_name for b in report.detected_biases]
    assert "Absolutism / Confirmation Bias" in biases

def test_framed_urgency():
    text = "You must act now before it's too late!"
    report = analyze_text(text)
    biases = [b.bias_name for b in report.detected_biases]
    assert "Framed Urgency / Bandwagon" in biases

def test_clean_input():
    text = "We will review the data calmly tomorrow morning."
    report = analyze_text(text)
    assert report.total_biases_found == 0
