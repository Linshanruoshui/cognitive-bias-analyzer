import spacy
from rich import print

# Load the spaCy model you just downloaded
nlp = spacy.load("en_core_web_sm")

sample_sentence = "We already spent $50,000 on this project, so obviously we cannot quit now!"
doc = nlp(sample_sentence)

print("[bold green]✔ spaCy setup verified successfully![/bold green]\n")
print(f"[bold cyan]Input Sentence:[/bold cyan] {sample_sentence}\n")

# Display token attributes extracted by the NLP model
print("[bold yellow]Linguistic Token Breakdown:[/bold yellow]")
for token in doc:
    if not token.is_punct:
        print(f" • [white]{token.text:<12}[/white] | Base Lemma: [italic cyan]{token.lemma_:<12}[/italic cyan] | POS Tag: [dim]{token.pos_}[/dim]")