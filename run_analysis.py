from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.rules import analyze_text

console = Console()

def main():
    sample_text = (
        "I recently saw a project fail online, so changing our plan now will obviously lead to a complete disaster!"
    )

    console.print("\n", Panel(sample_text, title="[bold cyan]Input Text[/bold cyan]", expand=False))

    report = analyze_text(sample_text)

    if report.total_biases_found == 0:
        console.print("[bold green]No explicit System 1 cognitive biases detected![/bold green]")
        return

    table = Table(title=f"[bold yellow]Cognitive Diagnostic Report (Total Found: {report.total_biases_found})[/bold yellow]")
    table.add_column("Bias Detected", style="bold red")
    table.add_column("Trigger Lemma", style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("System 2 Reframing Prompt", style="green")

    for item in report.detected_biases:
        table.add_row(
            item.bias_name,
            item.trigger_lemma,
            item.category,
            item.reframe_prompt
        )

    console.print(table)

    # Optional: Print raw JSON output to demonstrate Pydantic serialization
    console.print("\n[bold dim]Serialized JSON Output (Pydantic Model):[/bold dim]")
    console.print(report.model_dump_json(indent=2))

if __name__ == "__main__":
    main()