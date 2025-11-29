from core.schemas import AggregatedContext

def call_llm(context: AggregatedContext) -> str:
    """
    Simuliert den Call an das LLM.
    """
    print("  [LLM] Generiere Antwort basierend auf gesammelten Daten...")
    
    # Hier würde der echte API Call stehen (z.B. OpenAI, Gemini)
    # Wir dumpen einfach den Context als JSON zur Demonstration
    context_json = context.model_dump_json(indent=2)
    
    return f"LLM Entscheidung basierend auf:\n{context_json}\n\n-> Ergebnis: Das Bild zeigt vermutlich eine Person (basierend auf Tool1) und der Text ist neutral (basierend auf Tool2)."
