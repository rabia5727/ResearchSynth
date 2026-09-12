from typing import Any, Dict

def generate_json(prompt: str, schema: Any = None) -> Dict[str, Any]:
    """
    Generate JSON response from LLM (mock implementation).
    Provider selection (Gemini/Groq) belongs here.
    """
    raise NotImplementedError("This is a mock abstraction. Overwrite in tests.")
