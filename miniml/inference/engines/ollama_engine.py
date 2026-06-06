from ollama import chat as OllamaChat    # type: ignore
from typing import Dict, Any, List, Optional

def get_response_ollama(prompt: str, 
                        model: str, 
                        options_dict: Dict[str, Any],  
                        tools: Optional[List[Dict[str, Any]]] = None, 
                        columns: Optional[List[str]] = None, 
                        use_columns: bool = False) -> Dict[Any, Any]:
    response: Dict[Any, Any] = OllamaChat(   # type: ignore
            model=model,
            messages = [
                {"role": "user", "content": prompt}
            ],
            options=options_dict
        )
    
    print("[Ollama Response]", response)
    
    return response
