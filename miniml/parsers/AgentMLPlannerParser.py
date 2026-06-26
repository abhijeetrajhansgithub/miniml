from typing import Dict, Any
import re

from numpy.matlib import True_

class OuputTagsNotFoundError(Exception):
    """Exception raised when output tags are not found in the response."""
    pass

class JSONParsingError(Exception):
    """Exception raised when JSON parsing fails."""
    pass

class SubAgentModelSelectionGenerationParser:
    def __init__(self, response: str) -> None:
        self.response = response
    
    def parse(self):
        if not "<output>" in self.response.lower() or not "</output>" in self.response.lower():
            try:
                import json
                data: Dict[str, Any] = json.loads(self.response)
                
                return {
                        "_instance": "success",
                        "models": data.get("model") or []
                }
            except:
                return {
                    "_instance": "error",
                    "error": f"{str(OuputTagsNotFoundError.__name__)} {str(OuputTagsNotFoundError.__doc__)}"
                }
        
        match = re.search(r"<output>(.*?)</output>", self.response, re.DOTALL)

        if not match:
            return {
                "_instance": "error",
                "error": f"{str(OuputTagsNotFoundError.__name__)} {str(OuputTagsNotFoundError.__doc__)}"
            }
        
        output_content = match.group(1).strip()
        try:
            import json
            data: Dict[str, Any] = json.loads(output_content)
            
            return {
                "_instance": "success",
                "models": data.get("model") or []
            }
        except:
            return {
                "_instance": "error",
                "error": f"{str(JSONParsingError.__name__)} {str(JSONParsingError.__doc__)}"
            }