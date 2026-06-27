from typing import Dict, Any
import re
import json  # type: ignore

from miniml.inference.engines.engine_frame import AgentParserResponse

class OuputTagsNotFoundError(Exception):
    """Exception raised when output tags are not found in the response."""
    pass

class JSONParsingError(Exception):
    """Exception raised when JSON parsing fails."""
    pass

class AgentApplyRefsValidationParser:
    valid_map = {
        "true": True,
        "false": False
    }
    
    def __init__(self, response: str):
        self.response = response
    
    def parse(self) -> AgentParserResponse:
        if not "<output>" in self.response.lower() or not "</output>" in self.response.lower():
            try:
                import json
                data: Dict[str, Any] = json.loads(self.response)

                return AgentParserResponse(
                    instance_="success",
                    valid=self.valid_map.get(str(data.get("valid", True)).lower(), False),
                    reasoning=data.get("reasoning", "")
                )
            except:
                return AgentParserResponse(
                    instance_="error",
                    error=f"{str(OuputTagsNotFoundError.__name__)} {str(OuputTagsNotFoundError.__doc__)}"
                )

        match = re.search(r"<output>(.*?)</output>", self.response, re.DOTALL)
        if not match:
            return AgentParserResponse(
                instance_="error",
                error=f"{str(OuputTagsNotFoundError.__name__)} {str(OuputTagsNotFoundError.__doc__)}"
            )
        
        output_content = match.group(1).strip()
        try:
            import json
            data: Dict[str, Any] = json.loads(output_content)

            print("[VALIDATION DATA JSON]", data)

            if not data:
                return AgentParserResponse(
                    instance_="error",
                    error=f"{str(OuputTagsNotFoundError.__name__)} {str(OuputTagsNotFoundError.__doc__)}"
                )
            
            return AgentParserResponse(
                instance_="success",
                valid=self.valid_map.get(str(data.get("valid", True)).lower(), False),
                reasoning=data.get("reasoning", "")
            )
        except json.JSONDecodeError:  # type: ignore
            return AgentParserResponse(
                instance_="error",
                error=f"{str(JSONParsingError.__name__)} {str(JSONParsingError.__doc__)}"
            )

