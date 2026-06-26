from typing import Dict, Any
import re
import json
from miniml.inference.engines.engine_frame import AgentParserResponse, LLMResponse, LLMSingleResponse

class OuputTagsNotFoundError(Exception):
    """Exception raised when output tags are not found in the response."""
    pass

class JSONParsingError(Exception):
    """Exception raised when JSON parsing fails."""
    pass

class AgentFindTargetGenerationParser:
    def __init__(self, response: str) -> None:
        self.response = response
    
    def parse(self) -> AgentParserResponse:
        if not "<output>" in self.response.lower() or not "</output>" in self.response.lower():
            try:
                import json
                data: Dict[str, Any] = json.loads(self.response)

                return AgentParserResponse(
                    instance_="success",
                    target_column=data.get("target_column", "")
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

            return AgentParserResponse(
                instance_="success",
                target_column=data.get("target_column", "")
            )
        except json.JSONDecodeError:
            return AgentParserResponse(
                instance_="error",
                error=f"{str(JSONParsingError.__name__)} {str(JSONParsingError.__doc__)}"
            )



class AgentFindTargetValidationParser:
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

            return AgentParserResponse(
                instance_="success",
                valid=self.valid_map.get(str(data.get("valid", True)).lower(), False),
                reasoning=data.get("reasoning", "")
            )
        except json.JSONDecodeError:
            return AgentParserResponse(
                instance_="error",
                error=f"{str(JSONParsingError.__name__)} {str(JSONParsingError.__doc__)}"
            )
