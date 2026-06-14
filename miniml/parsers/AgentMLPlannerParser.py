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
        pass