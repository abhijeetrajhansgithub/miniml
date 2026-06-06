
class DatasetInferencingError(Exception):
    def __init__(self, message: str = "Dataset inferencing failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class ConfigFileError(Exception):
    def __init__(self, message: str = "Config file is not valid"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]


class DatasetLoadingError(Exception):
    def __init__(self, message: str = "Dataset loading failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class ApiKeyError(Exception):
    def __init__(self, message: str = "API key is not valid"):
        super().__init__(message)

    def __str__(self) -> str:
        
        return self.args[0]
    

class ToolError(Exception):
    def __init__(self, message: str = "Tool is not valid"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class InferenceError(Exception):
    def __init__(self, message: str = "Inference failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class AgentError(Exception):
    def __init__(self, message: str = "Agent failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class ValidationError(Exception):
    def __init__(self, message: str = "Validation failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class GenerationError(Exception):
    def __init__(self, message: str = "Generation failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class ModelError(Exception):
    def __init__(self, message: str = "Model failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class ModelTrainingError(Exception):
    def __init__(self, message: str = "Model training failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
    

class ModelSavingError(Exception):
    def __init__(self, message: str = "Model saving failed"):
        super().__init__(message)

    def __str__(self) -> str:
        return self.args[0]
