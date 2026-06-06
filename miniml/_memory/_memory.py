# _memory.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class MemoryProfile:
    role: str
    stage: str
    task: str
    data: str



class ActiveMemory:

    _instance = None
    _initialized = False

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self) -> None:

        if not self._initialized:

            self._memory: list[MemoryProfile] = []

            self._initialized = True

    def get_memory(self):

        return self._memory

    def add_to_memory(self, data: MemoryProfile):

        self._memory.append(data)
    

    def memory_to_dict(self) -> Dict[str, Dict[str, str]] | Dict[Any, Any]:
        dict_: Dict[str, Dict[str, str]] = {}

        for mem in self._memory:
            dict_[mem.role] = {
                "stage": mem.stage,
                "task": mem.task,
                "data": mem.data
            }

        return dict_
    
    def memory_to_dict_stringify(self) -> str:
        return str(self.memory_to_dict())
    
    def get_last_memory_to_dict_stringify(self) -> Optional[str]:
        if not self._memory:
            return None
        
        return str(self._memory[-1])