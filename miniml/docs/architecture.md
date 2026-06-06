# Active Memory Architecture

The active memory system is designed to provide a flexible and extensible way to manage and access data within the MiniML framework. It allows for the creation of different types of memory profiles, each with their own set of data and functionality.

### MemoryProfile Class

The `MemoryProfile` class is the base class for all memory profiles. It provides a way to store and access data in a structured manner. Each memory profile has a name and a dictionary of data.

### ActiveMemory Class

The `ActiveMemory` class is responsible for managing multiple memory profiles. It provides methods to add, remove, and access memory profiles, as well as to get the current active memory profile.

### Functions Supported for Memory Management

- `get_memory()`: Returns the list of all memory profiles.
- `add_to_memory(data: MemoryProfile)`: Adds a memory profile to the active memory.
- `memory_to_dict()`: Converts the memory profiles to a dictionary.
- `memory_to_dict_stringify()`: Converts the memory profiles to a string representation.
- `get_last_memory_to_dict_stringify()`: Returns the string representation of the last memory profile.
- `get_memory_by_role(role: str)`: Returns the memory profile with the specified role.
- `get_memory_by_stage(stage: str)`: Returns the memory profile with the specified stage.
- `get_memory_by_task(task: str)`: Returns the memory profile with the specified task.
- `get_memory_by_params(**kwargs)`: Returns the memory profile with the specified parameters.
- `get_last_n_memories(n: int)`: Returns the last n memory profiles.

