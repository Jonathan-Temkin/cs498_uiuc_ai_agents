def calculate(operation, a, b):
    """Perform basic arithmetic."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b
    

def define_calc_tool():
    """
    Return a dictionary representing a tool schema for a weather lookup function.

    The schema should follow the Anthropic tool definition format with:
    - name: "get_weather"
    - description: a clear description of the tool's purpose
    - input_schema: JSON Schema with location (required) and unit (optional) parameters

    Returns:
        dict: A tool schema dictionary
    """
    tool_definition = {
        "name": "calc",
        "description": "does the calc",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum" : ["add", "subtract", "multiply", "divide"], "description": "calculator"},
                "a": {"type": "integer", "description": "int a to perform calc on"},
                "b": {"type": "integer", "description": "int b to perform calc on"},
            },
            "required": ["operation", "a", "b"]
        }
    }
    return tool_definition
