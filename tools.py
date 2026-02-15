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
    

KNOWLEDGE_BASE = {
    "python": "Python is a programming language created by Guido van Rossum in 1991.",
    "earth": "Earth is the third planet from the Sun with a radius of 6371 km.",
    "pi": "Pi (π) is approximately 3.14159265359.",
}


def search_kb(query):
    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            print('SEARCH TOOL ', value)
            return value
    return "No relevant information found."



def generate_tool_schema(name, description, parameters):
    """
    Generate an Anthropic tool schema from parameter specifications.

    Args:
        name (str): The tool name
        description (str): The tool description
        parameters (list): List of parameter spec dicts, each with:
            - name (str): Parameter name
            - type (str): Parameter type ("string", "number", "boolean")
            - description (str): Parameter description
            - required (bool): Whether the parameter is required
            - enum (list, optional): List of allowed values

    Returns:
        dict: A complete tool schema dictionary

    Example:
        params = [{"name": "x", "type": "number", "description": "A number", "required": True}]
        generate_tool_schema("double", "Double a number", params)
    """
    properties = {}
    required = []
    for param in parameters:
        param_name = param["name"]
        desc = {}
        for key in param.keys():
            val = param[key]
            if key not in  ['name', 'required']:
                desc[key] = val
            if key == 'required':
                is_required = param[key]
                if is_required: required.append(param_name)
        properties[name] = desc
    result =  {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
    return result 



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


def define_search_tool():

    tool_definition = {
        "name": "search",
        "description": "search db",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "str query to use for searching database"},
            },
            "required": ["query"]
        }
    }
    return tool_definition