import re 
from key import *
import requests

def define_calendar_resource(year):
    """
    Return a dictionary representing an MCP resource for accessing calendar events.

    Args:
        year (int): The year for the calendar resource

    Returns:
        dict: An MCP resource dictionary with uri, name, description, and mimeType
    """
    result = {
        "uri": f"calendar://events/{year}",
        "name": f"Calendar Events {year}",
        "description": f"Access calendar events for the year {year}",
        "mimeType": "application/json"
        }
    return result
        

def define_database_query_tool():
    """
    Return a dictionary representing an MCP tool schema for querying a database.

    The schema should follow the MCP tool definition format with:
    - name: "query_database"
    - description: a clear description of the tool's purpose
    - input_schema: JSON Schema with table_name and limit parameters (both required)

    Returns:
        dict: An MCP tool schema dictionary
    """
    result = {
    "name": "query_database",
    "description": "tool to query database",
    "input_schema": {
        "type": "object",
        "properties": {
            "table_name": {"type": "string", "description": "..."},
            "limit": {"type": "integer", "description": "..."}
        },
        "required": ["table_name", "limit"]
    }
    }
    return result 

def define_database_query_tool():
    """
    Return a dictionary representing an MCP tool schema for querying a database.

    The schema should follow the MCP tool definition format with:
    - name: "query_database"
    - description: a clear description of the tool's purpose
    - input_schema: JSON Schema with table_name and limit parameters (both required)

    Returns:
        dict: An MCP tool schema dictionary
    """

    result = {
    "name": "query_database",
    "description": "tool to query a database",
    "input_schema": {
        "type": "object",
        "properties": {
            "table_name": {"type": "string", "description": "..."},
            "limit": {"type": "integer", "description": "..."}
        },
        "required": ["table_name", "limit"]
    }
    }
    return result 


server_info = {
    "resources": [{"name": "calendar://events", "uri": "calendar://events"}],
    "tools": [{"name": "query_db"}, {"name": "send_email"}],
    "prompts": []
}


def list_mcp_primitives(server_info):
    """
    Extract and categorize available primitives from an MCP server's information.

    Args:
        server_info (dict): Dictionary with keys "resources", "tools", and "prompts",
                           each containing a list of dictionaries with at least a "name" key

    Returns:
        dict: A dictionary with three keys ("resources", "tools", "prompts"),
              each mapping to a list of names (strings)
    """
    result = {}
    for key in server_info.keys():
        tools = server_info[key]
        lst = []
        for tool in tools:
            lst.append(tool['name'])
        result[key] = lst
    return result 

#print(list_mcp_primitives(server_info))

def build_resource_uri(scheme, path, params):
    """
    Construct an MCP resource URI from its components.

    Args:
        scheme (str): The URI scheme (e.g., "calendar", "file", "database")
        path (str): The resource path (e.g., "events/2024", "tables/users")
        params (dict): Query parameters as a dictionary (e.g., {"limit": 10, "offset": 0})

    Returns:
        str: A complete URI string in the format:
             - "{scheme}://{path}" if params is empty
             - "{scheme}://{path}?key1=value1&key2=value2" if params has values (sorted by key)
    """
    if not params:
        return f"{scheme}://{path}"
    
    base  = f"{scheme}://{path}?"
    params = dict(sorted(params.items()))
    for i, param in enumerate(params):
        #key_i = 'key' + str(i+1)
        val = params[param]
        key_i = param
        separater = '&' if i > 0 else ''  
        base = base + separater + (key_i + '=' + str(val))
    return base 

#print(build_resource_uri("database", "tables/users", {"limit": 100, "offset": 0}))



template = "Analyze the {data_type} data from {source} and provide insights about {topic}."
parameters = {"data_type": "sales", "source": "Q4 2024", "topic": "revenue trends"}

def fill_prompt_template(template, parameters):
    """
    Fill in a prompt template with provided parameters.

    Args:
        template (str): A string containing placeholders in the format {param_name}
        parameters (dict): A dictionary mapping parameter names to values

    Returns:
        str: The filled template string. If a placeholder's parameter is missing,
             leave the placeholder as-is (don't replace it).
    """
    for key, val in parameters.items():
        search_val = '{' + key + '}'
        print(search_val, val)
        template = template.replace(search_val, str(val))
    return template 

#print(fill_prompt_template(template, parameters))


#mcp_client_connect("mcp://localhost:8080", "my_client")

# Returns: {
#     "status": "connected",
#     "server_url": "mcp://localhost:8080",
#     "client_name": "my_client",
#     "stages_completed": ["connect", "initialize", "discover", "execute"]
# }


def mcp_client_connect(server_url, client_name):
    """
    Simulate the MCP client connection flow and return the connection state.

    Args:
        server_url (str): The MCP server URL
        client_name (str): The client identifier

    Returns:
        dict: A dictionary representing the connection state with keys:
              - "status": should be "connected"
              - "server_url": matches the input parameter
              - "client_name": matches the input parameter
              - "stages_completed": a list with ["connect", "initialize", "discover", "execute"]
    """

    result = {
        "status": "connected",
        "server_url": server_url,
        "client_name": client_name,
        "stages_completed": ["connect", "initialize", "discover", "execute"]
    }

    return result


request = {"question": "Window or aisle seat?", "preference_key": "seat_preference"}
preferences = {"seat_preference": "window", "meal": "vegetarian"}
# Returns: {"answer": "window", "source": "stored_preference"}

def handle_server_request(request, preferences):
    """
    Process a server-initiated request using stored preferences.

    Args:
        request (dict): A dictionary with keys "question" (string) and "preference_key" (string)
        preferences (dict): A dictionary storing user/agent preferences

    Returns:
        dict: A response dictionary with keys "answer" and "source":
              - If preference_key exists: {"answer": value, "source": "stored_preference"}
              - If preference_key doesn't exist: {"answer": None, "source": "requires_user_input"}
    """
    preference_name = request['preference_key']
    try:
        preference = preferences[preference_name]
        result = {"answer": preference, "source": "stored_preference"}
        return result
    except:
        return {"answer": None, "source": "requires_user_input"}


def validate_mcp_token(token, current_time, trusted_servers):
    """
    Validate an MCP server's authentication token.

    Args:
        token (dict): A dictionary with keys "server_id", "issued_at" (timestamp in seconds),
                     and "expires_at" (timestamp in seconds)
        current_time (int): The current timestamp in seconds
        trusted_servers (list): A list of trusted server IDs (strings)

    Returns:
        dict: A validation result with keys "valid" (bool) and "reason" (str):
              - {"valid": True, "reason": "token_valid"} if all checks pass
              - {"valid": False, "reason": "untrusted_server"} if server is not trusted
              - {"valid": False, "reason": "token_expired"} if current_time >= expires_at
              - {"valid": False, "reason": "token_not_yet_valid"} if current_time < issued_at
              Check in this order: server trust, then time validity
    """
    result =  {"valid": True, "reason": "token_valid"}
    server_id = token["server_id"]
    issued_time = token['issued_at']
    expiration_time = token["expires_at"]
    if server_id not in trusted_servers:
        result =  {"valid": False, "reason": "untrusted_server"}
    if current_time >= expiration_time:
        result =  {"valid": False, "reason": "token_expired"}
    if current_time < issued_time:
        result =  {"valid": False, "reason": "token_not_yet_valid"}
    return result

tool_name =  "send_email"
arguments = {"to": "user@example.com", "subject": "Hello", "api_key": "secret123"}
sensitive_keys =  ["api_key", "password"]

def execute_mcp_tool_with_logging(tool_name, arguments, sensitive_keys):
    """
    Execute a tool and create a sanitized log entry.

    Args:
        tool_name (str): String identifying the tool
        arguments (dict): Dictionary of tool arguments
        sensitive_keys (list): List of argument keys that contain sensitive data
                              (e.g., ["password", "api_key", "token"])

    Returns:
        dict: A dictionary with:
              - "log_entry": A dict with "tool", "arguments" (sanitized), and "status"
              - "status": should be "executed"

              Sanitized arguments should have sensitive_keys values replaced with "***REDACTED***"
    """
    arguments["api_key"] = "***REDACTED***"
    result = {
        "log_entry": {
            "tool": tool_name,
            "arguments": arguments,
            "status": "executed"
        },
        "status": "executed"
    }
    return result

#print(execute_mcp_tool_with_logging(tool_name, arguments, sensitive_keys))

def lookup_calendar(date):
    """Simulates calendar lookup"""
    return f"Events on {date}: Team meeting at 10am, Lunch at 12pm"


def define_lookup_calendar_tool():
    """
    Return a dictionary representing an MCP tool schema for querying a database.

    The schema should follow the MCP tool definition format with:
    - name: "query_database"
    - description: a clear description of the tool's purpose
    - input_schema: JSON Schema with table_name and limit parameters (both required)

    Returns:
        dict: An MCP tool schema dictionary
    """
    result = {
    "name": "lookup_calendar",
    "description": "Look up calendar events for a specific date",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "YYYY-MM-DD"},
        },
        "required": ["table_name"]
    }
    }
    return result 


def mcp_tool_integration(prompt, api_key):
    """
    Implement a complete MCP tool calling flow using a calendar lookup tool.

    Args:
        prompt (str): The user prompt (e.g., "What's on my calendar for 2024-03-15?")
        api_key (str): Your Anthropic API key

    Returns:
        str: The final text response from the model

    Requirements:
        - Define a tool named "lookup_calendar" with description "Look up calendar events for a specific date"
        - Parameter: date (string, required) - format: YYYY-MM-DD
        - Make API call to https://api.anthropic.com/v1/messages
        - Use model: claude-sonnet-4-5-20250929, max_tokens: 1024
        - If tool call requested: extract date, call lookup_calendar(date), send result back
        - Return the final text content from the model
    """
    tool_definition = define_lookup_calendar_tool()
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": """good boy good agent""",
        "messages" : [ {"role":"user", "content": prompt} ],
        "tools" : [define_lookup_calendar_tool()],
        #"messages": conversation_history

    }
    response = requests.post(url, json=body, headers=headers)
    #print(response.json())
    response = response.json()
    if response['content'][0]['type'] == 'tool_use':
        input = response['content'][0]['input']
        id =  response['content'][0]['id']
        cal_result = lookup_calendar(input)
        assistant_response_content =  {
            "role": "assistant",
            "content":  response['content']
        }
        user_tool_response = [{'type': 'tool_result', 'tool_use_id': id, 'content': cal_result}]
        new_user_msg = {'role': 'user', 'content': user_tool_response}
        new_msg = [{"role": "user", "content": prompt}]
        new_msg.append(assistant_response_content)
        new_msg.append(new_user_msg)
        print(new_msg)
        body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": """good boy good agent""",
        "messages" : new_msg,
        "tools" : [define_lookup_calendar_tool()],
        #"messages": conversation_history
        }
        
        final_response = requests.post(url, json=body, headers=headers)
        return final_response.json()['content'][0]['text']

    else:
        return response.json()['content'][0]['text']
    print(response.json())




prompt = "What's on my calendar for 2024-03-15?"

print(mcp_tool_integration(prompt, api_key))