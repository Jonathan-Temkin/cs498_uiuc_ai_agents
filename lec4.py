import requests
import re
from key import *

def define_weather_tool():
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
        "name": "get_weather",
        "description": "gets the weather",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "..."},
                "unit": {"type": "string", "enum": ['celsius', 'fahrenheit'], "description": "..."}
            },
            "required": ["location"]
        }
    }
    return tool_definition


def call_with_tools(prompt, tools, api_key):
    """
    Make an API call to Claude with tool definitions.

    Args:
        prompt (str): The user prompt
        tools (list): A list of tool schema dictionaries
        api_key (str): Your Anthropic API key

    Returns:
        dict: The full API response JSON

    Example:
        tools = [{"name": "get_weather", "description": "...", "input_schema": {...}}]
        response = call_with_tools("What's the weather in Paris?", tools, api_key)
        # response["stop_reason"] might be "tool_use"
    """
    print('call with tools')
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
        #"messages": [ {"role":"user", "content": prompt} ],
        # "temperature": 0, # smaller # is more deterministic
        "messages": prompt,
        "tools" : tools,
        #"stop_sequences" : ["</sentiment>"]
    }
    response = requests.post(url, json=body, headers=headers)
    #print("RESPONSE JSON: ", response.json())
    #msg = response.json()["content"][0]["text"]
    # msg_prefilled = prefill + msg
    # print(msg)
    # return msg
    dict_return = dict(response.json())
    print(dict_return)
    return dict_return

tools = define_weather_tool()
prompt = "What's the weather in Paris right now?"
# call_with_tools(prompt, tools, api_key)



def extract_tool_call(response):
    """
    Extract tool call information from an API response.

    Args:
        response (dict): The full API response dictionary

    Returns:
        dict or None: A dict with "tool_call_id", "tool_name", "arguments" keys,
                      or None if no tool_use block is found

    Example:
        response = {
            "content": [
                {"type": "text", "text": "Let me check..."},
                {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "Paris"}}
            ],
            "stop_reason": "tool_use"
        }
        result = extract_tool_call(response)
        # Returns: {"tool_call_id": "toolu_123", "tool_name": "get_weather", "arguments": {"location": "Paris"}}
    """
    contents = response["content"]
    for content in contents:
        if content["type"] == "tool_use":
            tool_id = content["id"]
            tool_name = content["name"]
            tool_arguments = content["input"]
            return {"`tool_call`_id": tool_id, "tool_name": tool_name, "arguments": tool_arguments}


response = { "content": [
        {"type": "text", "text": "Let me check..."},
        {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "Paris"}}
    ],
    "stop_reason": "tool_use" }

#print(extract_tool_call(response))


def build_tool_result_messages(original_user_prompt, assistant_response_content, tool_use_id, tool_result_str):
    """
    Build the messages array to continue a conversation after a tool call.

    Args:
        original_user_prompt (str): The original user prompt
        assistant_response_content (list): The content blocks from the assistant's tool_use response
        tool_use_id (str): The tool_use ID to match
        tool_result_str (str): The string result of the tool execution

    Returns:
        list: A list of 3 message dicts [user_msg, assistant_msg, tool_result_msg]

    Example:
        msgs = build_tool_result_messages(
            "What's the weather?",
            [{"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "Paris"}}],
            "toolu_123",
            "22 degrees Celsius, sunny"
        )
    """
    user_prompt =  {"role":"user", "content": original_user_prompt} 
    assistance_response = {"role":"assistant", "content": assistant_response_content} 
    if type(tool_result_str) == str:
        user_tool = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tool_use_id, "content": tool_result_str}]}
        return [user_prompt, assistance_response, user_tool]
    elif type(tool_result_str) == list:
        pass


def calculate(operation, a, b):
    """A simple calculator function."""
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




def tool_call_loop(prompt, api_key):
    """
    Implement the complete tool calling loop.

    1. Define a tool schema for the calculate function
    2. Call the API with the tool
    3. If the model requests a tool call, execute calculate() and send the result back
    4. Return the model's final text response

    Args:
        prompt (str): The user prompt (e.g., "What is 15 * 7?")
        api_key (str): Your Anthropic API key

    Returns:
        str: The model's final text response
    """

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
        "messages": [ {"role":"user", "content": prompt} ],
        # "temperature": 0, # smaller # is more deterministic
        # "messages": [ {"role":"user", "content": prompt},
        #              {"role":"assistant", "content": prefill} ],
        "tools" : [define_calc_tool()],
        #"stop_sequences" : ["</sentiment>"]
    }
    response = requests.post(url, json=body, headers=headers)
    response = response.json()
    print("RESPONSE JSON: ", response)
    operation = response["content"][0]["input"]["operation"]
    a = response["content"][0]["input"]["a"]
    b = response["content"][0]["input"]["b"]
    calc_result = calculate(operation, a, b)
    print("CALC RESULT: ", a, b, operation,  calc_result)
    new_message = build_tool_result_messages(prompt, response["content"], response["content"][0]["id"], str(calc_result))
    body["messages"] = new_message
    final_result =  requests.post(url, json=body, headers=headers)
    msg = final_result.json()["content"][0]["text"]
    print(msg)
    return msg

# prompt = "5 + 4"
# tool_call_loop(prompt, api_key)

def dispatch_tool_call(tool_name, arguments, tool_registry):
    """
    Dispatch a tool call to the correct function from the registry.

    Args:
        tool_name (str): The name of the tool to call
        arguments (dict): The arguments to pass as keyword arguments
        tool_registry (dict): Maps tool names to callable functions

    Returns:
        dict: {"status": "success", "result": <return_value>} or
              {"status": "error", "result": <error_message>}

    Example:
        def add(a, b): return a + b
        registry = {"add": add}
        dispatch_tool_call("add", {"a": 1, "b": 2}, registry)
        # Returns: {"status": "success", "result": 3}
    """

    if tool_name not in tool_registry:
        return {"status": "error", "result": f"Unknown tool: {tool_name}"}
    try:
        fun = tool_registry[tool_name]
        result = fun(*arguments.values())
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "result": str(e)}




def call_with_tool_choice(prompt, tools, tool_choice, api_key):
    """
    Make an API call with a specified tool_choice mode.

    Args:
        prompt (str): The user prompt
        tools (list): A list of tool schema dicts
        tool_choice (dict): The tool_choice parameter, e.g. {"type": "auto"}, {"type": "any"}, or {"type": "tool", "name": "get_weather"}
        api_key (str): Your Anthropic API key

    Returns:
        tuple: (stop_reason, content) where stop_reason is a string
               and content is the list of content blocks

    Example:
        stop_reason, content = call_with_tool_choice(
            "Hello", tools, {"type": "any"}, api_key
        )
        # stop_reason might be "tool_use"
    """

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
        "messages": [ {"role":"user", "content": prompt} ],
        # "temperature": 0, # smaller # is more deterministic
        # "messages": [ {"role":"user", "content": prompt},
        #              {"role":"assistant", "content": prefill} ],
        "tools" : [tools],
        "tool_choice" : tool_choice
        #"stop_sequences" : ["</sentiment>"]
    }
    response = requests.post(url, json=body, headers=headers)
    response = response.json()
    print("RESPONSE JSON: ", response)
    stop_reason = response["stop_reason"]
    content = response["content"]
    return response
    #return  (stop_reason, content) 

tool_choice = {"type": "any"}
#call_with_tool_choice(prompt, tools, tool_choice, api_key)


def validate_tool_arguments(arguments, schema):
    """
    Validate tool call arguments against the input schema.

    Args:
        arguments (dict): The arguments from the model's tool call
        schema (dict): The input_schema with "properties" and "required"

    Returns:
        dict: {"valid": True, "errors": []} if valid,
              {"valid": False, "errors": ["error msg 1", ...]} if invalid

    Example:
        schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
        validate_tool_arguments({"location": "Paris"}, schema)
        # Returns: {"valid": True, "errors": []}
        validate_tool_arguments({"unit": "celsius"}, schema)
        # Returns: {"valid": False, "errors": ["Missing required argument: location"]}
    """
    errors = []
    required_fields_lst = schema['required']
    fields_lst = arguments.keys()

    type_mapping = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
    }

    for field in fields_lst:
        # print(field)
        field_properties = schema["properties"][field]
        # print(field_properties)
        field_arg = arguments[field]
        if "enum" in field_properties.keys():
            if not (field_arg in field_properties["enum"]):
                #print(field_arg, field_properties["enum"])
                errors.append(f"invalid enum value '{field}': {field_arg}")
        if "type" in field_properties.keys():   
            print(field_properties["type"])
            if not (type(field_arg) == type_mapping[str(field_properties["type"])]):
                errors.append(f"invalid type value '{field}': {field_arg}")
    for field in required_fields_lst:
        #field_schema = schema["properties"][required_field]
            if not (field in arguments.keys()):
                errors.append(f"Missing required argument: {field}")
    if len(errors) == 0:
        return {"valid": True, "errors":[]} 
    return {"valid": False, "errors":errors}
    


    
schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
}



def tool_call(messages, tools, tool_choice, api_key):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    if type(messages) == str:
        messages = [ {"role":"user", "content": messages} ] 
    body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": """good boy good agent""",
        "messages": messages,
        # "temperature": 0, # smaller # is more deterministic
        # "messages": [ {"role":"user", "content": prompt},
        #              {"role":"assistant", "content": prefill} ],
        "tools" : tools if type(tools) == list else [tools],
        "tool_choice" : tool_choice
        #"stop_sequences" : ["</sentiment>"]
    }
    response = requests.post(url, json=body, headers=headers)
    response = response.json()
    print(response)
    return response




def multi_turn_tool_loop(prompt, tools, tool_executor, api_key):
    """
    Run a multi-turn tool calling loop until the model gives a final answer.

    Args:
        prompt (str): The user prompt
        tools (list): A list of tool schema dicts
        tool_executor (callable): A function(tool_name, arguments) -> str
        api_key (str): Your Anthropic API key

    Returns:
        str: The model's final text response

    Example:
        def my_executor(name, args):
            if name == "get_weather":
                return f"25°C in {args['location']}"
            return "Unknown"

        result = multi_turn_tool_loop(
            "Compare weather in Paris and Tokyo",
            [weather_tool],
            my_executor,
            api_key
        )
    """
    tool_choice = {"type": "auto"}
    result = tool_call(prompt, tools, tool_choice, api_key)
    original_user_prompt = {"role":"user", "content": prompt} 
    messages = [original_user_prompt]
    stop_reason = result['stop_reason']
    while stop_reason == "tool_use":
        print(result)
        fun_content = []
        assistant_response_content =  {
            "role": "assistant",
            "content": result["content"]
        }
        for item in result['content']:
            if item['type'] == 'tool_use':
                tool_name = item['name']
                tool_use_id = item['id']
                arguments = item['input']
                tool_result_str = str(dispatch_tool_call(tool_name, arguments, tool_executor)["result"])
                #tool_result_str = str(tool_executor(tool_name, arguments))
                print('\nTOOL RESULT', tool_name, arguments, tool_result_str)
                current_result = {'type': 'tool_result', 'tool_use_id': tool_use_id, 'content': tool_result_str}
                fun_content.append(current_result)
        new_user_msg = {'role': 'user', 'content': fun_content}
        messages.append(assistant_response_content)
        messages.append(new_user_msg)
        print('NEW MSG', messages)
        new_result = tool_call(messages, tools, tool_choice, api_key)
        result = new_result
        print('NEW RESULT', result)
        stop_reason = result['stop_reason']
        print(stop_reason)
    assistant_response_content =  {
            "role": "assistant",
            "content": result["content"]
        }
    messages.append(assistant_response_content)
    print(assistant_response_content)
    return assistant_response_content['content'][0]['text']


# prompt = "price of bananna and apple"
# tools = [define_lookup_price(), define_calc_tool()]
# tool_executor  = tool_fun

#multi_turn_tool_loop(prompt, tools, tool_fun, api_key)

#tool_call(prompt, tools, tool_choice, api_key)

# multi_turn_tool_loop(prompt, tools, tool_executor, api_key)


def define_lookup_price():
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
        "name": "lookup_price",
        "description": "look up item price",
        "input_schema": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "item to look up"},
            },
            "required": ["item"]
        }
    }
    return tool_definition

define_lookup_price()
define_calc_tool()


PRICES = {"apple": 1.50, "banana": 0.75, "orange": 2.00, "milk": 3.50, "bread": 2.50}

def lookup_price(item):
    """Look up the price of an item. Returns 0.0 if not found."""
    return PRICES.get(item.lower(), 0.0)

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

tool_fun = {
'lookup_price': lookup_price,
'calc' : calculate
}


def shopping_agent(question, api_key):
    """
    A shopping agent that can look up prices and do math to answer questions.

    Args:
        question (str): A shopping-related question
        api_key (str): Your Anthropic API key

    Returns:
        str: The agent's final text answer

    Example:
        shopping_agent("How much does one apple cost?", api_key)
        # Might return: "One apple costs $1.50."
    """
    # TODO: Implement this function
    # 1. Define tool schemas for lookup_price and calculate
    # 2. Run the tool call loop
    # 3. Dispatch tool calls to the correct function
    # 4. Return the final text response
    prompt = question
    tool_executor = tool_fun
    tool_choice = {"type": "auto"}
    tools = [define_lookup_price(), define_calc_tool()]
    result = tool_call(prompt, tools, tool_choice, api_key)
    original_user_prompt = {"role":"user", "content": prompt} 
    messages = [original_user_prompt]
    stop_reason = result['stop_reason']
    while stop_reason == "tool_use":
        print(result)
        fun_content = []
        assistant_response_content =  {
            "role": "assistant",
            "content": result["content"]
        }
        for item in result['content']:
            if item['type'] == 'tool_use':
                tool_name = item['name']
                tool_use_id = item['id']
                arguments = item['input']
                tool_result_str = str(dispatch_tool_call(tool_name, arguments, tool_executor)["result"])
                #tool_result_str = str(tool_executor(tool_name, arguments))
                print('\nTOOL RESULT', tool_name, arguments, tool_result_str)
                current_result = {'type': 'tool_result', 'tool_use_id': tool_use_id, 'content': tool_result_str}
                fun_content.append(current_result)
        new_user_msg = {'role': 'user', 'content': fun_content}
        messages.append(assistant_response_content)
        messages.append(new_user_msg)
        print('NEW MSG', messages)
        new_result = tool_call(messages, tools, tool_choice, api_key)
        result = new_result
        print('NEW RESULT', result)
        stop_reason = result['stop_reason']
        print(stop_reason)
    assistant_response_content =  {
            "role": "assistant",
            "content": result["content"]
        }
    messages.append(assistant_response_content)
    print(assistant_response_content)
    return assistant_response_content['content'][0]['text']

# question = "how much is apple"
# shopping_agent(question, api_key)