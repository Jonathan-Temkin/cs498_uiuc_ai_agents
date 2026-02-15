
from key import api_key
import json
import requests
from claude_api import *
import json
import re
import lec4


def retry_api_call(prompt, api_key, max_retries=3):
    """
    Make an API call with retry logic for error handling.

    Args:
        prompt (str): The user prompt to send
        api_key (str): Your Anthropic API key
        max_retries (int): Maximum number of retry attempts (default 3)

    Returns:
        str: The text content from a successful API response

    Raises:
        Exception: If all retries fail, with message containing "failed after"

    Example:
        text = retry_api_call("Hello!", api_key)
        # Returns the model's response text

        text = retry_api_call("Hello!", "invalid-key", max_retries=2)
        # Raises Exception("API call failed after 2 retries")
    """
    count = 0
    while count <= max_retries:
        response = claude_api_call(prompt, api_key, return_type = 'response')
        status_code = response.status_code
        if status_code == 200:
            return response.json()['content'][0]['text']
        count += 1
    e = f"API call failed after {max_retries} retries"
    raise Exception(e)



# prompt = "jeffery epstein was he a good person"
# retry_api_call(prompt, api_key, max_retries=3)


def trim_conversation(messages, max_turns):
    """
    Keep only the last max_turns complete user-assistant turn pairs.

    Args:
        messages (list): A list of message dicts with "role" and "content"
        max_turns (int): Maximum number of turns to keep

    Returns:
        list: Trimmed messages list

    Example:
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ]
        trim_conversation(messages, 1)
        # Returns: [{"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]
    """
    num_messages = len(messages)
    num_to_keep = max_turns*2
    stop_num = num_messages - num_to_keep
    return messages[stop_num:]

messages = [
    {"role": "user", "content": "A"},
    {"role": "assistant", "content": "B"},
    {"role": "user", "content": "C"},
    {"role": "assistant", "content": "D"},
]

#print(trim_conversation(messages, 1))

def parse_any_format(text):
    """
    Auto-detect and parse text in JSON, XML, or plain text format.

    Args:
        text (str): The text to parse

    Returns:
        dict or str: Parsed JSON as dict, XML content as string, or plain text stripped

    Example:
        parse_any_format('{"key": "value"}')  # Returns: {"key": "value"}
        parse_any_format('<result>hello</result>')  # Returns: "hello"
        parse_any_format('just text')  # Returns: "just text"
    """
    if type(text) == dict: return text
    if re.search('<result>(.*?)</result>', text):
        return re.search('<result>(.*?)</result>', text).group(1).replace('<', '').replace('>', '')
    try: return text.json()
    except: return text

# text = '<result>hello</result>'
# print(parse_any_format(text))



def fill_template_and_call(template, variables, api_key):
    """
    Fill template placeholders with XML-wrapped values and call the API.

    Args:
        template (str): Template with {{key}} placeholders
        variables (dict): Dict mapping keys to values
        api_key (str): Your Anthropic API key

    Returns:
        str: The API response text

    Example:
        template = "Analyze: {{text}}"
        variables = {"text": "Hello world"}
        # Sends: "Analyze: <text>Hello world</text>"
    """
    filled_template = template
    for variable in variables.keys():
        value = variables[variable]
        fill_value = f"<{variable}>{value}<{variable}>" 
        replace_value = "{{" + str(variable) + "}}"
        print(filled_template, replace_value, fill_value)
        filled_template = filled_template.replace(replace_value, fill_value)
    response = claude_api_call(filled_template, api_key, return_type='json')
    return response['content'][0]['text']

# template =  "Summarize this document: {{document}}"
# variables = {"document": "The quick brown fox..."}
# print(fill_template_and_call(template, variables, api_key))


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


# name = "get_temp"
# description = "Get temperature"
# parameters = [
#     {"name": "city", "type": "string", "description": "City name", "required": True},
#     {"name": "unit", "type": "string", "description": "Unit", "required": False, "enum": ["C", "F"]}
# ]

# print(generate_tool_schema(name, description, parameters))


def extract_entities(text, api_key):
    """
    Extract entities from text using forced tool calling.

    Args:
        text (str): The text to extract entities from
        api_key (str): Your Anthropic API key

    Returns:
        dict: The tool call's input dict with keys: people, locations, dates

    Example:
        result = extract_entities("John met Mary in Paris on Monday.", api_key)
        # Returns something like:
        # {"people": ["John", "Mary"], "locations": ["Paris"], "dates": ["Monday"]}
    """
    tool_name = "report_entities"
    tool_description = "get report entities" 
    tool_parameters = [
    {"name": "people", "type": "array", "description": "names of people mentioned",  "items": {"type": "string"}, "required": True},
    {"name": "locations", "type": "array", "description": "places mentioned", "items": {"type": "string"}, "required": True},
    {"name": "dates", "type": "array", "description": "dates/times mentioned", "required": True}
    ]
    report_entities_tool = generate_tool_schema(tool_name, tool_description, tool_parameters)
    tool_choice = {"type": "tool", "name": "report_entities"}
    print(report_entities_tool)
    response = call_with_tools(text, report_entities_tool, tool_choice, api_key)
    print(response)
    return response['content'][0]['input']

# text = 'use the tool bro'
# print(extract_entities(text, api_key))



def resilient_agent(prompt, tools, tool_executor, api_key, max_rounds=5):
    """
    Run a tool calling loop with error recovery and round limits.

    Args:
        prompt (str): The user prompt
        tools (list): List of tool schema dicts
        tool_executor (callable): Function(tool_name, arguments) that may raise exceptions
        api_key (str): Your Anthropic API key
        max_rounds (int): Maximum number of API call rounds (default 5)

    Returns:
        str: Final text response, or "Max rounds exceeded" if limit reached

    Example:
        def my_executor(name, args):
            if name == "divide" and args["b"] == 0:
                raise ValueError("Cannot divide by zero")
            return args["a"] / args["b"]

        result = resilient_agent("Divide 10 by 2", [divide_tool], my_executor, api_key)
    """
    current_round = 1
    stop_reason = "tool_use" 
    messages = [{"role":"user", "content": prompt}]
    while current_round <= max_rounds and stop_reason == "tool_use":
        print(messages)
        response = call_with_tools(messages, tools, api_key)
        print(current_round, response)
        stop_reason = response['stop_reason']
        if stop_reason !="tool_use":
            break
        agent_response = response['content']
        agent_response_message = {"role":"assistant", "content": agent_response}
        messages.append(agent_response_message)
        tool_i = 0 if response['content'][0]['type'] == 'tool_use' else 1
        fun_inputs = response['content'][tool_i]['input'].values()
        try:
            fun_result = tool_executor(*fun_inputs)
            tool_use_id = response['content'][tool_i]['id']
            user_tool_result_msg = {'type': 'tool_result', 'tool_use_id': tool_use_id, 'content': str(fun_result)}

        except Exception as e:
            tool_use_id = response['content'][tool_i]['id']
            user_tool_result_msg = {"type": "tool_result", "tool_use_id": tool_use_id, "content": str(e), "is_error": True}
        user_tool_response_msg = {'role': 'user', 'content': [user_tool_result_msg]}
        messages.append(user_tool_response_msg)
        current_round += 1
    if current_round >= max_rounds:
         return "Max rounds exceeded"
    final_response = response['content'][0]['text']  
    print(final_response)
    return final_response
    

prompt = "add 5 + 17 then divide the sum by 0. If error then divide by 3"
tools = [define_calc_tool()]
tool_executor = calculate
#resilient_agent(prompt, tools, tool_executor, api_key, max_rounds=5)
#print(call_with_tools(prompt, tools, api_key))



def tool_call(messages, api_key, system_prompt, stop_sequence):
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
        "system": system_prompt,
        "messages": messages,
        # "temperature": 0, # smaller # is more deterministic
        # "messages": [ {"role":"user", "content": prompt},
        #              {"role":"assistant", "content": prefill} ],
        "stop_sequences" : stop_sequence
    }
    response = requests.post(url, json=body, headers=headers)
    response = response.json()
    return response


def generate_report(topic, api_key):
    """
    Generate a structured report about a topic using system prompts, prefill, and stop sequences.

    Args:
        topic (str): The topic to generate a report about
        api_key (str): Your Anthropic API key

    Returns:
        dict: A dictionary with keys: summary (str), key_points (list of str), conclusion (str)

    Example:
        result = generate_report("renewable energy", api_key)
        # Returns:
        # {
        #     "summary": "...",
        #     "key_points": ["point 1", "point 2", ...],
        #     "conclusion": "..."
        # }
    """
    prefill = '<report>'
    system_prompt = "produce XML with: <summary>, <key_points> (containing multiple <point> tags), and <conclusion>. MAKE SURE IT'S PROPERLY FORMATTED XML"
    messages =[ {"role":"user", "content": topic}, {"role":"assistant", "content": prefill} ]
    stop_sequences  =  ["</report>"]
    response = tool_call(messages, api_key, system_prompt, stop_sequences)
    print(response)  
    text_response = response['content'][0]['text']
    # if re.search('<summary>\s*(.*?)\s*</summary>', text_response):
    #     summary =  re.search('<summary>\s*(.*?)\s*</summary>', text_response).group(1).replace('<', '').replace('>', '')
    # if re.search('<conclusion>\s*(.*?)\s*</conclusion>', text_response):
    #     conclusion =  re.search('<conclusion>\s*(.*?)\s*</conclusion>', text_response).group(1).replace('<', '').replace('>', '')
    # if re.search('<point>\s*(.*?)\s*</point>', text_response):
    #     key_points_lst = [result for result in re.findall('<point>\s*(.*?)\s*</point>', text_response, flags=re.DOTALL)]
    # result = {
    #     "summary": summary,
    #     "key_points": key_points_lst,
    #     "conclusion": conclusion
    # }
    # print(result)
    result = None
    return result
# topic = "the island: a jeffry epstein story from the perspective of a victim. 200 words"
# generate_report(topic, api_key)



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

def calculate(operation, a, b):
    if operation == "add": return a + b
    elif operation == "subtract": return a - b
    elif operation == "multiply": return a * b
    elif operation == "divide": return a / b

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
                "a": {"type": "integer", "description": "int a to perform calc on"} ,
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


def research_agent(question, api_key):
    """
    A research agent that can search a knowledge base and calculate.

    Args:
        question (str): The research question to answer
        api_key (str): Your Anthropic API key

    Returns:
        dict: {"answer": str, "tools_used": list of tool names}

    Example:
        result = research_agent("What year was Python created?", api_key)
        # Returns: {"answer": "Python was created in 1991.", "tools_used": ["search_kb"]}
    """
    prompt = (question + 'return in this format: if q is what year was python created, response should be in this format starting and ending with { and }. no text dict structure as follows only: {"answer": "Python was created in 1991.", "tools_used": ["search_kb"]}. Gurantee that the resulyting string response can be sucessfully converted to python dict using json.loads. ')
    tools = [define_calc_tool(), define_search_tool()]
    current_round = 1
    stop_reason = "tool_use" 
    messages = [{"role":"user", "content": prompt}]
    max_rounds  = 20
    while current_round <= max_rounds and stop_reason == "tool_use":
        print(messages)
        response = lec4.call_with_tools(messages, tools, api_key)
        print(current_round, response)
        stop_reason = response['stop_reason']
        if stop_reason !="tool_use":
            break
        agent_response = response['content']
        agent_response_message = {"role":"assistant", "content": agent_response}
        messages.append(agent_response_message)
        tool_i = 0 if response['content'][0]['type'] == 'tool_use' else 1
        fun_inputs = response['content'][tool_i]['input'].values()
        fun_name = response['content'][tool_i]['name']
        try:
            fun_result = calculate(*fun_inputs) if fun_name == 'calculate' else search_kb(*fun_inputs)
            tool_use_id = response['content'][tool_i]['id']
            user_tool_result_msg = {'type': 'tool_result', 'tool_use_id': tool_use_id, 'content': str(fun_result)}

        except Exception as e:
            tool_use_id = response['content'][tool_i]['id']
            user_tool_result_msg = {"type": "tool_result", "tool_use_id": tool_use_id, "content": str(e), "is_error": True}
        user_tool_response_msg = {'role': 'user', 'content': [user_tool_result_msg]}
        messages.append(user_tool_response_msg)
        current_round += 1
    if current_round >= max_rounds:
         return "Max rounds exceeded"
    final_response = (response['content'][0]['text'])
    final_response = final_response[final_response.find("{"): final_response.rfind("}") + 1]
    print('FINAL RESPONSE:', final_response, type(final_response))
    
    final_response = json.loads(final_response.strip())
    print('FINAL RESPONSE', final_response, type(final_response))
    return final_response


# question = 'get value for pi and multiply by 3'
# research_agent(question, api_key)


# messages = [{"role": "user", "content": "What's the weather?"}]
# result = add_tool_result(messages, "tool_123", "Sunny, 72°F")

# Result:
[
    {"role": "user", "content": "What's the weather?"},
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool_123",
                "content": "Sunny, 72°F"
            }
        ]
    }
]

