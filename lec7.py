from claude_api import *
import key
from tools import generate_tool_schema
import requests

def generate_tool_result(tool_result, tool_use_id):
    result = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_result
            }
        ]
    }
    return result

def add_tool_result(messages, tool_use_id, tool_result):
    """
    Add a tool result to the messages array.

    Args:
        messages (list): Current messages array
        tool_use_id (str): ID of the tool use to respond to
        tool_result (str): The result from executing the tool

    Returns:
        list: Updated messages array with tool result added
    """
    result = generate_tool_result(tool_result, tool_use_id) 
    messages_tool_result = messages + [result]
    print(messages_tool_result)
    return messages_tool_result
   
# messages = [{"role": "user", "content": "What's the weather?"}]
# add_tool_result(messages, "abc123", "it's nice today")


def add_assistant_response(messages, response_content):
    """
    Add an assistant's response to the messages array.

    Args:
        messages (list): Current messages array
        response_content (list): Content blocks from the API response

    Returns:
        list: Updated messages array with assistant response added
    """
    assistant_response = {
        "role": "assistant",
        "content": response_content
        }
    messages_w_response = messages + [assistant_response]
    print(messages_w_response)
    return messages_w_response



messages = [{"role": "user", "content": "What's the weather?"}]
response_content = [
    {"type": "text", "text": "I'll check that."},
    {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"city": "Paris"}}
]

#add_assistant_response(messages, response_content)


def determine_next_action(stop_reason):
    """
    Determine what the agent should do next based on stop_reason.

    Args:
        stop_reason (str): The stop_reason from the API response

    Returns:
        str: The next action ("execute_tools", "return_answer", "handle_error", or "unknown")

    If stop_reason is "tool_use", return "execute_tools"
    If stop_reason is "end_turn", return "return_answer"
    If stop_reason is "max_tokens", return "handle_error"
    If stop_reason is "stop_sequence", return "return_answer"
    """
    next_action = {
        "tool_use" : "execute_tools",
        "end_turn" : "return_answer",
        "max_tokens" : "handle_error",
        "stop_sequence" : "return_answer"
    }
    try:
        return next_action[stop_reason.lower()]
    except:
        return "unknown"
    
def extract_all_tool_uses(response):
    """
    Extract all tool use blocks from an API response.

    Args:
        response (dict): A full API response dictionary

    Returns:
        list: List of dicts with tool_use_id, tool_name, and arguments
    """
    # TODO: Implement this function
    # 1. Get the content blocks from response
    # 2. Find ALL tool_use blocks (not just the first)
    # 3. Extract id, name, and input from each
    # 4. Return list of tool use dicts
    content = response['content']
    result = []
    for content_block in content:
        if content_block["type"] == "tool_use":
            tool_block = {"tool_use_id": content_block["id"], "tool_name": content_block["name"], "arguments": content_block["input"]}
            result.append(tool_block)
    return result

response = {
    "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"query": "Python"}},
        {"type": "tool_use", "id": "toolu_2", "name": "search", "input": {"query": "Java"}}
    ]
}

#print(extract_all_tool_uses(response))


def should_terminate(current_step, max_steps, last_action, goal_reached):
    """
    Determine if the agent execution loop should terminate.

    Args:
        current_step (int): Current step number
        max_steps (int): Maximum allowed steps
        last_action (str): The most recent action taken
        goal_reached (bool): Whether the goal has been achieved

    Returns:
        bool: True if loop should stop, False otherwise
    """
    if goal_reached or current_step >= max_steps or last_action.lower() == "finish":
        return True
    return False




KNOWLEDGE_BASE = {
    "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "react framework": "ReAct is a framework that combines reasoning and acting in language models.",
    "agents": "AI agents are systems that use LLMs to take actions and make decisions autonomously."
}

def search_knowledge_base(query):
    """Simulated search function that looks up information in a knowledge base."""
    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return value
    return "No relevant information found."

tool_dict = {
    'search_database': search_knowledge_base,
}

def tool_executor(tool_name):
    try:
        return tool_dict[tool_name]
    except:
        raise Exception("The supplied tool name is not valid")


def react_agent(question, api_key):
    """
    Implement a ReAct agent that can use a search tool.

    Args:
        question (str): The question to answer
        api_key (str): Anthropic API key

    Returns:
        str: The final answer from the agent
    """
    tool_name = 'search_database'
    tool_description = 'Simulated search function that looks up information in a knowledge base'
    tool_parameters = [{"name": "query", "type": "string", "description": "db query", "required": True}]
    tool_schema  = generate_tool_schema(tool_name, tool_description, tool_parameters)
    tools = [tool_schema]
    result = react_agent(question, api_key, tools, tool_executor, max_rounds=20, return_type = None)
    return result 

# question = "what is python"
# react_agent(question, key.api_key)


def prompt_chain_workflow(text, target_language, api_key):
    """
    Execute a three-step workflow: Summarize → Translate → Format.

    Args:
        text (str): Input text to process
        target_language (str): Target language for translation (e.g., "French", "Spanish")
        api_key (str): Anthropic API key

    Returns:
        str: Final formatted announcement
    """
    # TODO: Implement the three-step workflow
    # Step 1: Summarize the text to 2 sentences
    # Step 2: Translate the summary to target_language
    # Step 3: Format as a professional announcement
    step_1_instructions = "create a 2 sentence summary"
    step_2_instructions = f"translate the text into {target_language}"
    step_3_instructions = "Format as a professional announcement with a title and the translated text"
    step_1 = claude_api_call(text, api_key, return_type='json', system_prompt = step_1_instructions, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False)
    step_1_text = step_1['content'][0]['text']
    step_2 = claude_api_call(step_1_text, api_key, return_type='json', system_prompt = step_2_instructions, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False)
    step_2_text = step_2['content'][0]['text']
    step_3 = claude_api_call(step_2_text, api_key, return_type='json', system_prompt = step_3_instructions, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False)
    step_3_text = step_2['content'][0]['text']
    print(step_3_text)
    return step_3_text

text = "epstein wasn't a good guy. in fact, he did a lot of really really bad things. For instance, check out the tunnels on his island"

# prompt_chain_workflow(text, "german", api_key)



def multi_attempt_agent(question, num_attempts, api_key):
    """
    Run the same question multiple times and use voting to select best answer.

    Args:
        question (str): The question to ask
        num_attempts (int): Number of times to run the query
        api_key (str): Anthropic API key

    Returns:
        str: The most common answer across all attempts
    """
    responses = []
    for attempt in range(num_attempts):
        response = claude_api_call(question, api_key, return_type='json', system_prompt = None, 
                        tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                        temperature = 0.7)
        response_text = (response['content'][0]['text'])[:50]
        responses.append(response_text)
    final_message = "return the best answer using voting from these prior responses" + str(responses)
    final_response = claude_api_call(final_message, api_key, return_type='json', system_prompt = None, 
                        tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                        temperature = 0.7)
    final =  final_response['content'][0]['text']
    print(final)
    return final

# question = "was jeffery epstein a hero or a villan?"
# num_attempts = 5
# multi_attempt_agent(question, num_attempts, api_key)


def dispatch_to_worker(task, available_workers):
    """
    Dispatch a task to an appropriate worker (orchestrator-worker pattern).

    Args:
        task (dict): Task with "type" and "description" keys
        available_workers (dict): Workers with capabilities, availability, and load

    Returns:
        dict: Dictionary with "worker" and "status" keys
    """
    # TODO: Implement orchestrator logic
    task_type = task["type"]
    result = None
    lowest_load = 100000
    for worker in available_workers:
        worker_info = available_workers[worker]
        worker_capabilities = worker_info["capabilities"]
        worker_availability = worker_info["available"]
        worker_load = worker_info["load"]
        if task_type in worker_capabilities and worker_availability and worker_load < lowest_load:
            result = worker
            lowest_load = worker_load
    if result == None: return {'worker': None, 'status': 'no_worker_available'}
    return  {"worker": result, "status": "dispatched"}


task = {"type": "translation", "description": "Translate text"}
workers = {
    "worker1": {"capabilities": ["translation"], "available": True, "load": 2},
    "worker2": {"capabilities": ["translation"], "available": True, "load": 1}
}

print(dispatch_to_worker(task, workers))