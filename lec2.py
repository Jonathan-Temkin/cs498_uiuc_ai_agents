import requests
from key import api_key
from vars import *


def create_message_object(role, content):
    """
    Create a properly formatted message object for the Anthropic API.

    Args:
        role (str): The role ("user" or "assistant")
        content (str): The message content

    Returns:
        dict: A message object with 'role' and 'content' keys
    """
    return {
        
        "role": role,
        "content" : content

    }


def call_with_prefill(user_message, prefill, api_key):
    """
    Make an API call to Claude using the assistant prefill technique.

    Args:
        user_message (str): The user's message
        prefill (str): The assistant prefill text (Claude will continue from here)
        api_key (str): Your Anthropic API key

    Returns:
        str: The complete response (prefill + Claude's continuation)
    """
    # TODO: Implement the function to:
    # 1. Make a POST request to https://api.anthropic.com/v1/messages
    # 2. Include messages array with user message AND assistant prefill
    # 3. Return prefill + the continuation from Claude
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
        #"messages": conversation_history
        "temperature": 0, # smaller # is more deterministic
        "messages": [ {"role":"user", "content": user_message},
                     {"role":"assistant", "content": prefill} ]
    }
    response = requests.post(url, json=body, headers=headers)
    msg = response.json()["content"][0]["text"]
    msg_prefilled = prefill + msg
    print(msg_prefilled)
    return msg_prefilled

user_message = "tell me about each of the 4 harry potter houses"
prefill = "1."

def validate_messages(messages):
    """
    Validate that messages follow the role alternation rules.

    Args:
        messages (list): A list of message dictionaries with 'role' and 'content'

    Returns:
        bool: True if valid, False otherwise
    """
    # TODO: Implement validation logic
    # Check: messages not empty, first is "user", roles alternate
    try:
         messages[0]["role"]
    except:
         return False
    if messages[0]["role"] != "user" or len(messages) < 1:
            return False
    prior_role = ''
    for message in messages:
        current_role = message["role"]
        if current_role not in ("user", "assistant"):
            return False
        if prior_role == current_role:
            return False
        prior_role = current_role
    return True


def get_last_user_message(messages):
    """
    Find and return the content of the last user message in a conversation.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'

    Returns:
        str or None: The content of the last user message, or None if no user messages exist
    """
    num_messages = len(messages)
    for i in range(num_messages -1, -1, -1):
        role = messages[i]["role"]
        if role == "user":
            return messages[i]["content"]
    return None


def get_last_assistant_message(messages):
    """
    Find and return the content of the last user message in a conversation.

    Args:
        messages (list): List of message dictionaries with 'role' and 'content'

    Returns:
        str or None: The content of the last user message, or None if no user messages exist
    """
    num_messages = len(messages)
    for i in range(num_messages -1, -1, -1):
        role = messages[i]["role"]
        if role == "assistant":
            return messages[i]["content"]
    return None
    
def merge_consecutive_messages(messages):
    """
    Merge consecutive messages with the same role into a single message.

    Args:
        messages (list): A list of message dictionaries with 'role' and 'content'

    Returns:
        list: Messages array with consecutive same-role messages merged
    """
    prior_role = ''
    prior_content = ''
    result = []
    for i in range(len(messages)):
        message = messages[i]
        current_role = message["role"]
        current_content = message["content"]
        if current_role not in ("user", "assistant"):
            return False
        if prior_role == current_role:
            result[i-1]["content"] = prior_content + "\n\n" + current_content
        else:
            result.append(message)
        prior_role = current_role
        prior_content = current_content
    return result



#print(merge_consecutive_messages(conversation_history))


def add_context_and_call(history, new_question, api_key):
    """
    Add a new question to conversation history and call the API.

    Args:
        history (list): Previous conversation messages
        new_question (str): New question to ask
        api_key (str): Your Anthropic API key

    Returns:
        str: Claude's response text
    """
    
    new_q =  {"role": "user", "content": new_question}
    history.append(new_q)
    print(history)
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
        #"messages": conversation_history
        "temperature": 0, # smaller # is more deterministic
        "messages": history
    }
    response = requests.post(url, json=body, headers=headers)
    print(response.json())
    msg = response.json()["content"][0]["text"]
    print(msg)
    return msg


# new_question = "eh whats up doc"
# new_q =  {"role": "user", "content": new_question}
# # print(conversation_history.append(new_q))
# add_context_and_call(conversation_history, new_question, api_key)


def build_conversation_array(user_messages, assistant_responses):
    """
    Build a messages array from separate user and assistant lists.

    Args:
        user_messages (list): List of user messages
        assistant_responses (list): List of assistant responses

    Returns:
        list: Properly formatted messages array
    """
    messages = []
    for i in range(len(user_messages)):
        user_message = user_messages[i]
        assistant_message = assistant_responses[i]
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    return messages

# user_messages = ["Hi", "How are you?"]
# assistant_responses = ["Hello!", "I'm great!"]
# print(build_conversation_array(user_messages, assistant_responses))


def call_with_xml_format(user_message, api_key):
    """
    Call Claude with a system prompt requiring XML format responses.

    Args:
        user_message (str): The user's message
        api_key (str): Your Anthropic API key

    Returns:
        str: Claude's response (should be valid XML with <response> root element)
    """

    """
    Make an API call to Claude using the assistant prefill technique.

    Args:
        user_message (str): The user's message
        prefill (str): The assistant prefill text (Claude will continue from here)
        api_key (str): Your Anthropic API key

    Returns:
        str: The complete response (prefill + Claude's continuation)
    """
    # TODO: Implement the function to:
    # 1. Make a POST request to https://api.anthropic.com/v1/messages
    # 2. Include messages array with user message AND assistant prefill
    # 3. Return prefill + the continuation from Claude
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": """return response in XMLThe response should be wrapped in a <response> root element.responses must be parseable XML""",
        #"messages": conversation_history
        "temperature": 0, # smaller # is more deterministic
        "messages": [ {"role":"user", "content": user_message} ]
    }
    response = requests.post(url, json=body, headers=headers)
    msg = response.json()["content"][0]["text"]
    print(msg)
    return msg

# call_with_xml_format(user_message, api_key)



def compare_max_tokens(user_message, api_key):
    """
    Compare Claude's responses with different max_tokens values.

    Args:
        user_message (str): The user's message
        api_key (str): Your Anthropic API key

    Returns:
        dict: {
            "short": {"response": str, "stop_reason": str},
            "long": {"response": str, "stop_reason": str}
        }
    """
    # TODO: Make two API calls with max_tokens=50 and max_tokens=500
    # Return both responses with their stop_reasons
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    short_body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": """return response in XMLThe response should be wrapped in a <response> root element.responses must be parseable XML""",
        #"messages": conversation_history
        "temperature": 0, # smaller # is more deterministic
        "messages": [ {"role":"user", "content": user_message} ],
        "max_tokens": 50
    }
    long_body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": """return response in XMLThe response should be wrapped in a <response> root element.responses must be parseable XML""",
        #"messages": conversation_history
        "temperature": 0, # smaller # is more deterministic
        "messages": [ {"role":"user", "content": user_message} ],
        "max_tokens": 500
    }

    response_long = requests.post(url, json=long_body, headers=headers)
    response_short = requests.post(url, json=short_body, headers=headers)
    #print(response_short.json())
    result = {
            "short": {"response": response_short.json()["content"][0]["text"], "stop_reason": response_short.json()["stop_reason"]},
            "long": {"response": response_long.json()["content"][0]["text"], "stop_reason": response_short.json()["stop_reason"]}
        }
    print(result)
    return result

compare_max_tokens(user_message, api_key)