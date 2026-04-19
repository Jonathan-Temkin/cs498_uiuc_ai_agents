import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *
import math
import re
import ast
import json



def tag_untrusted_content(content: str, tag: str = "untrusted") -> str:
    """
    Wrap untrusted external content with XML-style delimiter tags.

    Args:
        content (str): The untrusted content to wrap.
        tag (str): The XML tag name to use. Defaults to "untrusted".

    Returns:
        str: The content wrapped as "<tag>\n{content}\n</tag>".

    Raises:
        ValueError: If content is None.
        ValueError: If tag is an empty string or contains spaces.
    """
    # TODO: Raise ValueError if content is None

    # TODO: Raise ValueError if tag is empty or contains spaces

    # TODO: Return the content wrapped in opening and closing tags,
    # with each tag and the content on its own line.
    # Format: f"<{tag}>\n{content}\n</{tag}>"

    if content is None:
        raise ValueError
    if not tag or " " in tag:
        raise ValueError
    return f"<{tag}>\n{content}\n</{tag}>"


def detect_injection_in_retrieved_docs(documents: list) -> list:
    """
    Scan a list of retrieved documents for prompt injection patterns.

    Args:
        documents (list[str]): List of retrieved document strings.

    Returns:
        list[int]: Sorted list of 0-based indices of documents containing injection patterns.

    Raises:
        ValueError: If documents is None.
        ValueError: If any element in documents is not a string.

    Injection patterns (case-insensitive substring match):
        "ignore previous instructions", "ignore all previous",
        "disregard your", "new instructions:", "you are now",
        "your new task", "forget everything", "override your"
    """
    # TODO: Raise ValueError if documents is None

    # TODO: Raise ValueError if any element is not a string

    # TODO: Define the list of injection patterns

    # TODO: For each document, check if it contains any pattern (case-insensitive)
    # Collect and return sorted indices of suspicious documents
    results = []
    if documents is None:
        raise ValueError
    for doc in documents:
        if type(doc) != str:
            raise ValueError
    phrases = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard your",
    "new instructions:",
    "you are now",
    "your new task",
    "forget everything",
    "override your"
    ]
    for i, doc in enumerate(documents):
        phrase_in_doc = any(phrase.lower() in doc.lower() for phrase in phrases)
        if phrase_in_doc:
            results.append(i)
    return results


def enforce_least_privilege(requested_permissions: list, allowed_permissions: list) -> dict:
    """
    Enforce the principle of least privilege.

    Args:
        requested_permissions (list[str]): Permissions the agent is requesting.
        allowed_permissions (list[str]): Permissions allowed for this agent role.

    Returns:
        dict with keys:
            "granted" (list[str]): Sorted list of granted permissions (requested AND allowed).
            "denied"  (list[str]): Sorted list of denied permissions (requested but NOT allowed).

    Raises:
        ValueError: If either argument is None.
    """
    # TODO: Raise ValueError if either argument is None

    # TODO: Treat both lists as sets to deduplicate

    # TODO: Compute granted = intersection of requested and allowed

    # TODO: Compute denied = requested - allowed

    # TODO: Return sorted lists in the required dict
    
    if requested_permissions is None or allowed_permissions is None:
        raise ValueError
    requested_permissions = set(requested_permissions)
    allowed_permissions = set(allowed_permissions)
    granted = [request for request in requested_permissions if request in allowed_permissions]
    denied = [request for request in requested_permissions if request not in allowed_permissions]
    return  {"granted": sorted(granted), "denied": sorted(denied)}


def llm_classify_action_risk(action_name: str, args: dict, context: str, api_key: str) -> dict:
    """
    Use Claude to classify an agent action as low, medium, or high risk.

    Args:
        action_name (str): The name of the action/tool being called.
        args (dict): The arguments passed to the action.
        context (str): A description of the agent's purpose.
        api_key (str): Anthropic API key.

    Returns:
        dict with keys:
            "risk_level" (str): "low", "medium", or "high"
            "reasoning"  (str): Claude's full response text.

    Raises:
        Exception: If the API call returns a non-200 status.
    """
    # TODO: Build a prompt that asks Claude to classify the action risk.
    # Include: action_name, JSON-serialized args, and context.
    # Define the three risk levels explicitly in the prompt:
    #   high   = irreversible/destructive/sends data externally (delete, wipe, drop, transfer, execute)
    #   low    = read-only with no side effects (search, read, list, get, query)
    #   medium = everything else (write, update, create)
    # Tell Claude to classify based on the ACTION'S INHERENT DANGER, not the agent's purpose.
    # Instruct Claude that the LAST line of its response must be exactly one of:
    #   Risk: low
    #   Risk: medium
    #   Risk: high
    # with nothing after it.

    # TODO: POST to https://api.anthropic.com/v1/messages using requests.
    # Model: claude-haiku-4-5-20251001, max_tokens=256, temperature=0.
    # Headers: x-api-key, anthropic-version: 2023-06-01, content-type: application/json.

    # TODO: Raise an Exception if status != 200.

    # TODO: Extract response text from response.json()["content"][0]["text"].

    # TODO: Parse the risk level. Use re.findall() to find ALL occurrences of
    # "risk: low/medium/high" (case-insensitive) and take the LAST match.
    # This handles cases where Claude mentions risk mid-reasoning.
    # Default to "medium" if no match found.

    # TODO: Return {"risk_level": ..., "reasoning": ...}
    messages = f"CLASSIFY RISK: <ACTION> {action_name}</ACTION> <ARGS>{args}<ARGS> <CONTEXT> {context}<CONTEXT>. YOU WILL RETURN REASONING + EXACTLY one of low, medium, or high with no text after"
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    response = response["content"][0]["text"]
    matches = re.findall(r"(low|medium|high)", response, re.IGNORECASE)
    risk_level = matches[-1].lower() if matches else "medium"
    result = {"risk_level": risk_level, "reasoning": response}
    return result 