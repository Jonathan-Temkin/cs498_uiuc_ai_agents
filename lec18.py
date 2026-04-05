import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *
import math
import re
import ast
import json


def planner_node(state: dict, api_key: str) -> dict:
    """
    LangGraph planner node: calls Claude to generate a step-by-step plan
    and returns ONLY the state update dict {"steps": [...]}.

    Args:
        state (dict): The current agent state. Must contain "question" (str).
                      May optionally contain "steps" (list[str]).
        api_key (str): Anthropic API key. Must not be None.

    Returns:
        dict: Update dict containing only {"steps": [...existing steps..., <llm_plan_text>]}.
              The original state must NOT be mutated.

    Raises:
        ValueError: If state is None.
        ValueError: If "question" key is missing from state.
        ValueError: If api_key is None.

    Example:
        state = {"question": "How do I implement rate limiting in FastAPI?",
                 "steps": ["initialized", "context_loaded"]}
        result = planner_node(state, api_key="sk-ant-...")
        # result -> {"steps": ["initialized", "context_loaded", "<plan from Claude>"]}
        # state is NOT mutated
    """
    # TODO: Raise ValueError if state is None.

    # TODO: Raise ValueError if "question" is not in state.

    # TODO: Raise ValueError if api_key is None.

    # TODO: Build a system prompt instructing Claude to act as a planning agent
    # and produce a concise step-by-step implementation plan.

    # TODO: Build the user message content from state["question"].

    # TODO: Call requests.post("https://api.anthropic.com/v1/messages", headers=..., json=...)
    # Headers: x-api-key, anthropic-version="2023-06-01", content-type="application/json"
    # Body: model="claude-haiku-4-5-20251001", max_tokens=1024, system=...,
    #       messages=[{role: "user", content: user_content}]

    # TODO: Extract the plan text: response.json()["content"][0]["text"]

    # TODO: Return {"steps": list(state.get("steps", [])) + [llm_plan]}
    # Do NOT mutate the original state["steps"] list.
    if state is None:
        raise ValueError("state is None")
    if "question" not in state:
        raise ValueError("question key is missing from state")
    if api_key is None:
        raise ValueError("api_key is None")
    messages = state["question"]
    system_prompt = "as a planning agent and produce a concise step-by-step implementation plan."
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    response_text = ["content"][0]["text"]
    return  {"steps": list(state.get("steps", [])) + [response_text]}


def route_after_llm(state: dict) -> str:
    """
    LangGraph conditional edge function called after an LLM node.

    Args:
        state (dict): The current agent state. May contain "tool_needed" (bool).

    Returns:
        str: One of:
             - "tool_node"     if state["tool_needed"] is exactly True (bool)
             - "final_answer"  if state["tool_needed"] is exactly False (bool)
             - "error"         if "tool_needed" is missing or not a bool

    Raises:
        ValueError: If state is None.

    Examples:
        route_after_llm({"tool_needed": True})   # -> "tool_node"
        route_after_llm({"tool_needed": False})  # -> "final_answer"
        route_after_llm({"tool_needed": 1})      # -> "error"
        route_after_llm({})                      # -> "error"
        route_after_llm(None)                    # raises ValueError
    """
    # TODO: Raise ValueError if state is None.

    # TODO: If "tool_needed" is not in state, return "error".

    # TODO: Use isinstance(state["tool_needed"], bool) to check exact bool type.
    #       If not a bool, return "error".

    # TODO: If True, return "tool_node". If False, return "final_answer".
    if not state:
        raise ValueError
    tool_needed_val = state["tool_needed"]
    if "tool_needed" not in state.keys():
        return "error"
    if not(isinstance(state["tool_needed"], bool)):
        return "error"
    if state["tool_needed"]:
        return "tool_node"
    else:
        return "final_answer"
    



def run_llm_agent_loop(question: str, tool_fn: callable, api_key: str, max_steps: int = 5) -> dict:
    """
    LLM-driven agent loop with real tool execution.

    At each step Claude receives the FULL conversation history so far.
    - If Claude responds "TOOL_NEEDED": call tool_fn(question), append Claude's
      response as an assistant message and the tool result as a new user message,
      then continue to the next step.
    - If Claude responds "FINAL_ANSWER: <text>": extract the answer and stop.
    - If max_steps is reached without a final answer: stopped_by="max_steps_exceeded".

    Args:
        question  (str):      The question to answer. Must not be None or empty.
        tool_fn   (callable): A function that takes the question string and returns
                              a tool result string. Must not be None.
        api_key   (str):      Anthropic API key. Must not be None.
        max_steps (int):      Maximum LLM calls before stopping. Must be > 0.

    Returns:
        dict: {
            "answer":      str or None,   # extracted answer, None if max_steps exceeded
            "steps_taken": int,           # number of LLM calls made
            "stopped_by":  str,           # "final_answer" or "max_steps_exceeded"
            "history":     list[str]      # each raw LLM response in order
        }

    Raises:
        ValueError: If question is None or empty.
        ValueError: If tool_fn is None.
        ValueError: If api_key is None.
        ValueError: If max_steps <= 0.

    Example:
        def my_tool(q): return "The capital of France is Paris."
        result = run_llm_agent_loop("What is the capital of France?", my_tool, api_key="sk-ant-...")
        # Possible flow:
        #   Step 1: Claude responds "TOOL_NEEDED"
        #           -> tool_fn called -> "The capital of France is Paris."
        #           -> result added to conversation
        #   Step 2: Claude responds "FINAL_ANSWER: Paris"
        # result -> {"answer": "Paris", "steps_taken": 2,
        #            "stopped_by": "final_answer", "history": ["TOOL_NEEDED", "FINAL_ANSWER: Paris"]}
    """
    # TODO: Raise ValueError if question is None or empty.

    # TODO: Raise ValueError if tool_fn is None.

    # TODO: Raise ValueError if api_key is None.

    # TODO: Raise ValueError if max_steps <= 0.

    # TODO: Build a system_prompt instructing Claude to respond ONLY with:
    #   "TOOL_NEEDED" if it needs a tool result, or
    #   "FINAL_ANSWER: <answer>" if it can answer now.

    # TODO: Initialize the conversation:
    #   messages = [{"role": "user", "content": f"Question: {question}"}]
    #   history = [], step_count = 0, answer = None, stopped_by = "max_steps_exceeded"

    # TODO: Loop while step_count < max_steps:
    #   1. Call requests.post("https://api.anthropic.com/v1/messages", ...)
    #      Headers: x-api-key, anthropic-version="2023-06-01", content-type="application/json"
    #      Body:    model="claude-haiku-4-5-20251001", max_tokens=512,
    #               system=system_prompt, messages=messages  (the growing conversation)
    #   2. Extract: raw = response.json()["content"][0]["text"].strip()
    #   3. Append raw to history; increment step_count.
    #   4. If raw starts with "FINAL_ANSWER:":
    #         answer = raw after "FINAL_ANSWER:"; stopped_by = "final_answer"; break
    #   5. If raw starts with "TOOL_NEEDED":
    #         Append {"role": "assistant", "content": raw} to messages
    #         tool_result = tool_fn(question)
    #         Append {"role": "user", "content": f"Tool result: {tool_result}"} to messages
    #         (then continue the loop — Claude now has the tool result in context)
    #   6. Otherwise: treat as final answer; stopped_by = "final_answer"; break

    # TODO: Return {"answer": answer, "steps_taken": step_count,
    #               "stopped_by": stopped_by, "history": history}
    if not question:
        raise ValueError()
    if tool_fn is None:
        raise ValueError()
    if api_key is None:
        raise ValueError()
    if max_steps <= 0:
        raise ValueError()
    system_prompt = "respond ONLY with TOOL_NEEDED if it needs a tool result, or FINAL_ANSWER: <answer> if it can answer now"
    history = []
    step_count = 1
    answer = None
    stopped_by = "max_steps_exceeded"
    messages = [{"role": "user", "content": f"Question: {question}"}]
    while step_count < max_steps:
        response = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
        raw = response["content"][0]["text"].strip()
        if raw.startswith("FINAL_ANSWER"):
            answer = raw.split("FINAL_ANSWER:", 1)[1].strip()
            stopped_by = "final_answer"
            break
        elif raw.startswith("TOOL_NEEDED"):
            messages.append({"role": "assistant", "content": raw})
            tool_result = tool_fn(question)
            messages.append({"role": "user", "content": f"Tool result: {tool_result}"} )
        else:
            answer = raw
            stopped_by = "final_answer"
            break
    return {"answer": answer, "steps_taken": step_count,"stopped_by": stopped_by, "history": history}





def save_checkpoint(state: dict) -> str:
    """
    Serialize agent state to a JSON string for durable checkpointing.

    Args:
        state (dict): The agent state to serialize. All values must be JSON-serializable.

    Returns:
        str: A JSON string representation of state.

    Raises:
        ValueError: If state is None.

    Example:
        save_checkpoint({"step": 3, "done": False})
        # -> '{"step": 3, "done": false}'
    """
    # TODO: Raise ValueError if state is None.
    # TODO: Return json.dumps(state).
    if not state:
        raise ValueError
    return json.dumps(state)


def load_checkpoint(checkpoint_str: str) -> dict:
    """
    Deserialize a JSON checkpoint string back to a dict.

    Args:
        checkpoint_str (str): A JSON string previously produced by save_checkpoint.

    Returns:
        dict: The deserialized state dict.

    Raises:
        ValueError: If checkpoint_str is None or empty string.
        ValueError: If checkpoint_str is not valid JSON (with a helpful message).

    Example:
        load_checkpoint('{"step": 3, "done": false}')
        # -> {"step": 3, "done": False}
    """
    # TODO: Raise ValueError if checkpoint_str is None or empty ("").
    # TODO: Try json.loads(checkpoint_str).
    #       Catch json.JSONDecodeError and raise ValueError with a helpful message.
    if not checkpoint_str or checkpoint_str == "":
        raise ValueError
    try:
        return json.loads(checkpoint_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def resume_workflow(checkpoint_str: str) -> dict:
    """
    Load a checkpoint and append "resumed" to the events list.

    Args:
        checkpoint_str (str): A JSON checkpoint string.

    Returns:
        dict: The loaded state with "resumed" appended to state["events"].
              Creates state["events"] = ["resumed"] if the key was absent.

    Example:
        resume_workflow('{"events": ["started"], "step": 1}')
        # -> {"events": ["started", "resumed"], "step": 1}
    """
    # TODO: Load state using load_checkpoint(checkpoint_str).
    # TODO: Append "resumed" to state["events"] (create the list if missing).
    # TODO: Return the updated state.
    state  = load_checkpoint(checkpoint_str).
    state["events"].append("resumed")
    return state


def call_with_retry(primary_fn, fallback_fn, max_retries: int):
    """
    Call primary_fn() with retry logic; fall back to fallback_fn() if all attempts fail.

    Args:
        primary_fn (callable): The primary function to call. Takes no arguments.
        fallback_fn (callable): The fallback function to call if all primary attempts fail.
                                Takes no arguments.
        max_retries (int): Number of additional retries after the first attempt.
                           Must be >= 0.
                           Total attempts = 1 + max_retries.

    Returns:
        any: The return value of primary_fn (if any attempt succeeds) or fallback_fn.

    Raises:
        ValueError: If primary_fn is None, fallback_fn is None, or max_retries < 0.
        Exception:  If fallback_fn() raises, that exception propagates.

    Examples:
        call_with_retry(lambda: 42, lambda: 0, 3)   # -> 42

        attempts = [0]
        def flaky():
            attempts[0] += 1
            if attempts[0] < 3:
                raise RuntimeError("fail")
            return "success"
        call_with_retry(flaky, lambda: "fallback", 5)  # -> "success" (3rd attempt)

        def always_fails():
            raise RuntimeError("always")
        call_with_retry(always_fails, lambda: "fallback", 2)  # -> "fallback"
    """
    # TODO: Raise ValueError if primary_fn is None, fallback_fn is None, or max_retries < 0.

    # TODO: Try primary_fn() up to (1 + max_retries) times.
    #       On success, return the result immediately.
    #       On any exception, continue to the next attempt.

    # TODO: If all attempts failed, call fallback_fn() and return its result.
    #       Let any exception from fallback_fn propagate.
    if primary_fn is None or fallback_fn is None or max_retries < 0:
        raise ValueError
    max_tries = 1 + max_retries 
    for i in range(max_retries+1):
        try:
            return primary_fn()
        except:
            pass
    return fallback_fn()



def human_review(action: str, reviewer_decision: str, edited_action: str = None):
    """
    Apply a human reviewer's decision to a proposed agent action.

    Args:
        action (str): The proposed action string (must be non-empty).
        reviewer_decision (str): One of "approve", "edit", or "reject".
                                 Comparison is case-sensitive.
        edited_action (str, optional): Replacement action used when decision is "edit".
                                       May be any string including empty; must not be None.

    Returns:
        str or None:
            - "approve": return the original action unchanged.
            - "edit":    return edited_action.
            - "reject":  return None.

    Raises:
        ValueError: If action is None or empty string.
        ValueError: If reviewer_decision is not "approve", "edit", or "reject".
        ValueError: If reviewer_decision is "edit" and edited_action is None.

    Examples:
        human_review("delete_file('important.py')", "approve")
        # -> "delete_file('important.py')"

        human_review("delete_file('important.py')", "edit", "delete_file('temp.py')")
        # -> "delete_file('temp.py')"

        human_review("delete_file('important.py')", "reject")
        # -> None

        human_review("push_to_prod()", "edit", None)  # raises ValueError
        human_review("push_to_prod()", "APPROVE")      # raises ValueError (case-sensitive)
        human_review("", "approve")                   # raises ValueError
    """
    # TODO: Raise ValueError if action is None or empty string.

    # TODO: Raise ValueError if reviewer_decision is not one of
    # "approve", "edit", or "reject" (exact match, case-sensitive).

    # TODO: If reviewer_decision == "approve", return action unchanged.

    # TODO: If reviewer_decision == "edit", raise ValueError if edited_action is None,
    # otherwise return edited_action.

    # TODO: If reviewer_decision == "reject", return None.

    if not action or reviewer_decision not in ["approve", "edit", "reject" ]:
        raise ValueError
    if reviewer_decision == "approve":
        return action
    elif reviewer_decision == "edit":
        if edited_action == None:
            raise ValueError
        return edited_action
    elif  reviewer_decision == "reject":
        return None


def assign_tasks(agents: list, tasks: list) -> dict:
    """
    Assign each task to the first agent who has the required skill.

    Args:
        agents (list): List of agent dicts, each with:
                       "name" (str): agent identifier.
                       "skills" (list[str]): skills this agent possesses.
        tasks (list): List of task dicts, each with:
                      "id" (any): unique task identifier.
                      "required_skill" (str): skill needed to handle this task.

    Returns:
        dict: Maps task["id"] -> agent["name"] for matched tasks.
              Maps task["id"] -> "unassigned" for tasks with no matching agent.
              Returns {} if tasks is empty.

    Raises:
        ValueError: If agents is None.
        ValueError: If tasks is None.

    Examples:
        agents = [
            {"name": "Alice", "skills": ["search", "write"]},
            {"name": "Bob",   "skills": ["code", "test"]},
            {"name": "Carol", "skills": ["search", "code"]},
        ]
        tasks = [
            {"id": 1,    "required_skill": "write"},
            {"id": 2,    "required_skill": "code"},
            {"id": 3,    "required_skill": "design"},
            {"id": "t4", "required_skill": "search"},
        ]
        assign_tasks(agents, tasks)
        # -> {1: "Alice", 2: "Bob", 3: "unassigned", "t4": "Alice"}
    """
    # TODO: Validate inputs — raise ValueError for None agents or None tasks.

    # TODO: For each task, iterate through agents in order and find the FIRST
    # agent whose "skills" list contains the task's "required_skill" (case-sensitive).
    # If found, map task["id"] -> agent["name"].
    # If no agent matches, map task["id"] -> "unassigned".

    # TODO: Return the result dict. Return {} if tasks is empty.
    results = {}
    if tasks == []: return {}
    if not agents or not tasks:
        raise ValueError
    for task in tasks:
        id = task["id"]
        required_skill = task['required_skill']
        for agent in agents:
            name = agent["name"]
            skills = agent['skills']
            if required_skill in skills:
                results[id] = name
                break
            if id not in results.keys():
                results[id] = "unassigned"
    return results


def compact_messages(messages: list, max_messages: int) -> list:
    """
    Compact a message list using a summary-insertion strategy.

    Args:
        messages (list): List of message dicts (e.g., each has "role" and "content").
        max_messages (int): Maximum number of messages allowed in the output.
                            Must be >= 2 (need at least first message + summary).

    Returns:
        list: If len(messages) <= max_messages, a copy of the original list unchanged.
              Otherwise a new list of exactly max_messages items:
                [messages[0],
                 {"role": "system", "content": "SUMMARY: N messages omitted"},
                 ...last (max_messages - 2) messages...]
              where N = len(messages) - max_messages + 1.

    Raises:
        ValueError: If messages is None.
        ValueError: If max_messages < 2.

    Examples:
        msgs = [
            {"role": "system",    "content": "You are helpful."},
            {"role": "user",      "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user",      "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
        ]
        compact_messages(msgs, max_messages=4)
        # omitted_count = 5 - 4 + 1 = 2
        # -> [system("You are helpful."),
        #     system("SUMMARY: 2 messages omitted"),
        #     user("msg3"),
        #     assistant("msg4")]

        compact_messages(msgs, max_messages=5)  # no compaction
        # -> copy of all 5 messages
    """
    # TODO: Validate inputs — raise ValueError for None messages or max_messages < 2.

    # TODO: If len(messages) <= max_messages, return a copy (list(messages)) unchanged.

    # TODO: Compute omitted_count = len(messages) - max_messages + 1.
    # Build the summary message: {"role": "system", "content": f"SUMMARY: {omitted_count} messages omitted"}.
    # Keep messages[0] (first message) and the last (max_messages - 2) messages.
    # Return [messages[0], summary] + last_(max_messages-2)_messages.

    if not messages or max_messages < 2:
        raise ValueError
    if len(messages) <= max_messages:
        return copy(list(messages))
    omitted_count = len(messages) - max_messages + 1
    summary_message = {"role": "system", "content": f"SUMMARY: {omitted_count} messages omitted"}
    return [messages[0], summary_message] + messages[-(max_messages - 2):]


import json
import re
import requests


def llm_guardrail(report: dict, api_key: str) -> dict:
    """
    LLM-based guardrail: calls Claude to validate a structured report and
    returns the parsed JSON result {"valid": bool, "reason": str}.

    Args:
        report (dict): The report to validate. Expected to contain:
                       "title" (str), "summary" (str), "risk_score" (int).
                       Must not be None.
        api_key (str): Anthropic API key. Must not be None.

    Returns:
        dict: {"valid": bool, "reason": str} parsed from Claude's JSON response.

    Raises:
        ValueError: If report is None.
        ValueError: If api_key is None.
        ValueError("Invalid guardrail response: not valid JSON"): If Claude's
            response cannot be parsed as JSON.

    Example:
        report = {"title": "Q3 Security Audit",
                  "summary": "All critical vulnerabilities addressed.",
                  "risk_score": 3}
        result = llm_guardrail(report, api_key="sk-ant-...")
        # result -> {"valid": True, "reason": "Report is well-formed..."}
    """
    # TODO: Raise ValueError if report is None.

    # TODO: Raise ValueError if api_key is None.

    # TODO: Build a system prompt instructing Claude to act as a guardrail validator.
    # It should check: title is non-empty, summary is meaningful, risk_score is 0-10.
    # Tell it to respond with ONLY valid JSON: {"valid": true/false, "reason": "..."}

    # TODO: Build user_content as a readable string of all report fields,
    # including the title and risk_score.

    # TODO: Call requests.post("https://api.anthropic.com/v1/messages", headers=..., json=...)
    # Headers: x-api-key, anthropic-version="2023-06-01", content-type="application/json"
    # Body: model="claude-haiku-4-5-20251001", max_tokens=256, system=...,
    #       messages=[{role: "user", content: user_content}]

    # TODO: Extract raw_text = response.json()["content"][0]["text"].strip()

    # TODO: Strip markdown code fences if Claude wrapped the JSON (Claude sometimes responds
    # with ```json ... ``` even when told not to). Use re.sub to remove leading ```json\n
    # and trailing ``` before parsing.

    # TODO: Try to parse the cleaned text as JSON with json.loads().
    # If parsing fails, raise ValueError("Invalid guardrail response: not valid JSON")

    # TODO: Return the parsed dict.
    if report is None:
        raise ValueError("report is None")
    if api_key is None:
        raise ValueError("api_key is None")
    system_prompt = "act as a guardrail validator.It should check: title is non-empty, summary is meaningful, risk_score is 0-10.respond with ONLY valid JSON: {valid: true/false, reason: ..}"
    user_content = f"Title: {report.get('title')}\n" + f"Summary: {report.get('summary')}\n" + f"Risk Score: {report.get('risk_score')}"
    response  =claude_api_call(user_content, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    response_text = response["content"][0]["text"].strip()
    cleaned_text = re.sub(r'^```json\s*|```$', '', response_text, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        raise ValueError("Invalid guardrail response: not valid JSON")
    


def build_event(event_type: str, payload) -> dict:
    """
    Construct a typed event dict for use in an Autogen-style event-driven system.

    Args:
        event_type (str): The event type identifier (must be non-empty).
        payload: The data associated with this event (any value, including None).

    Returns:
        dict: {"type": event_type, "payload": payload}

    Raises:
        ValueError: If event_type is None or empty string.

    Examples:
        build_event("code_review", "def foo(): pass")
        # -> {"type": "code_review", "payload": "def foo(): pass"}

        build_event("", "x")   # raises ValueError
        build_event(None, "x") # raises ValueError
    """
    # TODO: Raise ValueError if event_type is None or empty string.

    # TODO: Return {"type": event_type, "payload": payload}.

    if event_type is None or event_type == "":
        return ValueError
    return {"type": event_type, "payload": payload}


def dispatch_event(event: dict, handlers: dict):
    """
    Dispatch a typed event to its registered handler.

    Args:
        event (dict): Must have a "type" key. May also have a "payload" key.
        handlers (dict): Maps event type strings to callables.
                         Each callable takes one argument (the payload).

    Returns:
        any: The return value of the matched handler,
             "unhandled" if event["type"] is not in handlers,
             or {"error": str(exception), "event_type": event["type"]} if the handler raises.

    Raises:
        ValueError: If event is None.
        ValueError: If handlers is None.
        ValueError: If event does not have a "type" key.

    Examples:
        handlers = {
            "code_review": lambda p: f"Reviewing: {p}",
        }
        dispatch_event({"type": "code_review", "payload": "foo"}, handlers)
        # -> "Reviewing: foo"

        dispatch_event({"type": "deploy", "payload": "v1.0"}, handlers)
        # -> "unhandled"

        def bad(p): raise RuntimeError("connection failed")
        dispatch_event({"type": "code_review", "payload": "x"}, {"code_review": bad})
        # -> {"error": "connection failed", "event_type": "code_review"}
    """
    # TODO: Raise ValueError if event is None, handlers is None,
    # or event does not have a "type" key.

    # TODO: If event["type"] is NOT in handlers, return "unhandled".

    # TODO: Call handlers[event["type"]](event.get("payload")) and return the result.
    # If the handler raises any exception, catch it and return
    # {"error": str(exception), "event_type": event["type"]}.

    if not event or not handlers or "type" not in event.keys():
        raise ValueError
    if  event["type"] not in handlers:
        return "unhandled"
    try:
        return handlers[event["type"]](event.get("payload"))
    except Exception as e:
        {"error": str(e), "event_type": event["type"]}