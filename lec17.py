import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *
import math
import re
import ast

def get_required_context(target: str, graph: dict) -> list:
    """
    Return a sorted list of all modules that `target` transitively depends on,
    excluding `target` itself.

    Args:
        target (str): The module whose dependencies you want to collect.
        graph (dict): Maps each module name to a list of its direct dependencies.
                      Modules that appear as dependencies but have no key in graph
                      are treated as leaf nodes (no further dependencies).

    Returns:
        list: Sorted list of all transitive dependency module names.
              Does not include `target` itself. Returns [] if no dependencies.

    Raises:
        ValueError: If target is None or empty.
        ValueError: If graph is None.
        ValueError: If target is not a key in graph.

    Examples:
        graph = {
            "api":    ["auth", "utils"],
            "auth":   ["db", "utils"],
            "db":     ["config"],
            "utils":  ["config"],
            "config": [],
        }
        get_required_context("api", graph)   # -> ["auth", "config", "db", "utils"]
        get_required_context("config", graph) # -> []
    """
    # TODO: Validate inputs — raise ValueError for None/empty target, None graph,
    # and target not in graph.

    # TODO: Use BFS or DFS to collect all transitive dependencies.
    # Start from graph[target] and keep expanding. Track visited nodes to
    # handle cycles. Do not include target itself in the result.
    # Nodes missing from graph keys are treated as leaves — add them to visited
    # but do not try to expand them.

    # TODO: Return sorted(visited)
    result = []
    if not target or not graph or target not in graph.keys():
        raise ValueError
    visited = []
    while target is not None:
        currents = []
        for element in graph[target]:
            if element not in visited and element != target and element not in result:
                result.append(element)
            currents.append(element)
        visited.append(target)
        target = next((current for current in currents if current not in visited and current in graph.keys()), None)
    return sorted(result)

def parse_react_step(text: str) -> dict:
    """
    Parse a ReAct-formatted text into a structured dictionary.

    Args:
        text: A string that may contain Thought:, Action:, and/or Observation: sections.

    Returns:
        A dict with keys "thought", "action", "observation".
        Missing sections are set to None. Content is stripped of whitespace.

    Raises:
        ValueError: if text is None or empty, or if no ReAct sections are found.
    """
    # TODO: Raise ValueError if text is None or empty string

    # TODO: Use re.split or re.findall/re.search to extract sections.
    # Hint: Use a pattern like r'(?i)(thought|action|observation):\s*' to split.
    # Content for each section runs until the next section header or end of string.

    # TODO: Build the result dict with keys "thought", "action", "observation"
    # Set to None for any section not found in the text.

    # TODO: Raise ValueError if none of the three sections are found.

    # TODO: Return the dict with stripped content values.
    
    if not text:
        raise ValueError
    thought = re.search(r'(?i)thought:\s*(.*?)(?=\s*(?:thought|action|observation):|$)', text, re.DOTALL)
    action = re.search(r'(?i)action:\s*(.*?)(?=\s*(?:thought|action|observation):|$)', text, re.DOTALL)
    observation = re.search(r'(?i)observation:\s*(.*?)(?=\s*(?:thought|action|observation):|$)', text, re.DOTALL)
    thought = thought.group(1).strip() if thought else None
    action = action.group(1).strip() if action else None
    observation = observation.group(1).strip() if observation else None


    if thought is None and action is None and observation is None:
        raise ValueError("No sections found")
    result = {"thought":thought,
    "action": action,
    "observation": observation}
    return result


def call_architect(task: str, codebase_context: str, api_key: str) -> str:
    """
    Call Claude Opus as the Architect to produce a step-by-step implementation plan.

    Args:
        task (str): Natural language description of the coding task. Must not be None/empty.
        codebase_context (str): Relevant codebase summary (e.g., file names and functions).
                                 Must not be None/empty.
        api_key (str): Anthropic API key. Must not be None.

    Returns:
        str: The implementation plan text returned by Claude.

    Raises:
        ValueError: If task is None/empty, codebase_context is None/empty, or api_key is None.

    Example:
        plan = call_architect(
            task="Add rate limiting to /login",
            codebase_context="File: auth.py  Functions: login(), logout()",
            api_key="sk-ant-..."
        )
        # plan is a numbered step-by-step plan string from Claude
    """
    # TODO: Validate task, codebase_context, api_key — raise ValueError as needed.

    # TODO: Build a system prompt instructing Claude to act as a software architect
    # and produce a step-by-step implementation plan.

    # TODO: Build the user message content combining task and codebase_context.

    # TODO: Call requests.post("https://api.anthropic.com/v1/messages", headers=..., json=...)
    # Headers: x-api-key, anthropic-version="2023-06-01", content-type="application/json"
    # Body: model="claude-opus-4-6", max_tokens=1024, system=..., messages=[{role:user, content:...}]

    # TODO: Return response.json()["content"][0]["text"]

    if not task:
        raise ValueError("task is None or empty.")
    if not codebase_context:
        raise ValueError("codebase_context is None or empty.")
    if api_key is None:
        raise ValueError("api_key is None.")
    
    messages = f"<task>{task}</task> <codebase>{codebase_context}</codebase>"
    system_prompt = "act as a software architect and produce a step-by-step implementation plan."
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    return response['content'][0]['text']


def is_safe_command(command: str) -> bool:
    """
    Check whether a shell command is safe to execute.

    Args:
        command: A shell command string to inspect.

    Returns:
        True  if the command is considered safe.
        False if the command contains a known-dangerous pattern.

    Raises:
        ValueError: if command is None.

    Unsafe patterns (substring match, case-insensitive):
        "rm -rf", "rm -fr", "curl ", "wget ", "sudo ",
        "chmod 777", "chmod -R", "dd if=", "mkfs",
        ":|:", "> /dev/", "shred"
    """
    # TODO: Raise ValueError if command is None

    # TODO: Return True for empty string

    # TODO: Check for each unsafe pattern (case-insensitive substring match)
    # Hint: command.lower() and then check "rm -rf" in lowered, etc.

    # TODO: Return True if no unsafe pattern is found
    if command == None:
        raise ValueError
    if command == "":
        return True
    command_lower = command.lower()
    unsafe_patterns = [
        "rm -rf", "rm -fr", 
        "curl ", "wget ", "sudo ", 
        "chmod 777", "chmod-R", 
        "dd if=", "mkfs", 
        ":|:", "> /dev/", "shred"
    ]
    for pattern in unsafe_patterns:
        if pattern in command_lower:
            return False
    return True



def rank_code_snippets(query: str, snippets: list) -> list:
    """
    Rank code snippets by token overlap with a query.

    Args:
        query:    A natural language or code search query string.
        snippets: A list of dicts, each with "name" (str) and "code" (str).

    Returns:
        A new list of dicts (original fields + "score" key), sorted by:
          1. score descending
          2. snippet["name"] ascending (for ties)

    Raises:
        ValueError: if query is None or empty, or snippets is None.
    """
    # TODO: Raise ValueError if query is None or empty
    # TODO: Raise ValueError if snippets is None

    # TODO: Return [] if snippets is empty list

    # TODO: Tokenize query using re.split(r'[\W_]+', query.lower()), discard empty strings
    # Using [\W_]+ splits on non-word chars AND underscores, so snake_case identifiers
    # like "parse_json" become ["parse", "json"] instead of one token.
    # Hint: [t for t in re.split(r'[\W_]+', query.lower()) if t]

    # TODO: For each snippet, compute score:
    #   - tokenize snippet["code"] the same way (re.split(r'[\W_]+', ...))
    #   - score = number of unique query tokens that appear in the snippet token set
    # Hint: len(set(query_tokens) & set(snippet_tokens))

    # TODO: Add "score" key to each snippet copy and sort
    # Sort: score DESC, name ASC

    # TODO: Return the sorted list
    results = []
    if query is None or query == "":
            raise ValueError("Query cannot be None or empty.")
    if snippets is None:
        raise ValueError("Snippets list cannot be None.")
    if snippets == []:
        return []
    query_tokens =  [t for t in re.split(r'[\W_]+', query.lower()) if t]
    
    for snippet in snippets:
        snippet_copy = snippet.copy()
        snippet_tokens =  [t for t in re.split(r'[\W_]+', snippet_copy['code'].lower()) if t]
        score = len(set(query_tokens) & set(snippet_tokens))
        snippet_copy["score"] = score
        results.append(snippet_copy)
    return results.sort(key=lambda x: (-x["score"], x["name"]))


def summarize_context_for_handoff(messages: list, api_key: str) -> str:
    """
    Call Claude Haiku to summarize an agent conversation into a compact handoff note.

    Args:
        messages (list): List of dicts, each with "role" and "content" keys.
                         Must not be None or empty.
        api_key (str): Anthropic API key. Must not be None.

    Returns:
        str: The compact summary text returned by Claude.

    Raises:
        ValueError: If messages is None/empty or api_key is None.

    Example:
        msgs = [
            {"role": "assistant", "content": "Found bug in auth.py line 42."},
            {"role": "assistant", "content": "Fix: update token expiry check."},
        ]
        summary = summarize_context_for_handoff(msgs, "sk-ant-...")
        # summary is a concise handoff paragraph
    """
    # TODO: Validate messages (not None and not empty list) and api_key (not None).

    # TODO: Build a transcript string by joining each message as "{role}: {content}", one per line.

    # TODO: Build a system prompt instructing Claude to summarize agent conversation
    # into a compact handoff note.

    # TODO: Call requests.post("https://api.anthropic.com/v1/messages", headers=..., json=...)
    # Headers: x-api-key, anthropic-version="2023-06-01", content-type="application/json"
    # Body: model="claude-haiku-4-5-20251001", max_tokens=512, system=...,
    #       messages=[{role:user, content: "Summarize this agent conversation:\n\n{transcript}"}]

    if not messages: 
            raise ValueError("messages is None or empty.")
    if api_key is None:
        raise ValueError("api_key is None.")
    transcript = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    system_prompt = "Summarize this agent conversation:\n\n{transcript}"
    response = claude_api_call(transcript, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    return response['content'][0]['text']


def validate_code_syntax(code: str) -> dict:
    """
    Check whether a Python code string is syntactically valid using ast.parse().

    Args:
        code (str): Python source code to validate. Must not be None.

    Returns:
        dict with keys:
            "valid" (bool):  True if code parses without error.
            "error" (str|None): str(e) from the SyntaxError, or None if valid.
            "line"  (int|None): e.lineno from the SyntaxError, or None if valid.

    Raises:
        ValueError: if code is None.

    Examples:
        validate_code_syntax("def foo(x):\\n    return x\\n")
        # -> {"valid": True, "error": None, "line": None}

        validate_code_syntax("def foo(x)\\n    return x\\n")  # missing colon
        # -> {"valid": False, "error": "...", "line": 1}

        validate_code_syntax("")
        # -> {"valid": True, "error": None, "line": None}
    """
    # TODO: Raise ValueError if code is None

    # TODO: Try ast.parse(code).
    #   If it succeeds, return {"valid": True, "error": None, "line": None}
    #   If it raises SyntaxError e, return {"valid": False, "error": str(e), "line": e.lineno}
    if not code:
        raise ValueError
    try:
        ast.parse(code)
        return {"valid": True, "error": None, "line": None}
    except SyntaxError as e:
        return  {"valid": False, "error": str(e), "line": e.lineno}


def select_best_patch(patches: list) -> dict:
    """
    Select the best patch from a list of candidate patches.

    Each patch is a dict with:
        "id"           (str)  - unique identifier
        "tests_passed" (int)  - number of tests that passed
        "tests_total"  (int)  - total number of tests

    Selection priority:
        1. Patches where tests_passed == tests_total (all tests pass) are preferred.
        2. Among all-passing patches, prefer fewest tests_total (simpler fix).
        3. If no all-passing patch exists, prefer highest tests_passed.
        4. Ties broken by lowest tests_total, then by id lexicographically ascending.

    Args:
        patches: List of patch dicts.

    Returns:
        The single best patch dict.

    Raises:
        ValueError: if patches is None or empty.
        ValueError: if any patch is missing required keys.
        ValueError: if any patch has tests_passed > tests_total or negative values.
    """
    # TODO: Validate patches is not None and not empty

    # TODO: Validate each patch has keys "id", "tests_passed", "tests_total"

    # TODO: Validate no patch has tests_passed > tests_total or negative values

    # TODO: Filter patches where tests_passed == tests_total
    # If any exist, select the one with fewest tests_total (tie-break: id ascending)

    # TODO: If no all-passing patch, select by highest tests_passed
    # Tie-break: lowest tests_total, then id ascending

    # TODO: Return the best patch dict
    pass_rate = 0 
    all_passed = []
    for path in patches:
        try:
            id = path['id']
            tests_passed = path['tests_passed']
            tests_total = path["tests_total"]
            pass_rate = tests_passed / tests_total
        except:
            raise ValueError
        if tests_passed > tests_total or tests_passed < 0 or tests_total < 0:
            raise ValueError
        if tests_passed == tests_total:
            all_passed.append(id)
    # if all_passed:
    #     all_passed = sorted(all_passed, key=lambda x: x["tests_total"])
    #     return all_passed[0]
    sorted_patches = sorted(patches, key=lambda x: (
            0 if x["tests_passed"] == x["tests_total"] else 1,
            -x["tests_passed"] if x["tests_passed"] < x["tests_total"] else 0,
            x["tests_total"],
            x["id"]
        ))
    return sorted_patches[0]


class TemporalMemory:
    """
    A temporal memory store for a coding agent.

    Stores observations as {"content": str, "timestamp": int} entries.
    Supports adding, retrieving recent entries, and pruning stale entries.
    """

    def __init__(self):
        """Initialize empty memory."""
        # TODO: Initialize an empty list to store memory entries
        self.memory = []

    def add(self, content: str, timestamp: int) -> None:
        """
        Add a memory entry.

        Args:
            content:   The observation text (must not be None or empty).
            timestamp: Non-negative integer timestamp.

        Raises:
            ValueError: if content is None/empty or timestamp < 0.
        """
        # TODO: Validate content is not None or empty
        # TODO: Validate timestamp >= 0
        # TODO: Append {"content": content, "timestamp": timestamp} to memory
        if not content or timestamp < 0:
            raise ValueError
        self.memory.append({"content": content, "timestamp": timestamp})

    def get_recent(self, n: int) -> list:
        """
        Return the n most recent entries (by timestamp, descending).

        Args:
            n: Number of entries to return (must be > 0).

        Returns:
            List of up to n entries sorted by timestamp descending.
            If fewer than n entries exist, return all.

        Raises:
            ValueError: if n <= 0.
        """
        # TODO: Validate n > 0
        # TODO: Sort entries by timestamp descending
        # TODO: Return the first n entries (or all if fewer than n exist)
        if n <= 0:
            raise ValueError
        sort_lst = sorted(self.memory, key=lambda x: x["timestamp"], reverse=True)
        return sort_lst[:n]

    def prune(self, before_timestamp: int) -> int:
        """
        Remove all entries with timestamp < before_timestamp.

        Args:
            before_timestamp: Remove entries older than this timestamp.

        Returns:
            The count of entries removed.

        Raises:
            ValueError: if before_timestamp < 0.
        """
        # TODO: Validate before_timestamp >= 0
        # TODO: Count and remove entries with timestamp < before_timestamp
        # TODO: Return the count of removed entries
        if before_timestamp < 0:
            raise ValueError
        total = len(self.memory)
        self.memory = [m for m in self.memory if m['timestamp'] > before_timestamp]
        pruned = len(self.memory)
        return total-pruned




def run_coding_agent_pipeline(task, snippets, commands, patches):
    """
    Orchestrate the full coding agent pipeline.

    Steps:
        1. Validate task (raise ValueError if None or empty).
        2. Rank snippets by token overlap with task using re.split(r'[\W_]+', ...).
           If snippets is None or empty, use [].
        3. Filter commands: keep only safe ones (same rules as Question 4).
           If commands is None, use [].
        4. Validate patch syntax: filter out patches whose "code" fails ast.parse()
           (same logic as Question 7). If patches is None or empty, use [].
        5. Select best patch from syntax-valid patches (same logic as Question 8).
           If no valid patches remain, best_patch_id = None.

    Each patch dict: {"id": str, "tests_passed": int, "tests_total": int, "code": str}

    Returns:
        {
            "top_snippet":          str or None,
            "safe_commands":        list of str,
            "best_patch_id":        str or None,
            "snippet_count":        int,
            "unsafe_command_count": int,
            "invalid_patch_count":  int,
        }

    Raises:
        ValueError: if task is None or empty.
    """
    # Step 1: Validate task
    # TODO: raise ValueError if task is None or (str and strip() == "")

    # Step 2: Rank snippets
    # TODO: if snippets is None or empty, ranked = []
    # TODO: otherwise tokenize task and each snippet["code"] with re.split(r'[\W_]+', text.lower())
    #       score = number of unique matching tokens; sort by (-score, name)

    # Step 3: Filter commands
    # TODO: if commands is None, use []
    # Unsafe patterns: "rm -rf", "rm -fr", "curl ", "wget ", "sudo ", "chmod 777",
    #                  "chmod -r", "dd if=", "mkfs", ":|:", "> /dev/", "shred"

    # Step 4: Validate patch syntax
    # TODO: if patches is None or empty, valid_patches = [], invalid_patch_count = 0
    # TODO: otherwise use ast.parse(patch["code"]) to filter — catch SyntaxError

    # Step 5: Select best patch from valid_patches
    # TODO: if no valid patches, best_patch_id = None
    # TODO: prefer tests_passed == tests_total; fewest tests_total; id asc
    #       else highest tests_passed; fewest tests_total; id asc

    # TODO: Build and return result dict
    valid_patches = []
    unsafe_count = 0
    invalid_patch_count = 0
    if task is None or task.strip() == "":
        raise ValueError
    if not snippets:
        ranked = {}
    else:
        ranked = rank_code_snippets(task, snippets) 
    top_snippet = ranked[0]["name"] if ranked else None
    if commands == [] or not commands:
        filtered_commands = []
    else:
        filtered_commands = [command for command in commands if is_safe_command(command)]
        unsafe_count = 0 if not commands else len(commands) - len(filtered_commands)
    if patches == None or patches == "":
        valid_patches = []
    else:
        for patch in patches:
            try:
                ast.parse(patch["code"])
                valid_patches.append(patch)
            except:
                invalid_patch_count += 1
    try:
        best_patch = select_best_patch(valid_patches)["id"]
    except:
        best_patch = None
    return {
        "top_snippet": top_snippet,
        "safe_commands": filtered_commands,
        "best_patch_id": best_patch,
        "snippet_count": len(ranked),
        "unsafe_command_count": unsafe_count,
        "invalid_patch_count": invalid_patch_count,
    }