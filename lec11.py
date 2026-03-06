import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *


def is_context_saturated(used_tokens, max_tokens, threshold=0.8):
    """
    Determine whether a context window has reached its saturation threshold.

    Args:
        used_tokens (int): number of tokens currently used
        max_tokens (int): maximum context window size (must be > 0)
        threshold (float): saturation fraction in (0, 1] — default 0.8

    Returns:
        bool: True if used_tokens / max_tokens >= threshold, else False

    Raises:
        ValueError: if max_tokens <= 0
        ValueError: if threshold is not in (0, 1]
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If max_tokens <= 0, raise ValueError("max_tokens must be > 0")
    #   - If threshold is not in (0, 1], raise ValueError("threshold must be in (0, 1]")
    #
    # Step 2: Compute the fill ratio: used_tokens / max_tokens
    #
    # Step 3: Return True if fill_ratio >= threshold, else False
    
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if threshold <=0 or not threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if used_tokens / max_tokens >= threshold:
        return True
    return False



tools_lst = [
    {"name": "execute_sql",  "allowed_roles": ["data_analyst"]},
    {"name": "read_file",    "allowed_roles": ["data_analyst", "code_reviewer"]},
    {"name": "web_search",   "allowed_roles": ["web_researcher"]},
    {"name": "run_python",   "allowed_roles": ["data_analyst", "debugger"]},
]

def filter_tools_for_agent(agent_role, available_tools):
    """
    Filter available tools to only those appropriate for a given agent role.

    Args:
        agent_role (str): role of the agent (e.g., 'data_analyst', 'web_researcher')
        available_tools (list[dict]): pool of tool dicts, each with 'name' (str)
                                      and 'allowed_roles' (list[str])

    Returns:
        list[str]: sorted list of tool names whose 'allowed_roles' includes agent_role
    """
    result = []
    for tool in available_tools:
        name = tool["name"]
        try:
            roles = tool["allowed_roles"]
        except:
            roles = []
        if agent_role in roles:
            result.append(name)
    result.sort()
    return (result)


def build_subagent_dispatch(role, description, tools, context_summary, max_tokens=4096):
    """
    Build a dispatch record for launching a sub-agent.

    Args:
        role (str): agent role name (e.g., 'data_analyst')
        description (str): system prompt / persona for the sub-agent
        tools (list[str]): tool names available to this sub-agent
        context_summary (str): distilled context to pass to the sub-agent
        max_tokens (int): max tokens for sub-agent response (default 4096)

    Returns:
        dict with keys: 'role', 'description', 'tools', 'context_summary', 'max_tokens'

    Raises:
        ValueError: if role is empty after stripping
        ValueError: if description is empty after stripping
        ValueError: if tools is not a list
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If tools is not a list, raise ValueError("tools must be a list")
    #   - If role.strip() is empty, raise ValueError("role cannot be empty")
    #   - If description.strip() is empty, raise ValueError("description cannot be empty")
    #
    # Step 2: Return a dict with the five required keys.
    #   - Strip leading/trailing whitespace from 'role' and 'description'
    #   - Store 'context_summary' as-is (no stripping)
    if type(tools) != list:
        raise ValueError("tools must be a list")
    if role.strip() is "":
        raise ValueError("role cannot be empty")   
    if description.strip() is "":
        raise ValueError("description cannot be empty")   
        
    result = {
    "role": role.strip(),
    "description": description.strip(),
    "tools": tools,
    "context_summary": context_summary,
    "max_tokens": max_tokens
    }


def create_context_handoff(subagent_role, result_text, key_findings, max_summary_chars=500):
    """
    Create a distilled context handoff from a sub-agent back to the orchestrator.

    Args:
        subagent_role (str): role of the sub-agent that produced this result
        result_text (str): full output from the sub-agent
        key_findings (list[str]): key findings to highlight for the orchestrator
        max_summary_chars (int): maximum characters to include in the summary (default 500)

    Returns:
        dict with keys:
            'agent'        : subagent_role (str)
            'summary'      : result_text[:max_summary_chars] (str)
            'key_findings' : copy of key_findings list
            'char_count'   : len(summary) after truncation (int)
    """
    # TODO: Implement this function.
    #
    # Step 1: Truncate result_text to max_summary_chars using slicing.
    #   summary = result_text[:max_summary_chars]
    #
    # Step 2: Return a dict with the four required keys.
    #   - 'char_count' = len(summary) (the truncated string, not the original)
    #   - 'key_findings' should be a copy of the input list (use list(...))
    summary = result_text[:max_summary_chars]
    result = {
        'agent':subagent_role,
        'summary': summary,
        'char_count': len(summary),
        'key_findings': list(key_findings)
    }
    return result


def score_system_prompt(system_prompt, min_words=10, max_words=200):
    """
    Score a sub-agent system prompt for the 'Goldilocks' property.

    Args:
        system_prompt (str): the system prompt text
        min_words (int): minimum word count for a non-vague prompt (default 10)
        max_words (int): maximum word count before over-specification (default 200)

    Returns:
        dict with keys:
            'word_count'         : int
            'score'              : one of 'too_vague', 'goldilocks', 'too_specific'
            'in_goldilocks_zone' : bool (True iff score == 'goldilocks')
    """
    # TODO: Implement this function.
    #
    # Step 1: Tokenize by calling system_prompt.strip().split()
    #         word_count = len(words)
    #
    # Step 2: Classify:
    #   - word_count < min_words  → score = 'too_vague'
    #   - word_count > max_words  → score = 'too_specific'
    #   - otherwise               → score = 'goldilocks'
    #
    # Step 3: Return {'word_count': wc, 'score': score, 'in_goldilocks_zone': score == 'goldilocks'}
    prompt_tokens = system_prompt.strip().split()
    word_count = len(prompt_tokens)
    if word_count < min_words:
        score = 'too_vague'
    elif word_count > max_words:
        score = 'too_specific'
    else:
        score = 'goldilocks'

    result = {'word_count': word_count, 'score': score, 'in_goldilocks_zone': score == 'goldilocks'}
    return result


def distribute_context_budget(total_tokens, agent_weights):
    """
    Distribute a total context token budget proportionally across sub-agents.

    Args:
        total_tokens (int): total token budget to distribute (must be > 0)
        agent_weights (dict[str, float]): mapping from agent name to weight (each > 0)

    Returns:
        dict[str, int]: mapping from agent name to integer token budget.
                        Budgets use floor division; any remainder goes to the
                        highest-weight agent. Sum always equals total_tokens.

    Raises:
        ValueError: if total_tokens <= 0
        ValueError: if agent_weights is empty
        ValueError: if any weight is <= 0
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - Raise ValueError if total_tokens <= 0
    #   - Raise ValueError if agent_weights is empty
    #   - Raise ValueError if any weight <= 0
    #
    # Step 2: Compute total_weight = sum of all weights.
    #
    # Step 3: For each agent, compute budget = int((weight / total_weight) * total_tokens)
    #
    # Step 4: Handle remainder.
    #   - remainder = total_tokens - sum(budgets.values())
    #   - Add remainder to the agent with the highest weight (use max(..., key=...))
    #
    # Step 5: Return the budgets dict.
    if total_tokens <= 0:
        raise ValueError()
    if agent_weights == {}:
        raise ValueError()
        
    if any(weight  <= 0 for weight in agent_weights.values()):
        raise ValueError()
    total_weight = sum(agent_weights.values())
    budgets = {}
    agent_max_weight = 0
    agent_max_name = None
    for agent in agent_weights.keys():
        agent_weight = agent_weights[agent]
        if agent_weight > agent_max_weight:
            agent_max_name = agent
            agent_max_weight = agent_weight
        agent_budget = int((agent_weight / total_weight) * total_tokens)
        budgets[agent] = agent_budget
    remainder = total_tokens - sum(budgets.values())
    budgets[agent_max_name] = budgets[agent_max_name] + remainder
    return budgets

def decompose_task(task_description, available_agents, keywords_map):
    """
    Decompose a task description into sub-task assignments for specific sub-agents
    using keyword matching.

    Args:
        task_description (str): description of the overall task
        available_agents (list[str]): list of available agent role names
        keywords_map (dict[str, list[str]]): mapping from agent role to keyword list.
                                             An agent matches if any keyword appears
                                             in task_description (case-insensitive).

    Returns:
        list[dict]: list of {'agent': str, 'subtask': task_description},
                    one per matched agent, sorted by 'agent' name.
                    Returns [] if no agents match.
    """
    # TODO: Implement this function.
    #
    # Step 1: Lowercase task_description for case-insensitive comparison.
    #
    # Step 2: For each agent in available_agents:
    #   - Get its keywords from keywords_map (default to [] if missing)
    #   - If any keyword.lower() appears as a substring of task_lower, add an assignment:
    #     {"agent": agent, "subtask": task_description}
    #
    # Step 3: Sort the assignments list by the "agent" key and return it.   
    assignments = []
    task_description = task_description
    for agent in available_agents:
        keywords = keywords_map.get(agent, [])
        for keyword in keywords:
            if keyword.lower() in task_description.lower():
                assignments.append({"agent": agent, "subtask": task_description})
    sorted_assignments = sorted(assignments, key=lambda x: x["agent"])
    return sorted_assignments


def build_agent_graph(agents, edges):
    """
    Build a directed agent communication graph.

    Args:
        agents (list[str]): list of agent names (graph nodes)
        edges (list[tuple[str, str]]): directed edges (from_agent, to_agent)

    Returns:
        dict with keys:
            'nodes'     : sorted list of agent names
            'edges'     : list of {'from': str, 'to': str} in input order
            'adjacency' : dict[str, list[str]] mapping each agent to its sorted
                          outgoing neighbors; every agent appears even if no edges

    Raises:
        ValueError: if any edge endpoint is not in the agents list
    """
    # TODO: Implement this function.
    #
    # Step 1: Build a set of known agents for quick lookup.
    #
    # Step 2: Validate edges — for each (from_a, to_a), raise ValueError if either
    #         endpoint is not in the agents set.
    #
    # Step 3: Initialize adjacency = {a: [] for a in agents}.
    #
    # Step 4: Build edge_dicts and populate adjacency:
    #   - edge_dicts.append({"from": from_a, "to": to_a})
    #   - adjacency[from_a].append(to_a)
    #
    # Step 5: Sort each adjacency list in-place.
    #
    # Step 6: Return {"nodes": sorted(agents), "edges": edge_dicts, "adjacency": adjacency}
    edge_dicts = []
    adjacency = {agent: [] for agent in agents}
    for edge in edges:
        agent_1 = edge[0]
        agent_2 = edge[1]
        if agent_1 not in agents or agent_2 not in agents:
            raise ValueError("invalid edge")
        edge_dict = {"from": agent_1, "to": agent_2}
        edge_dicts.append(edge_dict)
        adjacency[agent_1].append(agent_2)
    for agent in adjacency.values():
        agent.sort()

    result = {"nodes": sorted(agents), "edges": edge_dicts, "adjacency": adjacency}
    return result


def integrate_subagent_results(handoffs):
    """
    Merge context handoffs from multiple sub-agents into a unified orchestrator context.

    Args:
        handoffs (list[dict]): list of handoff dicts, each with keys:
                               'agent' (str), 'summary' (str),
                               'key_findings' (list[str]), 'char_count' (int)

    Returns:
        dict with keys:
            'agents_consulted' : sorted list of agent names
            'combined_summary' : summaries joined as "[{agent}]: {summary}"
                                 separated by "\n---\n", in input order
            'all_key_findings' : flat list of all key_findings in input order
            'total_chars'      : sum of all char_count values

        If handoffs is empty, returns:
            {'agents_consulted': [], 'combined_summary': '', 'all_key_findings': [], 'total_chars': 0}
    """
    # TODO: Implement this function.
    #
    # Step 1: Handle the empty case.
    #
    # Step 2: Build agents_consulted = sorted([h["agent"] for h in handoffs])
    #
    # Step 3: Build the combined_summary:
    #   parts = [f"[{h['agent']}]: {h['summary']}" for h in handoffs]
    #   combined = "\n---\n".join(parts)
    #
    # Step 4: Flatten key_findings across all handoffs (in order).
    #
    # Step 5: Sum char_counts.
    #
    # Step 6: Return the result dict.
    if handoffs == {} or len(handoffs) < 1:
        return {'agents_consulted': [], 'combined_summary': '', 'all_key_findings': [], 'total_chars': 0}
    
    agents_consulted = sorted([h["agent"] for h in handoffs])
    parts = [f"[{h['agent']}]: {h['summary']}" for h in handoffs]
    combined = "\n---\n".join(parts)
    all_key_findings = []
    for handoff in handoffs:
        all_key_findings.extend(handoff["key_findings"])
    total_chars = sum(handoff["char_count"] for handoff in handoffs)
    result = {'agents_consulted': agents_consulted, 'combined_summary': combined, 'all_key_findings': all_key_findings, 'total_chars': total_chars}
    return result

handoffs = [
    {"agent": "data_analyst",   "summary": "Trend up 12%",    "key_findings": ["up 12%"],          "char_count": 11},
    {"agent": "web_researcher", "summary": "3 new datasets",  "key_findings": ["dataset A", "dataset B"], "char_count": 14},
]

print(integrate_subagent_results(handoffs))