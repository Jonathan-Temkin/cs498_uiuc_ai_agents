import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *
import math
import re

def check_agent_security(agent_config):
    """
    Audit an agent's security configuration.

    Args:
        agent_config (dict): Keys:
            - "input_filter" (bool), "tool_safeguards" (bool),
              "process_guards" (bool), "output_filter" (bool)
            - "tool_count" (int): number of tools available to the agent

    Returns:
        dict: {"score": int, "issues": list[str]}
            Each present defense layer adds +20 (max 80).
            tool_count > 20: -20, add "excessive_tool_count"
            tool_count > 10 (<=20): -10, add "high_tool_count"
            Missing (False) layers are added to issues.
            Score clamped to [0, 100].

    Example:
        check_agent_security({
            "input_filter": True, "tool_safeguards": True,
            "process_guards": False, "output_filter": True, "tool_count": 5
        })
        # Returns {"score": 60, "issues": ["process_guards"]}
    """
    score = 0 
    issues_lst = []
    for config_component in agent_config:
        if config_component != 'tool_count':
            if agent_config[config_component] and score <= 80:
                score += 20
            elif not agent_config[config_component]:
                issues_lst.append(config_component)
    tool_count = agent_config['tool_count']
    if tool_count > 20:
        score -= 20
        issues_lst.append("excessive_tool_count")
    elif tool_count > 10:
        score -= 10
        issues_lst.append("high_tool_count")
    if score < 0:
        score = 0 
    elif score > 100:
        score = 100
    return {"score": score, "issues": issues_lst}



def validate_reasoning_format(response):
    """
    Validate that a model response follows the reasoning format used in CoT training.

    The required format is:
        <think>...non-empty reasoning...</think>
        <answer>...non-empty answer...</answer>
    where <think> must appear before <answer>.

    Args:
        response (str): The model's output string

    Returns:
        dict: {
            "valid": bool,        # True only if all checks pass
            "has_think": bool,    # True if <think>...</think> tag found
            "has_answer": bool,   # True if <answer>...</answer> tag found
            "issues": list[str]   # Problems found (empty if valid)
        }
        Possible issues: "missing_think_tag", "missing_answer_tag",
                         "empty_think", "empty_answer", "wrong_order"

    Example:
        validate_reasoning_format("<think>step 1</think><answer>42</answer>")
        # Returns {"valid": True, "has_think": True, "has_answer": True, "issues": []}

        validate_reasoning_format("<answer>42</answer>")
        # Returns {"valid": False, "has_think": False, "has_answer": True,
        #          "issues": ["missing_think_tag"]}
    """


    issues_lst = []
    think_pattern = r"<think>(.*?)</think>"
    think_match_obj = re.search(think_pattern, response, re.DOTALL)
    think_match = think_match_obj.group(1).strip() if think_match_obj else think_match_obj
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match_obj = re.search(answer_pattern, response,re.DOTALL)
    answer_match = answer_match_obj.group(1).strip() if answer_match_obj else answer_match_obj
    think_index = response.find("<think>")
    answer_index = response.find("<answer>")
    think_first = think_index < answer_index
    if not think_first:
        issues_lst.append("wrong_order")
    valid = True if answer_match and think_match and think_first else False
    has_think = True if think_match else False
    has_answer = True if answer_match else False
    if not has_think:
        issues_lst.append("empty_think")
    if not has_answer:
        issues_lst.append("empty_answer")
    if not think_match_obj:
        issues_lst.append("missing_think_tag")
    if not think_match_obj:
        issues_lst.append("missing_answer_tag")   
    return {
            "valid": valid,        # True only if all checks pass
            "has_think":  has_think, # True if <think>...</think> tag found
            "has_answer": has_answer,   # True if <answer>...</answer> tag found
            "issues": issues_lst   # Problems found (empty if valid)
        }


def aggregate_trial_results(trial_results, metric='pass_at_k'):
    """
    Aggregate multiple trial results into a single pass/fail outcome.

    Args:
        trial_results (list of bool): Outcomes for each trial (True = pass, False = fail)
        metric (str): Aggregation method. Either 'pass_at_k' or 'pass_to_k'.
            - 'pass_at_k': True if at least one trial passed (best-case capability)
            - 'pass_to_k': True if all trials passed (reliability)

    Returns:
        bool: Aggregated result

    Raises:
        ValueError: If metric is not 'pass_at_k' or 'pass_to_k'

    Example:
        aggregate_trial_results([True, False, True], 'pass_at_k')
        # Returns True

        aggregate_trial_results([True, False, True], 'pass_to_k')
        # Returns False

        aggregate_trial_results([], 'pass_at_k')
        # Returns False
    """
    if metric not in ["pass_at_k", "pass_to_k"]:
        raise ValueError()
    if metric == "pass_at_k":
        return any(trial_results)
    else:
        result = False if trial_results  == [] else  all(trial_results)
        return result


def build_preference_pairs(completions_with_rewards, min_reward_gap=0.3):
    """
    Build a preference pair dataset for RLHF/DPO training.

    Args:
        completions_with_rewards (list of tuple): Each tuple is (text: str, reward: float)
        min_reward_gap (float): Minimum required difference reward_chosen - reward_rejected
            to include a pair (default 0.3). Pairs with smaller gaps are excluded as noisy.

    Returns:
        list of dict: Each dict has:
            - "chosen" (str): the higher-reward completion
            - "rejected" (str): the lower-reward completion
            - "reward_gap" (float): reward_chosen - reward_rejected
        Sorted by reward_gap descending. Returns [] if fewer than 2 completions.

    Example:
        completions = [("Great!", 0.9), ("Okay.", 0.5), ("Bad.", 0.1)]
        build_preference_pairs(completions, min_reward_gap=0.3)
        # Returns 3 pairs sorted by gap desc:
        # [{"chosen":"Great!","rejected":"Bad.","reward_gap":0.8}, ...]
    """

    result = []
    for i in range(len(completions_with_rewards)):
        for j in range(i + 1, len(completions_with_rewards)):
            chosen_a = completions_with_rewards[i][0]
            chosen_a_reward = completions_with_rewards[i][1]
            rejected_a = completions_with_rewards[j][0]
            rejected_a_reward = completions_with_rewards[j][1]
            if chosen_a_reward - rejected_a_reward >= min_reward_gap:
                result.append({"chosen": chosen_a, "rejected": rejected_a, "reward_gap": chosen_a_reward - rejected_a_reward })
    result.sort(key=lambda x: x["reward_gap"], reverse=True)
    return result 



def build_golden_suite(examples, min_quality_score=0.8, allow_ambiguous=False):
    """
    Build a curated golden evaluation suite from a list of examples.

    Args:
        examples (list of dict): Each dict has:
            - "input" (str): the task input
            - "expected_output" (str): the gold answer
            - "quality_score" (float): quality score between 0.0 and 1.0
            - "is_ambiguous" (bool): True if the example is ambiguous
        min_quality_score (float): Minimum acceptable quality score (default 0.8)
        allow_ambiguous (bool): If False, exclude ambiguous examples (default False)

    Returns:
        list of dict: Filtered and deduplicated examples in original order.
            - Excludes quality_score < min_quality_score
            - Excludes is_ambiguous=True when allow_ambiguous=False
            - Deduplicates by "input" (keep first occurrence)

    Example:
        examples = [
            {"input": "Q1", "expected_output": "A1", "quality_score": 0.9, "is_ambiguous": False},
            {"input": "Q2", "expected_output": "A2", "quality_score": 0.6, "is_ambiguous": False},
            {"input": "Q3", "expected_output": "A3", "quality_score": 0.95, "is_ambiguous": True},
            {"input": "Q1", "expected_output": "A1b","quality_score": 1.0, "is_ambiguous": False},
        ]
        build_golden_suite(examples)
        # Returns [{"input": "Q1", "expected_output": "A1", "quality_score": 0.9, "is_ambiguous": False}]
    """
    result = []
    inputs = []
    for example in examples:
        input = example['input']
        quality_score = example['quality_score']
        ambiguous = example['is_ambiguous']
        if (quality_score >= min_quality_score) and (allow_ambiguous or not(ambiguous)) and (input not in inputs):
            result.append(example)
        inputs.append(input)
    return result


def validate_agent_communication(graph, sender, receiver):
    """
    Validate whether a direct agent-to-agent communication is allowed.

    Args:
        graph (dict): Adjacency list of allowed directed routes.
            Keys are agent names; values are lists of agents they can send to.
            Example: {"orchestrator": ["analyst", "researcher"], "analyst": ["orchestrator"]}
        sender (str): Name of the agent sending the message
        receiver (str): Name of the agent receiving the message

    Returns:
        dict: {
            "allowed": bool,       # True if direct route exists
            "reason": str,         # "direct_route", "no_direct_route", or "unknown_sender"
            "path_length": int     # 1 if allowed, 0 otherwise
        }

    Example:
        graph = {"orchestrator": ["analyst"], "analyst": ["orchestrator"]}
        validate_agent_communication(graph, "orchestrator", "analyst")
        # Returns {"allowed": True, "reason": "direct_route", "path_length": 1}

        validate_agent_communication(graph, "analyst", "analyst")
        # Returns {"allowed": False, "reason": "no_direct_route", "path_length": 0}
    """ 
    agents  = graph.keys()
    if sender not in agents:
        return {"allowed": False, "reason": "unknown_sender", "path_length": 0}
    if receiver in graph[sender]:
        allowed = True
        reason =  "direct_route"
        path_length = 1
    else:
        allowed = False
        reason =  "no_direct_route"
        path_length = 0
    return {"allowed": allowed, "reason": reason, "path_length": path_length}


def judge_preferences(prompt, response_a, response_b, criteria, api_key):
    """
    Use an LLM to judge which of two responses better satisfies given criteria.

    Args:
        prompt: The original question or task
        response_a: First candidate response
        response_b: Second candidate response
        criteria: List of evaluation criteria strings
        api_key: Anthropic API key

    Returns:
        dict with keys:
            "preferred": "A" or "B"
            "reasoning": explanation string
    """
    messages = f"YOU WILL DECIDE WHICH PROMPT IS BETTER. Evaluation according to criteria <criteria>{criteria} </criteria> RESPONSE ONLY WITH A OR B ON THE FIRST LINE FOLLOWED BY YOUR REASONING. <A> {response_a}</A> <B>{response_b} </B> <PROMPT>{prompt}</PROMPT?"
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    
    response_text = response['content'][0]['text']
    lines = response_text.strip().split('\n', 1)
    preference = lines[0].strip() if len(lines) > 0 else ""
    reasoning = lines[1].strip() if len(lines) > 1 else ""
    return {"preferred": preference, "reasoning": reasoning}


# prompt = "who was jeffrey epstein"
# response_a = "a great man"
# response_b = "a literal piece of shit"
# criteria =  ["accuracy", "clarity"]
# print(judge_preferences(prompt, response_a, response_b, criteria, api_key))


def orchestrate_task(task, subagent_specs, api_key):
    """
    Route a task to the most appropriate specialized subagent.

    Args:
        task (str): The user's task or question
        subagent_specs (list of dict): Available subagents, each with:
            - "name" (str): subagent identifier
            - "description" (str): what this subagent handles
            - "system_prompt" (str): system prompt for this subagent
        api_key (str): Anthropic API key

    Returns:
        dict: {
            "chosen_agent": str,  # name of the selected subagent
            "result": str         # the subagent's response to the task
        }

    Implementation steps:
        Step 1: Build a routing system prompt listing all subagents and descriptions.
                Ask the LLM to respond with ONLY the chosen subagent's name.
                Call the API with this routing prompt and the task as user message.

        Step 2: Parse the routing response to find which subagent was chosen
                (case-insensitive name match). Fall back to the first subagent if
                no valid name is found.

        Step 3: Call the API again with the chosen subagent's system_prompt and
                the original task. Return the result.

    Use model: claude-sonnet-4-5-20250929, max_tokens: 512

    Example:
        subagents = [
            {"name": "math_agent", "description": "Handles math problems",
             "system_prompt": "You are a math expert. Solve the problem step by step."},
            {"name": "writing_agent", "description": "Helps with writing tasks",
             "system_prompt": "You are a writing expert. Help craft clear prose."},
        ]
        result = orchestrate_task("What is 15 * 23?", subagents, api_key)
        # result["chosen_agent"] should be "math_agent"
        # result["result"] contains the math solution
    """
    messages = f"RESPOND ONLY WITH THE CHOSEN SUBAGENT'S NAME THAT'S BESTY FOR THE TASK: <TASK> {task}</TASK> <AGENTS>{subagent_specs} </AGENTS>"
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    response_text = response['content'][0]['text']
    agent_spec = [agent for agent in subagent_specs if agent["name"] == response_text][0]
    system_prompt  = agent_spec["system_prompt"]
    response_2 = claude_api_call(task, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    response_2_text = response_2['content'][0]['text']
    return {"chosen_agent": response_text, "result": response_2_text}



def text_to_sql_with_disambiguation(question, schema, api_key):
    """
    Two-step Text-to-SQL agent with ambiguity detection.

    Args:
        question: Natural language question to convert to SQL
        schema: Dict mapping table names to lists of column names
        api_key: Anthropic API key

    Returns:
        dict with keys:
            "status": "complete" or "needs_clarification"
            "sql": SQL string if status is "complete", else None
            "clarification": clarifying question string if status is "needs_clarification", else None
    """
    messages = f"DETECT AMBIGUITY. RESPONSE WITH AMBIGUOUS: <reason> or CLEAR  ON THE FIRST LINE. <QUESTION>{question} </QUESTION> <SCHEMA>{schema}</SCHEMA> "
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    response_text = response['content'][0]['text']
    response_clear_text = None
    response_ambiguous_text = None
    clarification = None
    lines = response_text.strip().split('\n', 1)
    result =  lines[0].strip() if len(lines) > 0 else ""
    if result == "CLEAR":
        result = "complete"
        message_clear = f'generate a SQL query wrapped in <sql>...</sql>.  <QUESTION>{question} </QUESTION> <SCHEMA>{schema}</SCHEMA>'
        response_clear = claude_api_call(message_clear, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
        response_clear_text = response_clear['content'][0]['text']
    else: #result is ambiguous
        result ='needs_clarification' 
        message_ambiguous= f'generate clarifying questions for the user.  <QUESTION>{question} </QUESTION> <SCHEMA>{schema}</SCHEMA>'
        response_ambiguous = claude_api_call(message_ambiguous, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
        response_ambiguous_text = response_ambiguous['content'][0]['text']
    return {"status": result, "sql": response_clear_text , "clarification": response_ambiguous_text}