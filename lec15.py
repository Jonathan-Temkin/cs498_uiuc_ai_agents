import re
import requests

def parse_tool_call(response):
    """
    Parse a tool-use model response and extract the tool name and argument.

    Tool-calling LLMs are trained to produce responses of the form:
        <tool>calculator</tool><arg>sqrt(144)</arg>

    Args:
        response (str): the raw response string from the model

    Returns:
        dict: {"tool": str_or_None, "arg": str_or_None}
              - None if the tag is absent
              - "" (empty string) if the tag is present but has no content
              - Stripped string if content is present
    """
    # TODO: Implement this function.
    #
    # Step 1: Use re.search with pattern r'<tool>(.*?)</tool>' and re.DOTALL
    #         to find the <tool>...</tool> block in response.
    #         If found, extract group(1) and strip whitespace.
    #         If not found, use None.
    #
    # Step 2: Use re.search with pattern r'<arg>(.*?)</arg>' and re.DOTALL
    #         to find the <arg>...</arg> block in response.
    #         If found, extract group(1) and strip whitespace.
    #         If not found, use None.
    #
    # Step 3: Return {"tool": <result from step 1>, "arg": <result from step 2>}
    pattern = '<tool>(.*?)</tool>'
    search_result = re.search(pattern, response, re.DOTALL)
    if search_result:
        tool_result = search_result.group(1).strip()
    else:
        tool_result = None
    pattern = '<arg>(.*?)</arg>'
    search_result = re.search(pattern, response, re.DOTALL)
    if search_result:
        arg_result = search_result.group(1).strip()
    else:
        arg_result = None
    return {"tool": tool_result, "arg": arg_result}


def validate_tool_format(response):
    """
    Validate that a model response has the correct tool-call XML format.

    Returns True only if:
        1. <tool>...</tool> is present with non-empty content (after stripping)
        2. <arg>...</arg> is present with non-empty content (after stripping)
        3. </tool> appears before <arg> in the string

    Args:
        response (str): the raw response string from the model

    Returns:
        bool: True if all three conditions are satisfied, False otherwise
    """
    # TODO: Implement this function.
    #
    # Step 1: Use re.search with r'<tool>(.*?)</tool>' and re.DOTALL to find
    #         the tool match. Use re.search with r'<arg>(.*?)</arg>' and re.DOTALL
    #         to find the arg match.
    #
    # Step 2: If either match is None, return False.
    #
    # Step 3: Check that tool_match.group(1).strip() is non-empty.
    #         Check that arg_match.group(1).strip() is non-empty.
    #         If either is empty, return False.
    #
    # Step 4: Check ordering: find the index of '</tool>' in response using
    #         response.index('</tool>') and the index of '<arg>' using
    #         response.index('<arg>'). If tool_end >= arg_start, return False.
    #
    # Step 5: If all checks pass, return True.
    tool_call_components = parse_tool_call(response)
    tool = tool_call_components["tool"]
    arg = tool_call_components["arg"]
    if not tool or not arg: 
        return False
    tool_index = response.index('</tool>')
    arg_index = response.index('<arg>')
    if tool_index >= arg_index: 
        return False
    return True


def compute_tool_reward(predicted_tool, predicted_arg, correct_tool, correct_arg):
    """
    Compute a shaped reward for a tool selection prediction.

    Reward shaping provides intermediate rewards to guide RL training:
        +3: correct tool AND correct argument (full credit)
        +2: correct tool, wrong argument (tool routing is correct)
        +1: format valid (non-None, non-empty) but wrong tool
         0: format broken (tool or arg is None or empty string)

    Args:
        predicted_tool (str or None): the tool name predicted by the model
        predicted_arg  (str or None): the argument predicted by the model
        correct_tool   (str): the ground-truth tool name
        correct_arg    (str): the ground-truth argument

    Returns:
        int: one of 0, 1, 2, or 3
    """
    # TODO: Implement this function.
    #
    # Step 1: Check if the format is broken.
    #         If predicted_tool is None or "" (falsy), OR
    #         if predicted_arg is None or "" (falsy), return 0.
    #
    # Step 2: Check if both tool and arg are correct.
    #         If predicted_tool == correct_tool AND predicted_arg == correct_arg,
    #         return 3.
    #
    # Step 3: Check if only the tool is correct.
    #         If predicted_tool == correct_tool (but arg differs), return 2.
    #
    # Step 4: Otherwise, format is valid but tool is wrong. Return 1.
    if not predicted_tool or not predicted_arg:
        return 0
    if predicted_tool == correct_tool and predicted_arg == correct_arg:
        return 3
    if predicted_tool == correct_tool:
        return 2
    else:
        return 1

def check_arg_type(arg_str, expected_type):
    """
    Validate that a string argument represents a value of the expected type.

    Args:
        arg_str       (str): the argument string to validate
        expected_type (str): one of "int", "float", "str", "bool"

    Returns:
        bool: True if arg_str represents a valid value of expected_type

    Raises:
        ValueError: if expected_type is not in {"int", "float", "str", "bool"}

    Rules:
        "int":   int(arg_str.strip()) must succeed AND arg_str.strip() must not contain "."
        "float": float(arg_str.strip()) must succeed
        "bool":  arg_str.strip().lower() must be in {"true", "false", "0", "1"}
        "str":   always True
    """
    # TODO: Implement this function.
    #
    # Step 1: Check if expected_type is valid.
    #         valid_types = {"int", "float", "str", "bool"}
    #         If expected_type not in valid_types, raise ValueError.
    #
    # Step 2: Strip the arg_str: stripped = arg_str.strip()
    #
    # Step 3: Handle "str" case — always return True.
    #
    # Step 4: Handle "int" case.
    #         - If "." in stripped, return False.
    #         - Try int(stripped). If it raises ValueError, return False. Else return True.
    #
    # Step 5: Handle "float" case.
    #         - Try float(stripped). If it raises ValueError, return False. Else return True.
    #
    # Step 6: Handle "bool" case.
    #         - Return stripped.lower() in {"true", "false", "0", "1"}
    valid_types = {"int", "float", "str", "bool"}
    stripped = arg_str.strip()
    if (expected_type) not in valid_types:
        raise ValueError()
    if expected_type == "str":
        return True
    elif expected_type == "int":
        if "." in stripped:
            return False
        try:
            strippe_int = int(stripped)
            return True
        except:
            return False
    elif expected_type == "float":
        try:
            stripped_float = float(stripped)
            return True
        except:
            return False
    else:
        if stripped.lower() in {"true", "false", "0", "1"}: 
            return True
        return False
    

def build_tool_prompt(task, tools):
    """
    Build a tool-selection prompt from a task description and a list of tools.

    Args:
        task  (str): the task or question for the model to answer
        tools (list): list of dicts, each with keys "name", "description", "arg_type"

    Returns:
        str: a prompt string containing the task, tool descriptions, and format instructions

    Raises:
        ValueError: if tools is empty or None
        ValueError: if task is empty, None, or whitespace-only

    The returned prompt must contain:
        - The task text
        - Each tool's name
        - The strings "<tool>" and "<arg>" as format instructions
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #         - If task is None, empty, or whitespace-only (not task.strip()), raise ValueError.
    #         - If tools is None or empty (not tools), raise ValueError.
    #
    # Step 2: Start building the prompt.
    #         Begin with a line that includes the task text, e.g.:
    #         lines = [f"Task: {task}", ""]
    #
    # Step 3: Add a section listing available tools.
    #         For each tool in tools, add a line with the tool's name, description,
    #         and arg_type. Example:
    #         f"  - {tool['name']}: {tool['description']} (argument type: {tool['arg_type']})"
    #
    # Step 4: Add format instructions.
    #         Include a line with: "<tool>TOOL_NAME</tool><arg>ARGUMENT</arg>"
    #         This is required so the prompt contains "<tool>" and "<arg>".
    #
    # Step 5: Return "\n".join(lines)
    if not task or not task.strip():
        raise ValueError
    if not tools:
        raise ValueError
    lines = [f"Task: {task}", ""]
    for tool in tools:
        tool_line = f"  - {tool['name']}: {tool['description']} (argument type: {tool['arg_type']})"
        lines.append(tool_line)
    lines.append("<tool>TOOL_NAME</tool><arg>ARGUMENT</arg>")
    return "\n".join(lines)



def grade_sql_result(predicted_rows, expected_rows):
    """
    Grade SQL query results using order-insensitive semantic equivalence.

    Args:
        predicted_rows (list of lists): rows returned by the predicted SQL query
        expected_rows  (list of lists): rows returned by the correct SQL query

    Returns:
        float: 1.0 if the result sets are semantically equivalent, 0.0 otherwise

    Raises:
        ValueError: if either argument is not a list

    Notes:
        - All values are converted to str before comparison
        - Row order is ignored (compare sorted lists of rows)
        - Column order within each row is preserved
        - Two empty lists are considered equal (returns 1.0)
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #         If not isinstance(predicted_rows, list), raise ValueError.
    #         If not isinstance(expected_rows, list), raise ValueError.
    #
    # Step 2: Define a normalization helper.
    #         Convert each row to a list of strings: [str(v) for v in row]
    #         Then sort the list of rows: sorted([...] for row in rows)
    #         This gives an order-independent, type-normalized representation.
    #
    # Step 3: Compare the normalized versions.
    #         If normalize(predicted_rows) == normalize(expected_rows), return 1.0.
    #         Otherwise, return 0.0.
    if not isinstance(predicted_rows, list):
        raise ValueError
    if not isinstance(expected_rows, list):
        raise ValueError
    predict_normalization_helper = sorted([[str(v) for v in row] for row in predicted_rows])
    expected_normalization_helper = sorted([[str(v) for v in row] for row in expected_rows])
    if expected_normalization_helper == predict_normalization_helper:
        return 1.0
    return 0.0


def evaluate_tool_policy(samples, policy_fn):
    """
    Evaluate a tool-use policy on a set of labeled samples.

    Uses the same +3/+2/+1/0 reward rule as compute_tool_reward (Q3):
        +3: correct tool AND correct arg
        +2: correct tool, wrong arg
        +1: format valid but wrong tool
         0: format broken (None or empty tool/arg)

    Args:
        samples   (list of dict): each dict has keys "prompt", "correct_tool", "correct_arg"
        policy_fn (callable): takes a prompt string, returns {"tool": str_or_None, "arg": str_or_None}

    Returns:
        dict with keys:
            "mean_reward"  (float): average reward across all samples
            "rewards"      (list of int): per-sample rewards
            "num_samples"  (int): total number of samples

    Raises:
        ValueError: if samples is empty
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #         If samples is empty (not samples), raise ValueError.
    #
    # Step 2: Initialize an empty rewards list.
    #
    # Step 3: For each sample in samples:
    #         a. Call policy_fn(sample["prompt"]) to get prediction dict.
    #         b. Extract predicted_tool = prediction.get("tool")
    #            and predicted_arg = prediction.get("arg").
    #         c. Compute reward using the +3/+2/+1/0 logic:
    #            - If predicted_tool is falsy OR predicted_arg is falsy: reward = 0
    #            - Elif predicted_tool == sample["correct_tool"] AND
    #                   predicted_arg == sample["correct_arg"]: reward = 3
    #            - Elif predicted_tool == sample["correct_tool"]: reward = 2
    #            - Else: reward = 1
    #         d. Append reward to rewards list.
    #
    # Step 4: Compute mean_reward = sum(rewards) / len(rewards)
    #
    # Step 5: Return {"mean_reward": ..., "rewards": rewards, "num_samples": len(samples)}
    if not samples:
        raise ValueError
    rewards_lst = []
    for sample in samples:
        prediction = policy_fn(sample["prompt"]) 
        predicted_tool = prediction.get("tool")
        predicted_arg = prediction.get("arg")
        if not predicted_tool or not predicted_arg:
            reward = 0
        elif predicted_tool == sample["correct_tool"] and predicted_arg == sample["correct_arg"]:
            reward = 3
        elif predicted_tool == sample["correct_tool"]:
            reward = 2
        else:
            reward = 1
        rewards_lst.append(reward)
    mean_reward = sum(rewards_lst) / len(rewards_lst)
    return {"mean_reward": mean_reward, "rewards": rewards_lst, "num_samples": len(samples)}


def compute_chain_reward(steps, gamma=0.9):
    """
    Compute the discounted cumulative reward for a chain of tool-use steps.

    For a sequence of T tool-use steps, the discounted reward is:
        R = r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^(T-1)*r_(T-1)

    Each step reward r_t uses the +3/+2/+1/0 rule:
        +3: correct tool AND correct arg
        +2: correct tool, wrong arg
        +1: format valid but wrong tool
         0: format broken (None or empty tool/arg)

    Args:
        steps (list of dict): each dict has keys:
                              "predicted_tool" (str or None)
                              "predicted_arg"  (str or None)
                              "correct_tool"   (str)
                              "correct_arg"    (str)
        gamma (float): discount factor in [0, 1]. Default: 0.9

    Returns:
        float: the discounted sum of per-step rewards

    Raises:
        ValueError: if steps is empty
        ValueError: if gamma is not in [0, 1]
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #         If not steps, raise ValueError("steps cannot be empty").
    #         If gamma < 0 or gamma > 1, raise ValueError.
    #
    # Step 2: Initialize total = 0.0
    #
    # Step 3: For each step at index t (use enumerate(steps)):
    #         a. Compute step_reward using the +3/+2/+1/0 rule:
    #            - If step["predicted_tool"] is falsy OR step["predicted_arg"] is falsy: 0
    #            - Elif both match correct: 3
    #            - Elif only tool matches: 2
    #            - Else: 1
    #         b. Add (gamma ** t) * step_reward to total.
    #
    # Step 4: Return total
    if gamma < 0 or gamma > 1:
        raise ValueError
    if not steps:
        raise ValueError("steps cannot be empty")
    total = 0.0
    for t, step in enumerate(steps):
        predicted_tool = step["predicted_tool"] 
        predicted_arg = step["predicted_arg"]
        correct_tool = step["correct_tool"]
        correct_arg = step["correct_arg"]
        if not predicted_tool or not predicted_arg:
            reward = 0
        elif predicted_tool == correct_tool and predicted_arg == correct_arg:
            reward = 3
        elif predicted_tool == correct_tool:
            reward = 2
        else:
            reward = 1
        total+= (gamma ** t) * reward
    return total


def build_rl_training_batch(prompts, responses, grader_fn):
    """
    Build a labeled RL training batch from prompts, responses, and a grader function.

    Args:
        prompts    (list of str): the input prompts
        responses  (list of str): the model's responses (same length as prompts)
        grader_fn  (callable): grader_fn(response: str, prompt: str) -> reward (int or float)

    Returns:
        list of dict: one entry per (prompt, response) pair, each with keys:
            "prompt"   (str)
            "response" (str)
            "reward"   (float or int): reward from grader_fn, or 0 if grader raised an exception

    Raises:
        ValueError: if len(prompts) != len(responses)
        ValueError: if prompts is empty
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #         If not prompts, raise ValueError("prompts cannot be empty").
    #         If len(prompts) != len(responses), raise ValueError.
    #
    # Step 2: Initialize batch = []
    #
    # Step 3: Iterate over zip(prompts, responses):
    #         For each (prompt, response) pair:
    #         a. Try to call grader_fn(response, prompt) to get reward.
    #         b. If grader_fn raises any exception, set reward = 0 (do not re-raise).
    #         c. Append {"prompt": prompt, "response": response, "reward": reward} to batch.
    #
    # Step 4: Return batch
    if not prompts:
        raise ValueError("prompts cannot be empty")
    if len(prompts) != len(responses):
        raise ValueError
    batch = []
    for prompt, response in zip(prompts, responses):
        try:
            reward = grader_fn(response, prompt)
        except:
            reward = 0
        batch.append({"prompt": prompt, "response": response, "reward": reward})
    return batch




# Few-shot prefix that matches the training prompt distribution.
# Including this in every API call is critical: the model was trained on
# completion-style prompts with these examples, so sending the same format
# at inference time elicits <tool>...</tool><arg>...</arg> responses instead
# of the base model's default <think>... behavior.
FEW_SHOT_PREFIX = (
    "Tools: calculator (math), weather (city name), search (query).\n"
    "Format: <tool>NAME</tool><arg>ARG</arg>\n\n"
    "Task: What is 2+3? <tool>calculator</tool><arg>2+3</arg>\n"
    "Task: Weather in Paris? <tool>weather</tool><arg>Paris</arg>\n"
    "Task: Search AI news? <tool>search</tool><arg>AI news</arg>\n"
)


def call_tool_finetuned_model(prompt, api_key):
    """
    Call your RL-trained Tinker tool-use model via the OpenAI-compatible API.

    Args:
        prompt  (str): the user's task/question (e.g. "What is the weather in Chicago?")
        api_key (str): Tinker API key (loaded from tinker_key.txt by the autograder)

    Returns:
        str: the model's raw response (should contain <tool>...</tool><arg>...</arg>)

    Raises:
        ValueError: if prompt is None or empty
    """
    # TODO: implement this function.
    #
    # Step 1: Validate — if not prompt, raise ValueError("prompt cannot be empty or None").
    #
    # Step 2: Set constants:
    #         TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    #         TINKER_MODEL    = "tinker://YOUR_MODEL_ID/..."  <- replace with your model path
    #
    # Step 3: Build the full prompt by appending the task to FEW_SHOT_PREFIX:
    #         full_prompt = FEW_SHOT_PREFIX + f"Task: {prompt}?"
    #
    # Step 4: Build headers:
    #         {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    #
    # Step 5: Build request body:
    #         {"model": TINKER_MODEL,
    #          "messages": [{"role": "user", "content": full_prompt}],
    #          "temperature": 0,
    #          "max_tokens": 64}
    #
    # Step 6: response = requests.post(f"{TINKER_BASE_URL}/chat/completions",
    #                                  headers=headers, json=data)
    #         response.raise_for_status()
    #
    # Step 7: return response.json()["choices"][0]["message"]["content"]
    if not prompt:
        raise ValueError("prompt cannot be empty or None")
    TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    TINKER_MODEL    = "tinker://YOUR_MODEL_ID/..."  
    full_prompt = FEW_SHOT_PREFIX + f"Task: {prompt}?"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": TINKER_MODEL,
             "messages": [{"role": "user", "content": full_prompt}],
             "temperature": 0,
             "max_tokens": 64}
    response = requests.post(f"{TINKER_BASE_URL}/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
    return "test"