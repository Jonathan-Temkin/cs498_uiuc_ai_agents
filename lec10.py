import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *

def compute_token_f1(prediction, ground_truth):
    """
    Compute token-level F1 score between prediction and ground truth.

    Args:
        prediction (str): agent's predicted answer
        ground_truth (str): reference correct answer

    Returns:
        dict: {"precision": float, "recall": float, "f1": float}
              all values rounded to 4 decimal places
              all 0.0 if either tokenized sequence is empty
    """
    # TODO: Implement this function.
    # Step 1: Tokenize both strings.
    #   - Lowercase the text
    #   - Replace every character in string.punctuation with ' '
    #   - Split on whitespace to get a list of tokens
    #
    # Step 2: Handle empty token lists → return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    #
    # Step 3: Build Counter objects for both token lists.
    #   - Use Counter(pred_tokens) & Counter(gt_tokens) to get the element-wise minimum (overlap)
    #   - num_common = sum of values in the overlap Counter
    #
    # Step 4: Compute:
    #   - precision = num_common / total pred tokens
    #   - recall    = num_common / total gt tokens
    #   - f1 = 2*precision*recall / (precision+recall), or 0.0 if both are 0
    #
    # Step 5: Return {"precision": round(..., 4), "recall": round(..., 4), "f1": round(..., 4)}
    prediction = prediction.translate(str.maketrans(string.punctuation, " " * len(string.punctuation))).lower().split()
    num_pred = len(prediction)
    ground_truth = ground_truth.translate(str.maketrans(string.punctuation, " " * len(string.punctuation))).lower().split()
    num_true = len(ground_truth)
    matches  = Counter(prediction) & Counter(ground_truth) 
    num_matches = sum(matches.values())
    precision = num_matches / num_pred
    recall = num_matches / num_true
    divisor = (0.0 if (precision + recall) == 0 else (precision + recall))
    f1 = 2*precision*recall / divisor
    result = {
        "precision": precision, 
        "recall": recall,
        "f1":f1
    }
    return result

#print(compute_token_f1("the cat sat on the mat", "the cat sat"))


def detect_do_nothing_vulnerability(tasks, score_fn, threshold=0.5):
    """
    Detect if a benchmark is vulnerable to a do-nothing agent.

    A do-nothing agent always returns "" (empty string) as its response.
    If such an agent achieves an average score >= threshold, the benchmark
    has a task validity issue.

    Args:
        tasks (list): list of task description strings
        score_fn (callable): score_fn(task, response) -> float in [0.0, 1.0]
        threshold (float): vulnerability threshold, default 0.5

    Returns:
        dict: {"vulnerable": bool, "average_score": float (rounded to 4 decimal places)}
    """
    # TODO: Implement this function.
    # 1. Handle empty tasks list: return {"vulnerable": False, "average_score": 0.0}
    # 2. For each task, call score_fn(task, "") — empty string is the do-nothing response
    # 3. Compute the average score, rounded to 4 decimal places
    # 4. Return vulnerable=True if average_score >= threshold
    scores = []
    vulnerable = False
    if tasks == []:
        return {"vulnerable": vulnerable, "average_score": 0.0}
    for task in tasks:
        task_score = score_fn(task, "") 
        scores.append(task_score)
    avg_score = round(sum(scores) / len(scores), 4)
    if avg_score >= threshold:
        vulnerable = True
    return {"vulnerable": vulnerable, "average_score":avg_score }



def build_stratified_golden_suite(candidates, min_quality_score=7.0, n_per_category=2):
    """
    Build a stratified golden evaluation suite with balanced category coverage.

    Args:
        candidates (list): list of dicts with keys:
            - "task_id" (str): unique identifier
            - "human_verified" (bool): whether a human verified this example
            - "quality_score" (float): quality rating 0–10
            - "category" (str): task category label
        min_quality_score (float): minimum quality score (inclusive), default 7.0
        n_per_category (int): max examples per category, default 2

    Returns:
        list: selected candidates ordered by:
              (1) category alphabetically
              (2) within category: quality_score descending, task_id ascending for ties
    """
    # TODO: Implement this function.
    #
    # Step 1 — Filter: keep only candidates where
    #   human_verified == True AND quality_score >= min_quality_score
    #
    # Step 2 — Group: use defaultdict(list) to group filtered candidates by "category"
    #
    # Step 3 — Select: for each category, sort by (-quality_score, task_id),
    #   then keep only the first n_per_category entries
    #
    # Step 4 — Combine: iterate over categories in sorted() order and extend a result list
    #
    # Return the result list (empty list if nothing qualifies)
    
    filtered_candidates = []
    task_ids_lst = []
    for candidate in candidates:
        can_human_verified = candidate["human_verified"]
        can_quality_score = candidate["quality_score"]
        if can_human_verified and can_quality_score >= min_quality_score:
            filtered_candidates.append(candidate)
    groups = defaultdict(list)
    for candidate in filtered_candidates:
        groups[candidate["category"]].append(candidate)
    for category in groups:
        groups[category] = sorted(
        groups[category],
        key=lambda x: (-x['quality_score'], x['task_id'])
    )[:n_per_category]
    groups = dict(groups)
    for group in groups.values():
        for item  in group:
            task_ids_lst.append(item['task_id'])
    return task_ids_lst

candidates = [
    {"task_id": "m1", "human_verified": True,  "quality_score": 9.0, "category": "math"},
    {"task_id": "m2", "human_verified": True,  "quality_score": 8.0, "category": "math"},
    {"task_id": "m3", "human_verified": True,  "quality_score": 7.5, "category": "math"},
    {"task_id": "c1", "human_verified": True,  "quality_score": 8.5, "category": "code"},
    {"task_id": "c2", "human_verified": False, "quality_score": 9.5, "category": "code"},
    {"task_id": "l1", "human_verified": True,  "quality_score": 6.0, "category": "lang"},
]

#print(build_stratified_golden_suite(candidates, min_quality_score=7.0, n_per_category=2))



def isolate_trial_environment(base_state):
    """
    Create an independent deep copy of the base environment state for a trial.

    Mutations to the returned copy must NOT affect base_state, even for nested
    dicts and lists.

    Args:
        base_state (dict): base environment state (may contain nested structures)

    Returns:
        dict: independent deep copy of base_state
    """
    base_state = copy.deepcopy(base_state)
    return base_state


def run_trials(base_state, trial_fn, num_trials):
    """
    Run a trial function multiple times, each with its own isolated environment.

    Args:
        base_state (dict): base environment state
        trial_fn (callable): trial_fn(env) -> any value; may mutate env
        num_trials (int): number of trials to run

    Returns:
        list: return values from each trial, in order
    """
    results = []
    for trial_num in range(num_trials):
        copy = isolate_trial_environment(base_state)
        result = trial_fn(copy)
        results.append(result)
    return results 


def compute_unbiased_pass_at_k(n, c, k):
    """
    Compute the unbiased pass@k estimator (Chen et al., 2021 / Codex paper).

    Formula: pass@k = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)

    Args:
        n (int): total number of samples run
        c (int): number of correct samples
        k (int): k in pass@k (must satisfy k <= n and c <= n)

    Returns:
        float: pass@k estimate rounded to 4 decimal places

    Raises:
        ValueError: if k > n or c > n (with a descriptive message)
    """
    # TODO: Implement this function.
    #
    # 1. Raise ValueError if k > n or c > n.
    # 2. Special cases:
    #    - c == 0      → return 0.0
    #    - n - c < k   → return 1.0  (not enough wrong answers to fill k draws)
    # 3. General case: compute the product
    #      product = 1.0
    #      for i in range(k):
    #          product *= (n - c - i) / (n - i)
    #    return round(1.0 - product, 4)
    if c < 0 or k > n  or c > n:
        raise ValueError("invalid inputs")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return round(1.0 - prod, 4)


def evaluate_dataset_pass_at_k(all_trial_results, k):
    """
    Apply the unbiased pass@k estimator across a dataset of tasks.

    Args:
        all_trial_results (list[list[bool]]): one list of bool results per task
        k (int): k in pass@k (must be <= len of each inner list)

    Returns:
        dict: {"per_task": [float, ...], "mean_pass_at_k": float}
              all values rounded to 4 decimal places
              {"per_task": [], "mean_pass_at_k": 0.0} for empty input
    """
    # TODO: Implement this function.
    # 1. Handle empty all_trial_results → return {"per_task": [], "mean_pass_at_k": 0.0}
    # 2. For each task's results: n = len(results), c = sum(results)
    #    call compute_unbiased_pass_at_k(n, c, k) and collect in per_task list
    # 3. mean = round(sum(per_task) / len(per_task), 4)
    # 4. Return {"per_task": per_task, "mean_pass_at_k": mean}
    per_task = []
    if all_trial_results == []:
        return {"per_task": [], "mean_pass_at_k": 0.0}
    for result in all_trial_results:
        n = len(result)
        c = sum(result)
        pass_result = compute_unbiased_pass_at_k(n, c, k)
        per_task.append(pass_result)
    mean = round(sum(per_task) / len(per_task), 4)
    return {"per_task": per_task, "mean_pass_at_k": mean}


def detect_outcome_validity_issue(graded_results, threshold=0.1):
    """
    Detect outcome validity issues: cases where automated grading accepts wrong solutions.

    A false positive is a result where passes_tests=True but is_actually_correct=False.

    Args:
        graded_results (list[dict]): each dict has:
            - "passes_tests" (bool): automated grader result
            - "is_actually_correct" (bool): ground truth
        threshold (float): false positive rate threshold, default 0.1

    Returns:
        dict: {
            "false_positive_count": int,
            "total_positives": int,
            "false_positive_rate": float (rounded to 4 decimal places; 0.0 if no positives),
            "has_validity_issue": bool (True if false_positive_rate > threshold)
        }
    """
    # TODO: Implement this function.
    # 1. Count total_positives: results where passes_tests=True
    # 2. Count false_positives: results where passes_tests=True AND is_actually_correct=False
    # 3. Compute false_positive_rate = false_positives / total_positives (0.0 if total_positives=0)
    # 4. has_validity_issue = false_positive_rate > threshold
    num_false_pos = 0
    num_total_pos = 0
    for result in graded_results:
        result_pass_test = result["passes_tests"]
        result_true = result["is_actually_correct"]
        if result_pass_test:
            num_total_pos += 1
        if result_pass_test and not result_true:
            num_false_pos += 1
    false_positive_rate = 0.0 if num_total_pos== 0 else num_false_pos / num_total_pos 
    has_validity_issue = false_positive_rate > threshold
    result = {
      "false_positive_count": num_false_pos,
      "total_positives": num_total_pos,
      "false_positive_rate": false_positive_rate,
      "has_validity_issue": has_validity_issue  
    }
    return result


def grade_agent_response(response, expected, method):
    """
    Grade an agent's response using a specified grading method.

    Args:
        response (str): agent's response
        expected (str): correct answer
        method (str): one of:
            - "exact_match": response.strip().lower() == expected.strip().lower()
            - "substring_match": expected (stripped, lowered) is in response (stripped, lowered)
            - "numeric_tolerance": both parsed as float, correct if within 1% relative difference
            - "starts_with": response (stripped, lowered) starts with expected (stripped, lowered)

    Returns:
        bool: True if the response is correct by the given method

    Raises:
        ValueError: if an unknown method string is provided
    """
    # TODO: Implement this function.
    # Handle all four methods and raise ValueError for unknown methods.
    # For numeric_tolerance: use abs(rv - ev) / max(abs(ev), 1e-9) <= 0.01
    # Return False (not raise) if numeric parsing fails.
    if method == "exact_match":
        return response.strip().lower() == expected.strip().lower()
    elif method == "substring_match":
        return expected.strip().lower() in response.strip().lower()
    elif method == "numeric_tolerance":
        r = float(response)
        e = float(expected)
        return abs(r - e) / max(abs(e), 1e-9) <= 0.01
    elif method == "starts_with":
        return response.strip().lower().startswith(expected.strip().lower())
    else:
        raise ValueError('Unexpected user input')



def llm_judge(question, response, rubric, api_key):
    """
    Use Claude as a judge to evaluate an agent's response against a rubric.

    Args:
        question (str): the question that was asked
        response (str): the agent's response to evaluate
        rubric (str): scoring rubric describing what makes a good answer
        api_key (str): Anthropic API key

    Returns:
        dict: {"score": int (0-10), "reasoning": str}
              score is -1 if parsing fails
    """
    url = "https://api.anthropic.com/v1/messages"

    # TODO: Implement this function using the requests library.
    #
    # 1. Set up headers:
    #      "x-api-key": api_key
    #      "anthropic-version": "2023-06-01"
    #      "content-type": "application/json"
    #
    # 2. Build a prompt that includes question, response, and rubric.
    #    Instruct Claude to reason step by step and end with "Score: <N>"
    #    where N is an integer 0-10.
    #
    # 3. POST to url with headers and json body:
    #      model: "claude-sonnet-4-6", max_tokens: 512, temperature: 0
    #      messages: [{"role": "user", "content": prompt}]
    #
    # 4. Check response.status_code == 200; raise Exception on failure.
    #
    # 5. Extract text: response.json()["content"][0]["text"]
    #
    # 6. Parse score: search for "Score: <N>" pattern (re.search),
    #    clamp to [0,10]. Fall back to first integer in [0,10] if not found.
    #    Return -1 if nothing parseable.
    #
    # 7. Return {"score": score, "reasoning": full_text}
    system_prompt = """
    YOU ARE A JUDGE. YOU WILL JUDGE THE PROVIDED RESPONSE BASED OFF THE PROVIDED Q ands rubric. YOU WILL RETURN THE FOLLOWING FORMAT WITHOUT ANY OTHER TEXT: {"score": 10, "reasoning": "The response correctly identifies Paris...ENSURE THE RESPONSE CAN BE PARSED AS A PYTHON DICTIONARY"}
    """
    messages = f"QUESTION:{question}, RESPONSE:{response}, RUBRIC: {rubric}"
    prefill = "{"
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = prefill, print_response = False,
                    temperature = None)
    response = prefill + response['content'][0]['text']
    print(type(response))
    return json.loads(response)

question = "who is jeff epstein"
response = "pedo friend of bill gates he ate babies"
rubric = "be based AF"

# print(llm_judge(question, response, rubric, api_key))



def classify_eval_cases(historical_results, current_results):
    """
    Classify evaluation cases by comparing historical and current results.

    Args:
        historical_results (dict): task_id -> bool (results from previous eval run)
        current_results (dict): task_id -> bool (results from current eval run)

    Returns:
        dict: task_id -> classification string for every task in current_results:
            "regression"  : was True in historical, now False
            "improvement" : was False in historical, now True
            "stable_pass" : was True in historical, still True
            "stable_fail" : was False in historical, still False
            "new_pass"    : task not in historical, now True
            "new_fail"    : task not in historical, now False
    """
    # TODO: Implement this function.
    # Iterate over current_results.
    # For each task_id, check if it exists in historical_results.
    # Classify based on (historical value, current value) combinations.
    result_dict = {}
    for result_id in current_results.keys():
        result = current_results[result_id]
        if result_id in historical_results.keys():
            historical_result = historical_results[result_id]
            if historical_result and not result:
                result_dict[result_id] = "regression"
            elif result and not historical_result:
                result_dict[result_id] = "improvement"
            elif result and  historical_result:
                result_dict[result_id] = "stable_pass"
            elif not result and not historical_result:
                result_dict[result_id] = "stable_fail"
        else:
            if result:
                result_dict[result_id] = "new_pass"
            else:
                result_dict[result_id] = "new_fail"  
    return result_dict




def run_evaluation_pipeline(tasks, agent_fn, num_trials, baseline_results=None):
    """
    Run a complete evaluation pipeline integrating trial execution, metric aggregation,
    and regression detection.

    Args:
        tasks (list[str]): list of task IDs to evaluate
        agent_fn (callable): agent_fn(task_id) -> bool, called once per trial
        num_trials (int): number of independent trials per task
        baseline_results (dict, optional): task_id -> bool from a previous evaluation run

    Returns:
        dict with keys:
            "trial_results"     : dict[task_id -> list[bool]] (length = num_trials each)
            "pass_at_k"         : dict[task_id -> bool]       (any trial succeeded)
            "pass_to_k"         : dict[task_id -> float]      (fraction succeeded, 4dp)
            "overall_pass_at_k" : float                       (mean of pa
            ss_at_k, 4dp)
            "overall_pass_to_k" : float                       (mean of pass_to_k, 4dp)
            "regression_report" : dict[task_id -> str] or None if no baseline provided
                                  Categories: "regression", "improvement", "stable_pass",
                                              "stable_fail", "new_pass", "new_fail"
    """
    # TODO: Implement this function.
    # 1. For each task, call agent_fn(task_id) num_trials times; store in trial_results
    # 2. Compute pass_at_k (any True) and pass_to_k (fraction True, 4dp) per task
    # 3. Compute overall_pass_at_k and overall_pass_to_k as means across tasks (4dp)
    # 4. If baseline_results is provided, classify each task_id against baseline pass_at_k
    # 5. Return all results in a single dict
    trial_results = {}
    pass_at_k_dict = {}
    pass_to_k_dict = {}
    regression_report = {}
    for task_id in tasks:
        task_results_lst = []
        for _ in range(num_trials):
            trial_result =agent_fn(task_id)
            task_results_lst.append(trial_result)
        pass_at_k = any(task_results_lst)
        pass_to_k = sum(task_results_lst) / len(task_results_lst)
        if baseline_results:
            current_results = {task_id :task_results_lst }
            regression = classify_eval_cases(baseline_results, current_results)
            regression_report[task_id] = regression[task_id]
        trial_results[task_id]  = task_results_lst
        pass_at_k_dict[task_id]  = pass_at_k
        pass_to_k_dict[task_id]  = pass_to_k
    try:
        overall_pass_at_k = [pass_at_k_dict[id] for id in pass_at_k_dict.keys()]
        overall_pass_at_k = sum(overall_pass_at_k) / len(overall_pass_at_k)
    except:
        overall_pass_at_k  = 0
    report = {
        "trial_results":trial_results,
        "pass_at_k":pass_at_k,
        "pass_to_k":pass_to_k, 
        "overall_pass_at_k": overall_pass_at_k,
        "regression_report": regression_report
    }
    return report

def mock_agent(bs): return True 
tasks = ["t1", "t2"]
baseline = {"t1": True, "t2": False}
report = run_evaluation_pipeline(tasks, mock_agent, num_trials=4, baseline_results=baseline)
print(report)