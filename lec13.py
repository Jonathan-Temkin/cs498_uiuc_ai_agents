import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *
import math
import re

def parse_reasoning_response(response):
    """
    Parse a reasoning model response and extract the thinking trace and final answer.

    Reasoning models like DeepSeek R1 wrap their chain-of-thought in <think>...</think>
    tags and the final answer in <answer>...</answer> tags. This function extracts both
    components from the raw response string.

    Args:
        response (str): the raw response string from the reasoning model

    Returns:
        dict: a dictionary with keys:
            "thinking" (str or None): content inside <think>...</think>, stripped of
                                      leading/trailing whitespace, or None if not found
            "answer"   (str or None): content inside <answer>...</answer>, stripped of
                                      leading/trailing whitespace, or None if not found
    """
    # TODO: Implement this function.
    #
    # Step 1: Use re.search with the pattern r'<think>(.*?)</think>' and re.DOTALL
    #         to find the <think>...</think> block in response.
    #         If found, extract group(1) and strip whitespace; otherwise use None.
    #
    # Step 2: Use re.search with the pattern r'<answer>(.*?)</answer>' and re.DOTALL
    #         to find the <answer>...</answer> block in response.
    #         If found, extract group(1) and strip whitespace; otherwise use None.
    #
    # Step 3: Return {"thinking": <result from step 1>, "answer": <result from step 2>}
    
    pattern_think = '<think>...</think>'
    pattern_answer = '<answer>...</answer>'
    think_block =re.search(pattern_think, response, re.DOTALL)
    answer_block =re.search(pattern_answer, response, re.DOTALL)
    think_block = think_block.group(1).strip() if think_block else None
    answer_block = answer_block.group(1).strip()  if answer_block else None
    return {"thinking": think_block, "answer": answer_block}


def compute_format_reward(response):
    """
    Compute the format reward for a reasoning model response.

    In DeepSeek R1 training, the format reward is 1.0 if the response contains both
    a <think>...</think> block and an <answer>...</answer> block, with the think block
    appearing before the answer block. Otherwise it is 0.0.

    Args:
        response (str): the raw response string from the reasoning model

    Returns:
        float: 1.0 if both tags are present in the correct order, 0.0 otherwise
    """
    # TODO: Implement this function.
    #
    # Step 1: Use re.search with pattern r'<think>.*?</think>' and re.DOTALL
    #         to find the <think>...</think> block. Store the match object.
    #
    # Step 2: Use re.search with pattern r'<answer>.*?</answer>' and re.DOTALL
    #         to find the <answer>...</answer> block. Store the match object.
    #
    # Step 3: Check all three conditions:
    #         - think_match is not None (think block exists)
    #         - answer_match is not None (answer block exists)
    #         - think_match.end() <= answer_match.start() (think ends before answer begins)
    #         If all three are true, return 1.0. Otherwise return 0.0.

    pattern_think =r'<think>(.*?)</think>'
    pattern_answer = r'<answer>(.*?)</answer>'
    think_block =re.search(pattern_think, response, re.DOTALL)
    answer_block =re.search(pattern_answer, response, re.DOTALL)
    if think_block and  answer_block and (think_block.end() <= answer_block.start()):
        return 1.0
    return 0.0



def verify_math_answer(model_response, ground_truth):
    """
    Verify whether a reasoning model's answer matches the ground truth.

    This implements the "verifiable correctness reward" used in DeepSeek R1 training.
    For math problems, answers can be compared programmatically: first try numeric
    comparison (to handle "5" vs "5.0"), then fall back to case-insensitive string
    comparison (for text answers like "Paris").

    Args:
        model_response (str): the full response string from the model, which may
                              contain <think>...</think> and <answer>...</answer> tags
        ground_truth (str):   the expected correct answer

    Returns:
        bool: True if the extracted answer matches the ground truth, False otherwise
              (including when no <answer> tag is found)
    """
    # TODO: Implement this function.
    #
    # Step 1: Use re.search with pattern r'<answer>(.*?)</answer>' and re.DOTALL
    #         to find the <answer>...</answer> block in model_response.
    #         If not found, return False immediately.
    #
    # Step 2: Extract the matched content with .group(1) and call .strip() on it.
    #         Also call .strip() on ground_truth. Store these as 'extracted' and 'gt'.
    #
    # Step 3: Try numeric comparison first:
    #         - Wrap in a try/except ValueError block
    #         - Attempt float(extracted) and float(gt)
    #         - If both succeed, return abs(float(extracted) - float(gt)) < 1e-6
    #
    # Step 4: If the ValueError is raised (not numbers), fall back to string comparison:
    #         - Return extracted.lower() == gt.lower()
    pattern_think =r'<think>(.*?)</think>'
    pattern_answer = r'<answer>(.*?)</answer>'
    think_block =re.search(pattern_think, model_response, re.DOTALL)
    answer_block =re.search(pattern_answer, model_response, re.DOTALL)
    if not answer_block:
        return False
    extracted = answer_block.group(1).strip()
    gt = ground_truth.strip()
    try:
        return abs(float(extracted) - float(gt)) < 1e-6
    except ValueError:
        return extracted.lower() == gt.lower()
    


def compute_grpo_advantages(rewards):
    """
    Compute GRPO group-relative advantages for a list of rewards.

    In GRPO (Group Relative Policy Optimization), instead of training a separate
    value function (as PPO does), the advantage for each completion is computed
    relative to the other completions in the same group. For a group of G
    completions with rewards [r_1, ..., r_G], the advantage of completion i is:

        A_i = (r_i - mean(rewards)) / std(rewards)

    where std is the POPULATION standard deviation:

        std = sqrt( sum((r - mean)^2 for r in rewards) / n )

    Args:
        rewards (list of float): reward values for each completion in the group.
                                 Must be non-empty.

    Returns:
        list of float: advantage for each completion, same length as rewards.
                       Returns all 0.0s if std == 0.0 (all rewards identical).

    Raises:
        ValueError: if rewards is empty
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate input.
    #   - If rewards is empty, raise ValueError("rewards list cannot be empty")
    #
    # Step 2: Compute the mean.
    #   - mean = sum(rewards) / len(rewards)
    #
    # Step 3: Compute the population variance and std.
    #   - variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    #   - If variance == 0.0, return [0.0] * len(rewards)
    #   - std = variance ** 0.5
    #
    # Step 4: Normalize each reward.
    #   - return [(r - mean) / std for r in rewards]
    if rewards ==[]:
        raise ValueError("rewards list cannot be empty")
    mean = sum(rewards) / len(rewards)
    variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    if variance == 0.0:
        return [0.0] * len(rewards)
    std = variance ** 0.5
    if std == 0:
        raise ValueError("all rewards are identical — cannot compute advantages")
    return [(r - mean) / std for r in rewards]


def compose_rewards(format_reward, correctness_reward, format_weight=0.1, correctness_weight=1.0):
    """
    Compose a total RL reward from format and correctness components.

    DeepSeek R1 and similar reasoning models use multiple reward signals during
    training. The format reward measures whether the model's output follows the
    required structure (e.g., using <think> and <answer> tags). The correctness
    reward measures whether the final answer is right. These are combined with
    weights to produce a single scalar reward signal for the RL update.

    The format_weight is kept small (default 0.1) so that format compliance does
    not overwhelm the correctness signal during training.

    Args:
        format_reward (float): reward for formatting compliance. Must be in [0.0, 1.0].
        correctness_reward (float): reward for answer correctness. Must be in [0.0, 1.0].
        format_weight (float): weight for the format reward. Must be >= 0. Default: 0.1.
        correctness_weight (float): weight for the correctness reward. Must be >= 0. Default: 1.0.

    Returns:
        float: format_weight * format_reward + correctness_weight * correctness_reward

    Raises:
        ValueError: if format_weight < 0 or correctness_weight < 0
        ValueError: if format_reward not in [0.0, 1.0] or correctness_reward not in [0.0, 1.0]
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate weights.
    #   - If format_weight < 0, raise ValueError("format_weight must be >= 0, got <format_weight>")
    #   - If correctness_weight < 0, raise ValueError("correctness_weight must be >= 0, got <correctness_weight>")
    #
    # Step 2: Validate reward values.
    #   - If not (0.0 <= format_reward <= 1.0), raise ValueError("format_reward must be in [0, 1], got <format_reward>")
    #   - If not (0.0 <= correctness_reward <= 1.0), raise ValueError("correctness_reward must be in [0, 1], got <correctness_reward>")
    #
    # Step 3: Compute and return the weighted sum.
    #   - return format_weight * format_reward + correctness_weight * correctness_reward    
    if format_weight < 0:
        raise ValueError("format_weight must be >= 0, got <format_weight>")
    if correctness_weight < 0:
        raise ValueError("correctness_weight must be >= 0, got <correctness_weight>")
    if  not (0.0 <= format_reward <= 1.0):
        raise ValueError("format_reward must be in [0, 1], got <format_reward>")
    if  not (0.0 <= correctness_reward <= 1.0):
        raise ValueError("correctness_reward must be in [0, 1], got <correctness_reward>")
    return format_weight * format_reward + correctness_weight * correctness_reward    


def analyze_length_bias(completions):
    """
    Analyze the length bias in the original GRPO loss formulation.

    In the original GRPO loss, each completion's contribution to the gradient
    is normalized by its token count (1/|O_i|). The per-token weight for
    completion i is therefore: advantage_i / length_i.

    This creates a length bias: tokens in short completions contribute MORE
    to the gradient than tokens in long completions. When the advantage is
    negative (wrong answer), short wrong completions are penalized more per
    token than long wrong ones — perversely incentivizing longer outputs.

    The fix (Dr. GRPO / DAPO) equalizes contribution across the group:
    each completion contributes advantage_i / G to the gradient, regardless
    of its length, where G is the total number of completions.

    Args:
        completions (list of dict): each dict has keys:
            - "advantage" (float): the GRPO advantage for this completion
            - "length" (int): the token count of this completion (must be > 0)

    Returns:
        dict with two keys:
            - "original_weights" (list of float): advantage_i / length_i for each completion
            - "equal_weights" (list of float): advantage_i / G for each completion

    Raises:
        ValueError: if completions is empty
        ValueError: if any completion has length <= 0
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If completions is empty, raise ValueError("completions list cannot be empty")
    #   - For each completion c, if c['length'] <= 0, raise ValueError(
    #       f"completion length must be > 0, got {c['length']}")
    #
    # Step 2: Compute G (the group size).
    #   - G = len(completions)
    #
    # Step 3: Compute per-token weights.
    #   - original_weights = [c['advantage'] / c['length'] for c in completions]
    #   - equal_weights = [c['advantage'] / G for c in completions]
    #
    # Step 4: Return the result dict.
    #   - return {'original_weights': original_weights, 'equal_weights': equal_weights}
    if completions == []:
        raise ValueError("completions list cannot be empty")
    for completetion in completions:
        if completetion['length'] <= 0:
            raise ValueError(f"completion length must be > 0, got {completetion['length']}")
    G = len(completions)
    original_weights = [c['advantage'] / c['length'] for c in completions]
    equal_weights = [c['advantage'] / G for c in completions]
    return {'original_weights': original_weights, 'equal_weights': equal_weights}


def rejection_sample(completions, verifier_fn, max_samples=None):
    """
    Filter completions using a verifier function, optionally capping the result.

    In the DeepSeek R1 pipeline, rejection sampling is used to collect high-quality
    SFT training data: a reasoning model generates many completions for each prompt,
    and only those that pass a verifier (e.g., correct final answer, passing unit tests)
    are kept. This curated dataset is then used for the second SFT stage.

    Args:
        completions (list of str): candidate completions generated by the model
        verifier_fn (callable): function that takes a completion string and returns
                                True if the completion is correct, False otherwise
        max_samples (int or None): if not None, return at most this many correct
                                   completions (first-come-first-served)

    Returns:
        list of str: correct completions, up to max_samples if specified

    Raises:
        ValueError: if completions is empty
        ValueError: if no completions pass the verifier
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If completions is empty (falsy), raise ValueError("completions list cannot be empty")
    #
    # Step 2: Filter to keep only passing completions.
    #   - Build a list: correct = [c for c in completions if verifier_fn(c)]
    #
    # Step 3: Check that at least one completion passed.
    #   - If correct is empty, raise ValueError("no completions passed the verifier — cannot rejection sample")
    #
    # Step 4: Apply max_samples cap (if provided).
    #   - If max_samples is not None, slice: correct = correct[:max_samples]
    #   - Note: if max_samples > len(correct), slicing returns all elements (no extra handling needed)
    #
    # Step 5: Return the filtered list.

    if completions == []:
        raise ValueError("completions list cannot be empty")
    correct = [c for c in completions if verifier_fn(c)]
    if correct == []:
         raise ValueError("no completions passed the verifier — cannot rejection sample")
    if max_samples != None:
        correct = correct[:max_samples]
    return correct



def compute_grpo_token_objective(old_log_prob, new_log_prob, ref_log_prob, advantage,
                                  epsilon=0.2, beta=0.01):
    """
    Compute the GRPO per-token objective including the KL divergence penalty.

    GRPO (Group Relative Policy Optimization) is the RL algorithm used in DeepSeek R1.
    For each token, its contribution to the loss has two parts:
    (1) a clipped surrogate objective (like PPO) that reinforces tokens with positive
        group-relative advantages while preventing excessively large policy updates,
    (2) a KL penalty that keeps the new policy close to the reference SFT model.

    Args:
        old_log_prob (float): log probability of this token under the old (frozen) policy; must be <= 0
        new_log_prob (float): log probability of this token under the current policy; must be <= 0
        ref_log_prob (float): log probability of this token under the reference SFT model; must be <= 0
        advantage (float): group-relative advantage for the completion containing this token
        epsilon (float): clipping range for the ratio (default 0.2); must be > 0
        beta (float): KL penalty coefficient (default 0.01); must be >= 0

    Returns:
        float: per-token GRPO objective value

    Raises:
        ValueError: if old_log_prob, new_log_prob, or ref_log_prob > 0
        ValueError: if epsilon <= 0 or beta < 0
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If old_log_prob > 0: raise ValueError(f"old_log_prob must be <= 0, got {old_log_prob}")
    #   - If new_log_prob > 0: raise ValueError(f"new_log_prob must be <= 0, got {new_log_prob}")
    #   - If ref_log_prob > 0: raise ValueError(f"ref_log_prob must be <= 0, got {ref_log_prob}")
    #   - If epsilon <= 0:     raise ValueError(f"epsilon must be > 0, got {epsilon}")
    #   - If beta < 0:         raise ValueError(f"beta must be >= 0, got {beta}")
    #
    # Step 2: Compute the probability ratio.
    #   ratio = math.exp(new_log_prob - old_log_prob)
    #   (This is the ratio of new policy probability to old policy probability for this token.)
    #
    # Step 3: Clip the ratio to [1 - epsilon, 1 + epsilon].
    #   clipped_ratio = max(1.0 - epsilon, min(1.0 + epsilon, ratio))
    #
    # Step 4: Compute the clipped surrogate objective.
    #   clip_objective = min(ratio * advantage, clipped_ratio * advantage)
    #   (This is the PPO-style clipped objective — it prevents excessively large updates.)
    #
    # Step 5: Compute the per-token KL penalty (token-level approximation).
    #   kl_penalty = new_log_prob - ref_log_prob
    #   (Positive when new policy assigns higher probability than the reference SFT model.)
    #
    # Step 6: Return the combined objective.
    #   return clip_objective - beta * kl_penalty
    if old_log_prob > 0:
        raise ValueError(f"old_log_prob must be <= 0, got {old_log_prob}")
    if new_log_prob > 0:
        raise ValueError(f"old_log_prob must be <= 0, got {new_log_prob}")
    if ref_log_prob > 0:
        raise ValueError(f"old_log_prob must be <= 0, got {ref_log_prob}")
    if epsilon <= 0:  
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    ratio = math.exp(new_log_prob - old_log_prob)
    clipped_ratio = max(1.0 - epsilon, min(1.0 + epsilon, ratio))
    clip_objective = min(ratio * advantage, clipped_ratio * advantage)
    kl_penalty = new_log_prob - ref_log_prob
    return clip_objective - beta * kl_penalty



def run_grpo_pipeline(completions, epsilon=0.2, beta=0.01):
    """
    Run the full GRPO training pipeline on a batch of completions.

    This is how DeepSeek R1-Zero is trained: for each prompt, multiple completions are
    generated and evaluated with a verifiable reward. Their rewards are normalized into
    group-relative advantages, and the GRPO objective is computed per token, then averaged
    over tokens (per completion) and over completions (batch objective).

    Args:
        completions (list of dict): each dict has:
            - "old_log_probs": list of float (per-token log probs under old policy)
            - "new_log_probs": list of float (per-token log probs under current policy)
            - "ref_log_probs": list of float (per-token log probs under reference SFT model)
            - "reward": float (scalar reward for this completion)
        epsilon (float): clipping range for GRPO token objective (default 0.2)
        beta (float): KL penalty coefficient (default 0.01)

    Returns:
        dict with keys:
            - "advantages": list of float, group-relative advantages (one per completion)
            - "objective": float, mean over completions of mean per-token objectives

    Raises:
        ValueError: if completions is empty
        ValueError: if any completion has mismatched log_prob list lengths
    """
    # TODO: Implement this function.
    #
    # Step 1: Validate inputs.
    #   - If completions is empty, raise ValueError("completions list cannot be empty")
    #   - For each completion, check that len(old_log_probs) == len(new_log_probs) == len(ref_log_probs)
    #     If not, raise ValueError(f"completion {i} has mismatched log_prob lengths: ...")
    #
    # Step 2: Extract rewards.
    #   rewards = [c["reward"] for c in completions]
    #
    # Step 3: Compute group-relative advantages.
    #   advantages = compute_grpo_advantages(rewards)
    #
    # Step 4: For each completion i:
    #   a. Get advantage = advantages[i]
    #   b. Get old_log_probs, new_log_probs, ref_log_probs from completions[i]
    #   c. Compute per-token objectives:
    #      token_objs = [compute_grpo_token_objective(old[t], new[t], ref[t], advantage, epsilon, beta)
    #                    for t in range(len(old_log_probs))]
    #   d. Compute mean token objective for this completion:
    #      completion_obj = sum(token_objs) / len(token_objs)
    #
    # Step 5: Compute the overall objective (mean over completions).
    #   objective = sum(completion_objectives) / len(completion_objectives)
    #
    # Step 6: Return the result dict.
    #   return {"advantages": advantages, "objective": objective}
    if completions == []:
        raise ValueError("completions list cannot be empty")
    for i, completition in enumerate(completions):
        old_log_probs = completition['old_log_probs']
        new_log_probs = completition['new_log_probs']
        ref_log_probs = completition['ref_log_probs']
        if not (len(old_log_probs) == len(new_log_probs) == len(ref_log_probs)):
            raise ValueError(f"completion {i} has mismatched log_prob lengths: ...")
    rewards = [c["reward"] for c in completions]
    advantages = compute_grpo_advantages(rewards)
    completion_obj_lst = []
    for i, completition in enumerate(completions):
        advantage_i = advantages[i]
        old_log_probs = completition['old_log_probs']
        new_log_probs = completition['new_log_probs']
        ref_log_probs = completition['ref_log_probs']
        token_objs = [compute_grpo_token_objective(old_log_probs[t], new_log_probs[t], ref_log_probs[t], advantage_i, epsilon, beta) for t in range(len(old_log_probs))]
        completion_obj = sum(token_objs) / len(token_objs)
        completion_obj_lst.append(completion_obj)
    objective = sum(completion_obj_lst) / len(completion_obj_lst)
    return {"advantages": advantages, "objective": objective}




def call_finetuned_model(prompt: str, api_key: str) -> str:
    """
    Call your Tinker-hosted fine-tuned reasoning model with the given prompt.

    The autograder loads your API key from tinker_key.txt and passes it as
    the api_key parameter. You must hardcode your Tinker base_url and model path below.

    Args:
        prompt (str): the user's question or task
        api_key (str): your Tinker API key (provided by the autograder)

    Returns:
        str: the model's full response text

    Raises:
        ValueError: if prompt is empty or None
    """
    # TODO Step 1: Input Validation
    # Check if prompt is None or empty, raise ValueError if so


    # TODO Step 2: Configuration
    # Set TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    # Set TINKER_MODEL = your model path from training (tinker://...)


    # TODO Step 3: Prepare HTTP Request
    # Set headers with Authorization (Bearer token) and Content-Type
    # Create request body with model, messages, temperature, max_tokens


    # TODO Step 4: Make API Call
    # POST to {TINKER_BASE_URL}/chat/completions
    # Use requests.post() with headers and json parameters


    # TODO Step 5: Extract and Return Response
    # Parse JSON response and return the message content


    if prompt == None or prompt == "":
        raise ValueError("invalid prompt")
    TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    TINKER_MODEL =
