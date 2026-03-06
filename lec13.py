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
    think_block = re.match(pattern_think, response).group(1).strip()
    answer_block = re.match(pattern_answer, response).group(1).strip()
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
    #         I