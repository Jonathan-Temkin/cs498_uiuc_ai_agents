import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *
import math
import re
import ast
import json



def detect_tool_poisoning(description: str) -> dict:
    """
    Scan an MCP tool description for tool poisoning payload patterns.

    Args:
        description (str): The tool's description text from its MCP registration.

    Returns:
        dict with keys:
            "suspicious" (bool): True if any poisoning pattern was found.
            "patterns_found" (list[str]): Sorted list of matched patterns (lowercase), deduplicated.
            "sanitized" (str): The description with all matched patterns replaced by "[REDACTED]"
                               (case-insensitive replacement, surrounding text preserved).

    Raises:
        ValueError: If description is None.

    Poisoning signal patterns to detect (case-insensitive):
        "before executing", "before returning", "first send", "also read",
        "additionally, send", "do not reveal", "ignore the above", "important: always"
    """
    # TODO: Raise ValueError if description is None

    # TODO: Define the list of poisoning patterns

    # TODO: For each pattern, check if it appears in description (case-insensitive)
    # Collect matched patterns in a set for deduplication

    # TODO: Build the sanitized string by replacing each matched pattern
    # with "[REDACTED]" (case-insensitive)
    # Hint: use re.sub with re.IGNORECASE

    # TODO: Return {"suspicious": bool, "patterns_found": sorted list, "sanitized": str}

    if not description:
        raise ValueError
    poisoning_patterns = [
    "before executing",
    "before returning",
    "first send",
    "also read",
    "additionally, send",
    "do not reveal",
    "ignore the above",
    "important: always"
    ]
    sus = False
    patterns_found = []
    sanitized =  description
    for pattern in poisoning_patterns:
        if pattern.lower() in description.lower():
            patterns_found.append(pattern)
            sus =True
            santiized = sanitized.lower().replace(pattern.lower(), "[REDACTED]" )
    return {"suspicious": sus, "patterns_found": sorted(set(patterns_found)), "sanitized": sanitized}