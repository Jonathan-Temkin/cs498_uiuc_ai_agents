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
    
    if not content:
        raise ValueError()
