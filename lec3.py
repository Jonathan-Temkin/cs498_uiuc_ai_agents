import requests
from key import api_key
import re
from vars import *

def wrap_input_with_xml(content, tag_name):
    """
    Wrap content with XML tags to create clear boundaries in prompts.

    Args:
        content (str): The content to wrap
        tag_name (str): The XML tag name to use

    Returns:
        str: The content wrapped in XML tags with newlines

    Example:
        wrap_input_with_xml("Hello", "greeting") returns:
        <greeting>
        Hello
        </greeting>
    """
    return ('<' + tag_name + '>' + "\n" + content + "\n" + '</' + tag_name + '>')

def extract_xml_content(text, tag_name):
    """
    Extract content between XML tags from text.

    Args:
        text (str): The full text containing XML tags
        tag_name (str): The XML tag name to search for

    Returns:
        str or None: The extracted content (stripped) or None if not found

    Example:
        extract_xml_content("<answer>42</answer>", "answer") returns "42"
        extract_xml_content("no tags here", "answer") returns None
    """

    pattern = rf"<{tag_name}>\s*([\s\S]*?)\s*</{tag_name}>"
    match = re.search(pattern, text)
    try:
        return match.group(1)
    except:
        return None

sample = wrap_input_with_xml("jon", "name")
print(sample)
content = extract_xml_content(sample, "name")
print(content)


def call_with_stop_sequence(prompt, prefill, stop_seq, api_key):

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": """good boy good agent""",
        #"messages": conversation_history
        "temperature": 0, # smaller # is more deterministic
        "messages": [ {"role":"user", "content": prompt},
                     {"role":"assistant", "content": prefill} ],
        "stop_sequences" : [stop_seq]
    }
    response = requests.post(url, json=body, headers=headers)
    print(response.json())
    msg = response.json()["content"][0]["text"]
    msg_prefilled = prefill + msg
    print(msg_prefilled)
    return msg


# prompt = "tell a dumb joke thats not funny"
# prefill = "it goes like this:"
# stop_seq = 'answer'
# call_with_stop_sequence(prompt, prefill, stop_seq, api_key)


def create_translation_prefill(source_text, source_lang, target_lang):

    # TODO: Create a JSON prefill with source_language, target_language, original, and partial translated fiel
    pattern = "\{.*?\}"
    match = re.search(pattern, text)
    try:
        return match.group(1)
    except:
        return None
    
def parse_review_xml(text):
    summary  = extract_xml_content(text, "summary")
    if summary == None or summary == "": return None
    tag_name = "issue"
    pattern = rf"<{tag_name}>\s*([\s\S]*?)\s*</{tag_name}>"
    issues_lst = []
    issues = re.finditer(pattern, text)
    for issue in issues:
        severity = extract_xml_content(text, "severity")
        description = extract_xml_content(text, "description")
        issues_lst.append({"severity": severity, "description": description })
    result = {
        "summary" : summary, 
        "issues" : issues_lst
    }
    print(result)
    return result


#parse_review_xml(sample_xml)



def analyze_sentiment(text, api_key):
    """
    Analyze sentiment using a complete structured I/O pipeline.

    This function should:
    1. Create a prompt with the text wrapped in <text> tags
    2. Use prefilling with <sentiment> tag
    3. Use </sentiment> as a stop sequence
    4. Parse and return the sentiment

    Args:
        text (str): The text to analyze
        api_key (str): Your API key

    Returns:
        str: The sentiment ("positive", "negative", or "neutral")

    Example:
        analyze_sentiment("I love this!", api_key) -> "positive"
    """
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 1024,
        "system": """good boy good agent. analyze sentiment""",
        #"messages": conversation_history
        "temperature": 0, # smaller # is more deterministic
        "messages": [ {"role":"user", "content": text},
                     {"role":"assistant", "content": "<sentiment>"} ],
        "stop_sequences" : ["</sentiment>"]
    }

    response = requests.post(url, json=body, headers=headers)
    print(response.json())
    msg = response.json()["content"][0]["text"]
    #msg_prefilled = prefill + msg
    msg_parsed = extract_xml_content(msg, "sentiment")
    print(msg)
    return msg

text = "i hate you you are the worst"
analyze_sentiment(text, api_key)