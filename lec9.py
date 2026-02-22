from tools import *
from claude_api import *

def classify_task(description):
    """
    Classify whether a task should be implemented as a simple program,
    workflow, or agent.

    Args:
        description (str): Description of the task

    Returns:
        str: One of "simple_program", "workflow", or "agent"
    """

    simple_program_keywords = ['calculate', 'sort']
    workflow_keywords = ['generate', 'translate', 'summarize', 'create']
    agent_keywords = ['analyze', 'debug', 'solve', 'investigate', 'research']

    for word in description.split():
        word = word.lower()
        if word in simple_program_keywords:
            return "simple_program"
        if word in workflow_keywords:
            return "workflow"
        if word in agent_keywords:
            return "agent"
    return "simple_program"

def create_file_search_tool():
    """
    Create a well-documented tool schema for a file search tool.

    Returns:
        dict: Tool schema with name, description, and input_schema
    """
    name = "file_search"
    description = "Search for files containing specific text..."
    parameters = [{"name": "query", "type": "string", "description": "query for db to search", "required": True},{"name": "file_types", "type": "array", "description": "query for db to search", "required": False},{"name": "max_results", "type": "integer", "description": "query for db to search", "required": False}]

    return generate_tool_schema(name, description, parameters)




# Handler functions (already provided)
def handle_technical_support(query):
    return f"For technical support: {query}"


def handle_billing(query):
    return f"For billing: {query}"


def handle_general_info(query):
    return f"For general info: {query}"



def route_query(user_query, api_key):
    """
    Classify the user query and route it to the appropriate handler.

    Args:
        user_query (str): The user's question or request
        api_key (str): Anthropic API key

    Returns:
        str: The result from the appropriate handler
    """
    messages = user_query
    system_prompt = "classify the user query as one of the following: technical_support, billing, general_info. return answer only"
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                        tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                        temperature = None)
    response = response['content'][0]['text']
    if response == "technical_support":
        return handle_technical_support(messages)
    if response == "billing":
        return handle_billing(messages)
    return handle_general_info(messages)

user_query = "How do I reset my password?"
#route_query(user_query, api_key)


# result = parallel_classification(
#     "This movie was amazing! Best film of the year.",
#     ["positive", "negative", "neutral"],
#     5,  # 5 voters
#     api_key
# )

def parallel_classification(text, categories, num_voters, api_key):
    """
    Classify text using multiple parallel API calls and majority voting.

    Args:
        text (str): Text to classify
        categories (list): List of possible categories
        num_voters (int): Number of parallel classification attempts
        api_key (str): Anthropic API key

    Returns:
        dict: {"category": winning_category, "votes": {category: count, ...}}
    """

    counter = dict.fromkeys(categories, 0)
    winning_cat = None
    winning_vote_count = 0 
    system_prompt = f"given the categories provided, select one that best fits the user prompt. return only the name of the winning category. RETURN THE CATEGORY NAME ONLY. CATEGORIES: {categories}"
    messages = text
    for _ in range(num_voters):
        response = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
        vote = response['content'][0]['text']
        try:
            counter[vote] = counter[vote] +1
        except:
            counter[vote] = 1
    for category in categories:
        cat_vote_count = counter[category]
        if cat_vote_count > winning_vote_count:
            winning_cat = category
    return {"category": winning_cat, "votes": counter}
        



# text = "bill gates went to epsteins island to do illegal stuff"
# categories = ["consparicies", "intelligent", "retard"]
# num_voters = 5
# print(parallel_classification(text, categories, num_voters, api_key))



def draft_and_refine(task_description, max_iterations, api_key):
    """
    Generate and iteratively refine a response using the evaluator-optimizer pattern.

    Args:
        task_description (str): Description of what to generate
        max_iterations (int): Maximum number of refinement iterations
        api_key (str): Anthropic API key

    Returns:
        dict: {"final_text": refined_text, "iterations": num_iterations}
    """
    draft_prompt = "create a draft based on the user request"
    review_prompt = "RETURN AN INTEGER SCORE 1-10. SCORE ONLY NO ADDITIONAL TEXT. score the text out of 10. Return ONLY an integer score. Ensure the response is recongized in python as an integer"
    draft = task_description 
    num_iterations = 0 
    for iteration in range(max_iterations):
        draft = claude_api_call(draft, api_key, return_type='json', system_prompt = draft_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)['content'][0]['text']
        score = claude_api_call(draft, api_key, return_type='json', system_prompt = review_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)['content'][0]['text']
        print(score)
        if int(score) > 8:
            return draft
        num_iterations += 1
    return {"final_text": draft, "iterations": num_iterations} 


# task_description = "write a short 100 word story about bill gates and jeffery epstein vacationing on an island"
# print(draft_and_refine(task_description, 5, api_key))


def score_tool_definition(tool_schema):
    """
    Score a tool definition based on quality best practices.

    Args:
        tool_schema (dict): Tool definition with name, description, input_schema

    Returns:
        dict: {"score": int (0-100), "issues": [list of issues]}
    """
    # TODO: Implement quality scoring
    # Start with 100 points
    # Deduct points for problems:
    # - Missing or short description (< 20 chars): -30 points
    # - Vague name (contains "tool", "function", "use"): -20 points
    # - Missing parameter descriptions: -15 points each
    # - Short parameter descriptions (< 10 chars): -10 points each
    # - Missing required field: -20 points
    # Track issues in a list

    score = 100
    issues_lst = []
    tool_name = tool_schema['name']
    tool_desc = tool_schema['description']
    if len(tool_desc) < 20:
        score -= 30
        issues_lst.append("description too short")
    if "tool" or "function" or "use" in tool_name:
        score -= 20
        issues_lst.append("vague tool name")
    params = tool_schema['input_schema']['properties']
    for param in params:
        try:
            param_desc = params[param]['description']
            if len(param_desc) < 10:
                score -= 10
                issues_lst.append("param description too short")
        except:
            issues_lst.append("missing param description")
    return {"score": score, "issues": issues_lst}


#print(score_tool_definition(define_search_tool()))




def sop_to_system_prompt(sop_text, api_key):
    """
    Convert a standard operating procedure into a system prompt using Claude.

    Args:
        sop_text (str): The SOP document text
        api_key (str): Anthropic API key

    Returns:
        str: Generated system prompt
    """
    # TODO: Implement SOP to system prompt conversion
    # 1. Call Claude API with instructions to convert the SOP
    # 2. Ask it to:
    #    - Define a clear agent role
    #    - Convert steps into actionable instructions
    #    - Preserve safety requirements
    #    - Use imperative language
    # 3. Return the generated system prompt
    system_prompt = "Return a claude API system prompt based on the provided SOP. specify the agent's role, how it should act, and what it should do. RETURN ONLY THE SYSTEM PROMPT"
    response = claude_api_call(sop_text, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    return response['content'][0]['text']

sop = """
Customer Support SOP:
1. Greet the customer politely
2. Identify the issue
3. Check if issue is in knowledge base
4. NEVER share internal system details
5. Escalate to human if unable to resolve
"""

# print(sop_to_system_prompt(sop, api_key))

def validate_user_input(user_input, max_len=5000):
    """
    Validate user input for security issues.

    Args:
        user_input (str): The user's input text

    Returns:
        dict: {"is_safe": bool, "issues": [list of issues]}
    """
    is_safe = True
    issues_lst = []
    user_input = user_input.lower()
    if "ignore previous instructions" in user_input:
        is_safe = False
        issues_lst.append("Detected prompt injection pattern: ignore previous instructions")
    if "hacker" or "pirate" in user_input:
        is_safe = False
        issues_lst.append("Detected prompt injection pattern:")
    if len(user_input) > max_len:
        is_safe = False
        issues_lst.append("Input exceeds maximum length of 5000 characters")        
    return {"is_safe": is_safe, "issues": issues_lst}

# result = validate_user_input("What is the weather today?")
# print(result)


def should_split_agents(agent_description):
    """
    Determine if a single agent should be split into multiple specialized agents.

    Args:
        agent_description (str): Description of the agent's responsibilities

    Returns:
        dict: {
            "should_split": bool,
            "reason": str,
            "suggested_agents": [list of agent names] or []
        }
    """
    # TODO: Implement splitting decision logic
    # Check for:
    # 1. Multiple distinct responsibilities (5+ different tasks/domains)
    # 2. Keywords indicating complexity: "handles everything", "all types of"
    # 3. If splitting recommended, suggest specialized agent names
    # If not splitting, return empty list for suggested_agents
    agent_responsibilities_lst = agent_description.split(',')
    should_split = False
    reason = "Single focused responsibility - no need to split"
    suggested_agents = []
    if len(agent_responsibilities_lst) == 1:
        return  {
                "should_split": should_split,
                "reason": reason,
                "suggested_agents": []
                }
    for responsibility in agent_responsibilities_lst:
        agent_name = responsibility + '_agent'
        suggested_agents.append(agent_name)
    reason = "multiple responsibilities"
    should_split = True
    return {
        "should_split": should_split,
        "reason": reason,
        "suggested_agents": []
    }



# Tool function provided for you
def search_knowledge_base(query):
    """Simulate knowledge base search"""
    return f"Knowledge base result for '{query}': Product information found."

def tool_exec():
    return search_knowledge_base

def process_customer_request(request, api_key):
    """
    Process a customer request with validation, routing, and error handling.

    Args:
        request (str): Customer's request text
        api_key (str): Anthropic API key

    Returns:
        dict: {
            "status": "success" or "error",
            "result": str,
            "metadata": {
                "category": str or None,
                "used_tool": bool
            }
        }
    """
    prompt_validation = validate_user_input(request, max_len=5000)
    
    sort_system_prompt = "sort the request as one of these. return result only: product_question, account_issue, general"
    sort_result = claude_api_call(request, api_key, return_type='json', system_prompt = sort_system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    sort_result = sort_result['content'][0]['text']
    tool_params = [{"name": "query", "type": "string", "description": "search query", "required": True}]
    knowledge_tool_schema = generate_tool_schema("knowledge_base", "searches the knowledge base", tool_params)
    if sort_result == "product_question":
        system_prompt = "use provided tool to get error."
        response = react_agent(request, api_key, knowledge_tool_schema, tool_exec, max_rounds=20, return_type = None, system_prompt=None)
    elif sort_result == "account_issue":
        response = claude_api_call(request, api_key, return_type='json', system_prompt = "help with account security", 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    else:
        response = claude_api_call(request, api_key, return_type='json', system_prompt = None,
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
        response = response['content'][0]['text']
    status =  "error" if "error" in response else "success"
    status = status if prompt_validation["is_safe"] == True else 'error'
    result = {
        "status": status,
        "result": response,
        "metadata": {"category": sort_result, "used_tool": True if sort_result == "account_issue" else False}
    }
    return result
request = "find info on jeffery epstein from db"
print(process_customer_request(request, api_key))