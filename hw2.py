import string
from collections import Counter
from collections import defaultdict
import copy 
from claude_api import *
import re

def parse_mcp_uri(uri):
    """
    Parse an MCP resource URI into its components.

    Args:
        uri (str): The MCP resource URI in format "resource_type://path"

    Returns:
        dict: Dictionary with keys "resource_type" and "path"

    Raises:
        ValueError: If URI format is invalid (doesn't contain ://)

    Examples:
        parse_mcp_uri("calendar://events/2024")
        # Returns: {"resource_type": "calendar", "path": "events/2024"}

        parse_mcp_uri("invalid-uri")
        # Raises: ValueError("Invalid URI format")
    """

    if "://" not in uri:
        raise ValueError("Invalid URI format")
    uri_split = uri.split("://" )
    resource_type = uri_split[0]
    path = uri_split[1]
    return  {"resource_type": resource_type, "path": path}


def calculate_bm25_tf(term, document, avg_doc_len, k1=1.5, b=0.75):
    """
    Calculate the BM25 term frequency score for a term in a document.

    Args:
        term (str): The search term
        document (str): The document text
        avg_doc_len (float): Average document length in the collection
        k1 (float): Term frequency saturation parameter (default 1.5)
        b (float): Length normalization parameter (default 0.75)

    Returns:
        float: The BM25 term frequency score

    Example:
        document = "Python is great Python rocks"
        term = "python"
        avg_doc_len = 5
        calculate_bm25_tf(term, document, avg_doc_len)
        # Returns approximately 1.04
    """
    # TODO: Implement this function
    # Formula: TF = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    doc = document.lower().split()
    term = term.lower()
    doc_len = len(doc)
    doc_counts = Counter(doc)
    term_freq = doc_counts[term]
    print(term_freq)
    tf_score =  (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    return tf_score


document = "Python is great Python rocks"
term = "python"
avg_doc_len = 5
# print(calculate_bm25_tf(term, document, avg_doc_len, k1=1.5, b=0.75))


def parse_react_output(text):
    """
    Parse ReAct-formatted agent output to extract components.

    Args:
        text (str): The ReAct-formatted text

    Returns:
        dict: Dictionary with keys "thought", "action", "action_input"
              Missing fields should have None as value

    Example:
        text = '''Thought: I need to search
    Action: search
    Action Input: Python'''
        parse_react_output(text)
        # Returns: {"thought": "I need to search", "action": "search", "action_input": "Python"}
    """

    thought = re.search(r"Thought:\s*(.*)", text) 
    action = re.search(r"Action:\s*(.*)", text)
    input = re.search(r"Action Input:\s*(.*)", text)
    result = {
    "thought": thought.group(1).strip() if thought else None ,
    "action": action.group(1).strip() if action else None,
    "action_input": input.group(1).strip() if input else None
    }
    return result 


# text = """Thought: I need to search for information
# Action: search
# Action Input: Python programming"""
# print(parse_react_output(text))


class MCPToolRegistry:
    """
    A simplified MCP tool registry for managing tool definitions.
    """

    def __init__(self):
        """Initialize the registry."""
        self._tools = {}

    def register_tool(self, name, description, schema):
        """
        Register a tool in the registry.

        Args:
            name (str): The tool name
            description (str): The tool description
            schema (dict): The tool's parameter schema
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "schema": schema
        }

    def get_tool(self, name):
        """
        Retrieve a tool's definition by name.

        Args:
            name (str): The tool name

        Returns:
            dict or None: Tool definition with keys "name", "description", "schema",
                          or None if not found
        """
        try:
            return self._tools[name]
        except:
            return None

    def list_tools(self):
        """
        Get a list of all registered tool names.

        Returns:
            list: Alphabetically sorted list of tool names
        """
        return sorted(self._tools.keys())

    def get_all_tools(self):
        """
        Get all registered tools.

        Returns:
            dict: Dictionary mapping tool names to their definitions
        """
        return self._tools



def chunk_document(text, chunk_size, overlap):
    """
    Split a document into overlapping chunks.

    Args:
        text (str): The document text
        chunk_size (int): Number of words per chunk
        overlap (int): Number of overlapping words between chunks

    Returns:
        list: List of chunk strings

    Example:
        text = "The quick brown fox jumps over the lazy dog"
        chunk_document(text, chunk_size=5, overlap=2)
        # Returns: ["The quick brown fox jumps", "fox jumps over the lazy", "the lazy dog"]
    """
    text = text.split()
    text_len = len(text)
    if text_len < chunk_size: return [" ".join(text)]
    chunks = []
    chunk_incremenet = chunk_size - overlap
    current_i = chunk_size
    start_i = 0
    # num_iterations = (((text_len - chunk_size) / chunk_incremenet) + 1)
    # num_iterations = num_iterations if num_iterations == int(num_iterations) else int(num_iterations) + 1
    start_i = 0 
    end_i = start_i+chunk_size
    while end_i < text_len + chunk_incremenet:
        print(start_i, end_i, text_len)
        chunk = text[start_i:end_i]
        start_i = end_i - overlap
        end_i += chunk_incremenet
        chunks.append(" ".join(chunk))
        if not chunk:
            break
    return chunks
    

    

# text = "The quick brown fox jumps over the lazy dog runs"
# chunk_size = 5
# overlap = 2
# print(chunk_document(text, chunk_size, overlap))


def rerank_documents(query, documents, api_key, top_k=3):
    """
    Re-rank documents using LLM-based relevance scoring.

    Args:
        query (str): The search query
        documents (list): List of document strings
        api_key (str): Your Anthropic API key
        top_k (int): Number of top documents to return (default 3)

    Returns:
        list: Top-k most relevant documents (strings)

    Example:
        query = "Python programming"
        documents = ["Python is great", "Java tutorial", "Python guide"]
        rerank_documents(query, documents, api_key, top_k=2)
        # Returns the 2 most relevant documents
    """
    # TODO: Implement this function
    # 1. For each document, call the API to get a relevance score
    # 2. Parse the numeric score from the response
    # 3. Sort documents by score (highest first)
    # 4. Return top-k documents
    scores = {}
    docs = []
    for i, doc in enumerate(documents):
        system_prompt = f"assign this document a relevance score 0-100 based on the query: {query}. YOU WILL RETURN ONLY THE INTEGER SCORE WITH NO TEXT. RESPONSE MUST BE CAST TO AN INT IN PYTHON."
        messages = doc
        response = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
        score = int(response['content'][0]['text'])
        scores[i] = score
    scores_sorted = sorted(scores, key=scores.get, reverse=True)
    for top_doc in range(top_k):
        doc_id = scores_sorted[top_doc]
        doc = documents[doc_id]
        docs.append(doc)
    return docs



# query = "Python programming"
# documents = ["Python is a language", "Java tutorial", "Python for beginners"]
# print(rerank_documents(query, documents, api_key, top_k=2))
# Returns the 2 most relevant documents based on LLM scoring


class MCPClient:
    """
    A simplified MCP client for discovering and executing server tools.
    """

    def __init__(self, server):
        """
        Initialize the client with a server reference.

        Args:
            server: The MCP server object
        """
        self._server = server

    def discover_tools(self):
        """
        Discover available tools from the server.

        Returns:
            list: List of tool names
        """
        return self._server.list_tools() 

    def get_tool_info(self, tool_name):
        """
        Get information about a specific tool.

        Args:
            tool_name (str): The name of the tool

        Returns:
            dict: Dictionary with "name" and "schema"
        """
        # TODO: Call server.get_tool_schema(tool_name) and return formatted dict
        schema = self._server.get_tool_schema(tool_name) 
        return {
            "name":tool_name,
            "schema":schema
            }

    def call_tool(self, tool_name, **params):
        """
        Execute a tool with the given parameters.

        Args:
            tool_name (str): The name of the tool to execute
            **params: Keyword arguments to pass to the tool

        Returns:
            The result from tool execution
        """
        # TODO: Call server.execute_tool(tool_name, params) and return the result
        return self._server.execute_tool(tool_name, params)
    


def rag_pipeline(query, document_db, api_key, top_k=3):
    """
    Complete RAG pipeline: retrieve, re-rank, and generate answer.

    Args:
        query (str): The user's question
        document_db (list): List of document strings
        api_key (str): Your Anthropic API key
        top_k (int): Number of documents to use as context (default 3)

    Returns:
        str: The generated answer

    Example:
        docs = ["Python is great", "Java is powerful", "Python is easy"]
        answer = rag_pipeline("What is Python?", docs, api_key)
        # Returns an answer based on relevant Python documents
    """
    # TODO: Implement the full RAG pipeline
    # 1. Retrieval: Filter documents containing query terms (top 10)
    # 2. Re-ranking: Score each with LLM (0-10)
    # 3. Selection: Keep top-k by score
    # 4. Generation: Create context and generate answer
    scores = {}
    query = query.lower().split()
    for i, doc in enumerate(document_db):
        score = 0
        doc = doc.lower().split()
        for word in doc:
            if word in query:
                score += 1
        scores[i] = score
    scores_sorted = sorted(scores, key=scores.get, reverse=True)[:10]
    top_10_docs = [document_db[id] for id in scores_sorted]
    relevant_docs = rerank_documents(query, top_10_docs, api_key, top_k)
    system_prompt = "USING THE PROVIDED CONTEXT ANSWER THE QUERY"
    message  = f"PROMPT: {query}, CONTEXT: {str(relevant_docs)}"
    answer = claude_api_call(message, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    return answer['content'][0]['text']


document_db = [
"Python is a high-level programming language",
"Java is used for enterprise applications",
"Python has simple syntax and is beginner-friendly"
]
# query = "What is Python?"
# print(rag_pipeline(query, document_db, api_key, top_k=2))
# # Returns an answer based on the relevant Python documents



def route_task(task_description, api_key):
    """
    Route a task to the appropriate agent pattern.

    Args:
        task_description (str): Description of the task
        api_key (str): Your Anthropic API key

    Returns:
        dict: Dictionary with keys "pattern" and "explanation"
              pattern is one of: "workflow", "react_agent", "orchestrator_worker", or "unknown"

    Example:
        task = "Translate then summarize"
        route_task(task, api_key)
        # Returns: {"pattern": "workflow", "explanation": "Sequential steps..."}
    """
    # TODO: Implement this function
    # 1. Call API with system prompt to determine pattern
    #    Hint: Be explicit about what each pattern means:
    #    - workflow: predetermined steps in fixed order
    #    - react_agent: dynamic reasoning with tools
    #    - orchestrator_worker: decomposable into parallel subtasks
    # 2. Parse the pattern from response
    # 3. Call API again to get explanation
    # 4. Return dict with pattern and explanation
    system_prompt = "You are an AI agent architect. Analyze the task and determine the best pattern: 'workflow' (deterministic steps), 'react_agent' (requires reasoning/tools), or 'orchestrator_worker' (decomposable into subtasks). Respond with ONLY the pattern name. THE RESPONSE MUST BE ONE OF THE FOLLOWING AS STRING WITH NO ADDITIONAL TEXT: ['workflow', 'react_agent', 'orchestrator_worker', 'unknown']"
    response = claude_api_call(task_description, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)['content'][0]['text']
    system_prompt = f"explain why the provided task was categorized as {response}"
    explanation = claude_api_call(task_description, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)['content'][0]['text']
    result = {
        "pattern" : response,
        "explanation" :explanation
    }
    return result

task = "Translate this document from English to French, then summarize it"
print(route_task(task, api_key))