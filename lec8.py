from claude_api import *
import key
from tools import generate_tool_schema
import requests
import math


def calculate_api_cost(input_tokens, output_tokens, input_price_per_million, output_price_per_million):
    """
    Calculate the total API cost based on input and output token counts.

    Args:
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens
        input_price_per_million (float): Price per million input tokens in dollars
        output_price_per_million (float): Price per million output tokens in dollars

    Returns:
        float: Total cost in dollars
    """
    return (input_tokens/1000000 * input_price_per_million) + (output_tokens/1000000  * output_price_per_million)



# input_tokens=10000
# output_tokens=2000
# input_price_per_million=5.0
# output_price_per_million=25.0
# print(calculate_api_cost(input_tokens, output_tokens, input_price_per_million, output_price_per_million))


def calculate_cost_with_reasoning(input_tokens, completion_tokens, reasoning_tokens, input_price_per_million, output_price_per_million):
    """
    Calculate API cost including hidden reasoning tokens.

    Args:
        input_tokens (int): Number of input tokens
        completion_tokens (int): Number of visible output tokens
        reasoning_tokens (int): Number of hidden reasoning tokens
        input_price_per_million (float): Price per million input tokens in dollars
        output_price_per_million (float): Price per million output tokens in dollars

    Returns:
        float: Total cost in dollars
    """
    total_output = completion_tokens + reasoning_tokens
    return ((input_tokens/1000000)*input_price_per_million) + ((total_output/1000000)*output_price_per_million)



def calculate_cache_break_even(base_price_per_million, cache_write_price_per_million, cache_hit_price_per_million):
    """
    Calculate the minimum number of cache reuses needed to break even.

    Args:
        base_price_per_million (float): Base input price per million tokens
        cache_write_price_per_million (float): Cache write price per million tokens
        cache_hit_price_per_million (float): Cache hit price per million tokens

    Returns:
        int: Minimum number of reuses needed (rounded up)
    """
    num_reuses = 0
    break_even = cache_write_price_per_million + (num_reuses * cache_hit_price_per_million) <= (num_reuses + 1) * base_price_per_million
    while not break_even:
        num_reuses += 1
        break_even = cache_write_price_per_million + (num_reuses * cache_hit_price_per_million) <= (num_reuses + 1) * base_price_per_million
    return num_reuses






def structure_for_caching(system_prompt, documentation, user_query):
    """
    Structure messages to maximize cache efficiency.

    Args:
        system_prompt (str): System instructions (static, cacheable)
        documentation (str): Documentation content (static, cacheable)
        user_query (str): User query (dynamic, changes per request)

    Returns:
        list: Messages array structured for optimal caching
    """
    result = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": documentation},
    {"role": "user", "content": user_query}
    ]
    return result




# messages = structure_for_caching(
#     system_prompt="You are a helpful assistant.",
#     documentation="API docs: use endpoint /v1/chat",
#     user_query="How do I make a request?"
# )



def estimate_tokens(text):
    """Rough token estimation: word count * 1.3"""
    return int(len(text.split()) * 1.3)


def truncate_messages(messages, max_tokens):
    """
    Truncate messages to fit within token limit.

    Args:
        messages (list): List of message dicts with "role" and "content"
        max_tokens (int): Maximum tokens allowed

    Returns:
        list: Truncated messages list
    """
    # TODO: Implement message truncation
    # 1. Always keep system message (first message if role="system")
    # 2. Remove oldest non-system messages until under max_tokens
    # 3. Use estimate_tokens() to count tokens
    total_tokens = 0 
    for message in messages:
        message_content = message['content']
        tokens  = estimate_tokens(message_content)
        total_tokens += tokens
    # for message in messages_reversed:
    #     message_content = message['content']
    #     tokens  = estimate_tokens(message_content)
    #     messages = messages[:-1]
    #     total_tokens = total_tokens - tokens
    #     if total_tokens < max_tokens:
    #         return messages
    i = len(messages) - 1
    while total_tokens > max_tokens and messages:
        last_message = messages[i]
        last_message_speaker = last_message['role']
        if last_message_speaker == 'user':
            tokens = estimate_tokens(last_message['content'])
            total_tokens -= tokens
            messages.pop(i)
        i = i-1
    return messages


messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "user", "content": "How are you"},
    {"role": "assistant", "content": "I'm great"}
]
    
# print(truncate_messages(messages, 10))



def summarize_conversation(messages, api_key):
    """
    Summarize conversation history using Claude API.

    Args:
        messages (list): List of message dicts with "role" and "content"
        api_key (str): Anthropic API key

    Returns:
        str: Summary of the conversation
    """
    system_prompt = 'summarize the current conversation'
    messages = "summarize this conversation history. it should be significantly shorter (less than 1/4) than original. summary only no title or leading text:" + str(messages)
    result = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    return result['content'][0]['text']


messages = [
    {"role": "user", "content": "I need to build a web app"},
    {"role": "assistant", "content": "What framework?"},
    {"role": "user", "content": "React with TypeScript"},
    {"role": "assistant", "content": "Great choice. What features?"}
]

# print(summarize_conversation(messages, api_key))


class NotesManager:
    """
    A structured note-taking system for agents to persist context.
    """

    def __init__(self):
        """Initialize the notes manager."""
        self.notes = {}

    def write_note(self, key, value):
        """
        Write a note with the given key.

        Args:
            key (str): The key for the note
            value: The value to store (can be any type)
        """
        self.notes[key] = value

    def read_note(self, key):
        """
        Read a note by key.

        Args:
            key (str): The key to look up

        Returns:
            The stored value, or None if not found
        """
        return self.notes[key]

    def get_all_notes(self):
        """
        Get all notes as a dictionary.

        Returns:
            dict: All stored notes
        """
        return self.notes

    def clear_notes(self):
        """Clear all notes."""
        self.notes.clear()



def get_doc_from_id(id, docs):
    for doc in docs:
        if doc['id'] == id:
            return doc

def adaptive_top_k_selection(query, documents, token_budget, cost_per_million_tokens=5.0):
    """
    Adaptively select documents to maximize relevance while staying within token budget.

    Args:
        query (str): Search query
        documents (list): List of dicts with "id", "title", "content" keys
        token_budget (int): Maximum tokens allowed for context
        cost_per_million_tokens (float): Cost per million input tokens

    Returns:
        tuple: (selected_doc_ids, total_tokens, total_cost, tokens_saved)
            - selected_doc_ids: List of doc IDs selected within budget
            - total_tokens: Total tokens used by selected documents
            - total_cost: Cost in dollars for the selected documents
            - tokens_saved: Tokens from unselected docs (that would be in top-10 without budget)
    """
    # TODO: Implement adaptive top-K selection
    # 1. Rank documents by relevance:
    #    - Extract keywords from query (lowercase, split by whitespace)
    #    - For each doc, count keyword matches in title + content
    #    - Sort by match count (descending)
    #
    # 2. Greedily select documents:
    #    - Estimate tokens for each doc: len(doc["content"]) // 4
    #    - Iterate through ranked docs, adding them if they fit within budget
    #    - Stop adding when next doc would exceed token_budget
    #
    # 3. Calculate tokens saved:
    #    - Sum tokens of docs ranked in top-10 that were NOT selected
    #
    # 4. Calculate cost:
    #    - total_cost = (total_tokens / 1_000_000) * cost_per_million_tokens
    #
    # Return: (selected_doc_ids, total_tokens, total_cost, tokens_saved)
    total_tokens = 0 
    doc_selection = []
    top_10_tokens = 0
    top_10_ids = []
    query_words = query.lower().split()
    doc_scores = {}
    for doc in documents:
        doc_score = 0 
        doc_text = doc['content'].lower().split() + doc['title'].lower().split()
        print
        for word in query_words:
            word_count = doc_text.count(word)
            doc_score += word_count
        doc_id = doc['id']
        doc_scores[doc_id] = doc_score 
    print(doc_scores)
    doc_scores_sorted = dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))
    i = 0
    for doc_id in doc_scores_sorted:
        doc = get_doc_from_id(doc_id, documents)
        doc_content = doc['content'] + doc['title']
        doc_tokens = len(doc['content']) // 4
        if total_tokens + doc_tokens <= token_budget:
            doc_id = doc['id']
            doc_selection.append(doc_id)
            total_tokens += doc_tokens
            if i <= 10:
                top_10_tokens += doc_tokens
                top_10_ids.append(doc_id)
        i += 1
    total_cost = (total_tokens /1000000) * (cost_per_million_tokens)
    top_10_saved_ids = [id for id in top_10_ids if id not in doc_selection]
    top_10_saved_tokens = [get_doc_from_id(id, docs)['content']//4 for id in top_10_saved_ids]
    top_10_saved_tokens_sum = sum(top_10_saved_tokens)
    tokens_saved = total_tokens - top_10_saved_tokens_sum
    result = (doc_selection, total_tokens, total_cost, tokens_saved)
    print(result)
    return result


documents = [
    {"id": "doc1", "title": "Python Guide", "content": "x" * 1200},  # 300 tokens, matches "Python"
    {"id": "doc2", "title": "Java Basics", "content": "y" * 1600},   # 400 tokens, no match
    {"id": "doc3", "title": "Python APIs", "content": "z" * 2000},   # 500 tokens, matches "Python"
    {"id": "doc4", "title": "Web Dev", "content": "w" * 800}         # 200 tokens, no match
]

# query = "python"
# token_budget = 800
# adaptive_top_k_selection(query, documents, token_budget, cost_per_million_tokens=5.0)


def simple_embed(text):
    """
    Mock embedding function (replaces real embedding API/model).
    In production, this would be OpenAI's text-embedding-ada-002,
    Cohere's embed API, or a local sentence-transformer model.
    """
    # Simple hash-based embedding for demonstration
    hash_val = hash(text.lower()) % 1000
    return [hash_val / 1000.0, (hash_val * 2 % 1000) / 1000.0, (hash_val * 3 % 1000) / 1000.0]


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)


def cached_embedding_retrieval(query, documents, embedding_cache, compute_cost_per_embedding=0.001):
    """
    Retrieve documents using cached embeddings when available.

    Args:
        query (str): Search query
        documents (list): List of dicts with "id" and "content" keys
        embedding_cache (dict): Cache mapping doc_id -> embedding vector
        compute_cost_per_embedding (float): Cost to compute one embedding

    Returns:
        dict with keys:
            - relevant_docs: List of top 3 doc IDs ranked by similarity
            - cache_hits: Number of embeddings reused from cache
            - cache_misses: Number of embeddings computed
            - total_cost: Total cost (query embedding + cache misses)
            - updated_cache: Cache with new embeddings added
    """
    # TODO: Implement cached embedding retrieval
    # 1. Compute query embedding (always costs compute_cost_per_embedding)
    # 2. For each document:
    #    - If doc_id in embedding_cache: reuse (cache hit, free)
    #    - Else: compute with simple_embed() and add to cache (cache miss, costs compute_cost_per_embedding)
    # 3. Calculate cosine similarity between query and each doc
    # 4. Return top 3 most similar doc IDs
    # 5. Calculate total_cost = (1 + cache_misses) * compute_cost_per_embedding
    # 6. Return dict with all required key

    scores = {}
    query_embedding = simple_embed(query)
    cache_hits = 0 
    cache_misses = 0
    for doc in documents:
        doc_id = doc['id']
        doc_content = doc['content']
        doc_embed = simple_embed(doc_content)
        if doc_id in embedding_cache:
            cache_hits += 1
        else:
            cache_misses += 1
            embedding_cache[doc_id] = doc_embed
        cos_sim_query_doc = cosine_similarity(doc_embed, query_embedding)
        scores[doc_id] = cos_sim_query_doc
    sorted_scors =  sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_3_ids = sorted_scors[:3]
    total_cost = (1 + cache_misses) * compute_cost_per_embedding
    result = {
        'relevant_docs': top_3_ids,
        'cache_hits': cache_hits, 
        'cache_misses': cache_misses, 
        'total_cost': total_cost,
        'updated_cache': embedding_cache
    }
    return result


import requests


def filter_tools_by_query(query, tools):
    """Filter tools where query keywords match tool name or description."""
    keywords = query.lower().split()
    relevant_tools = []
    for tool in tools:
        tool_text = (tool["name"] + " " + tool["description"]).lower()
        if any(keyword in tool_text for keyword in keywords):
            relevant_tools.append(tool)
    return relevant_tools


def truncate_history(history, max_messages=5):
    """Keep only the most recent messages."""
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def cost_optimized_agent(user_query, conversation_history, all_tools, api_key, system_prompt="You are a helpful assistant."):
    """
    Build a cost-optimized agent with caching, context management, and tool filtering.

    Args:
        user_query (str): The user's current query
        conversation_history (list): Previous messages (list of dicts with "role" and "content")
        all_tools (list): All available tools (list of dicts with "name", "description", "input_schema")
        api_key (str): Anthropic API key
        system_prompt (str): System prompt (default: "You are a helpful assistant.")

    Returns:
        str: The assistant's text response
    """
    # TODO: Implement cost-optimized agent
    # 1. Truncate conversation_history to max 5 messages
    # 2. Filter all_tools to only relevant ones based on user_query
    # 3. Build messages array: [system message, ...truncated history, user query]
    # 4. Make API call with filtered tools
    # 5. Return assistant's text response
    pass


    conversation_history = truncate_history(conversation_history, max_messages=5)
    all_tools = filter_tools_by_query(user_query, tools)
    messages = conversation_history +  [ {"role":"user", "content": messages} ] 
    response = claude_api_call(messages, api_key, return_type='json', system_prompt = system_prompt, 
                    tools = all_tools, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)