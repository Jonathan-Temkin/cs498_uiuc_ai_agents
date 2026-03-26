import re
from claude_api import *
import json

def build_schema_context(question: str, full_schema: dict, max_tables: int = 5, max_columns_per_table: int = 8) -> str:
    """
    Build a ranked schema context string for inclusion in an LLM prompt.

    Scores each table by how many question tokens appear in its vocabulary,
    selects the top-scoring tables, and formats them as a structured string.

    Args:
        question (str): The user's natural language question.
        full_schema (dict): Mapping of table name -> list of column name strings.
        max_tables (int): Maximum number of tables to include. Default 5.
        max_columns_per_table (int): Maximum columns per table to include. Default 8.

    Returns:
        str: Concatenated table blocks. Each block is:
             "Table: {name}\\n  Columns: {col1}, {col2}, ...\\n"

    Raises:
        ValueError: If question is None or empty, or full_schema is None or empty.

    Examples:
        schema = {"products": ["product_id", "product_name", "price"],
                  "orders": ["order_id", "customer_id", "total"]}
        build_schema_context("show me products", schema, max_tables=1)
        # Returns: "Table: products\\n  Columns: product_id, product_name, price\\n"
    """
    # TODO Step 1: Validate inputs.
    # If question is None or empty string, raise ValueError.
    # If full_schema is None or empty dict, raise ValueError.

    # TODO Step 2: Tokenize the question.
    # tokens = [t for t in re.split(r'\W+', question.lower()) if t]

    # TODO Step 3: Score each table.
    # For each (table_name, columns) in full_schema.items():
    #   vocabulary = {table_name.lower()}
    #   For each col in columns:
    #       vocabulary |= set(re.split(r'[_\s]+', col.lower()))
    #       vocabulary.add(col.lower())
    #   score = sum(1 for t in tokens if t in vocabulary)
    #   Store (score, table_name, columns) in a list.

    # TODO Step 4: Sort by (-score, table_name) then take top max_tables.

    # TODO Step 5: Build and return the output string.
    # For each selected table, use only the first max_columns_per_table columns.
    # Format: f"Table: {name}\n  Columns: {', '.join(cols)}\n"

    if not question or not full_schema:
        raise ValueError
    tokens = [t for t in re.split(r'\W+', question.lower()) if t]
    lst = []
    for (table_name, columns) in full_schema.items():
        vocabulary = {table_name.lower()}
        for col in columns: 
            vocabulary |= set(re.split(r'[_\s]+', col.lower()))
            vocabulary.add(col.lower())
        score = sum(1 for t in tokens if t in vocabulary)
        lst.append((score, table_name, columns))
    lst.sort(key=lambda x: (-x[0], x[1]))
    top_lst = lst[:max_tables]
    result = []
    for score, table_name, columns in top_lst:
        cols = columns[:max_columns_per_table]
        result.append(f"Table: {table_name}\n  Columns: {', '.join(cols)}\n")
    return "".join(result)



def choose_disambiguation_strategy(question: str) -> dict:
    """
    Analyze a user question and choose how the agent should handle ambiguity.

    Args:
        question (str): The user's natural language question.

    Returns:
        dict: {"strategy": str, "reason": str}
              strategy is one of: "branch", "ask", "default"

    Raises:
        ValueError: If question is None.

    Examples:
        choose_disambiguation_strategy("What are the top 5 selling products last year?")
        # {"strategy": "branch", "reason": "ambiguous ranking metric: could be by revenue or by units"}

        choose_disambiguation_strategy("Show me the latest invoices")
        # {"strategy": "ask", "reason": "time period is ambiguous"}

        choose_disambiguation_strategy("Show orders in the last 7 days")
        # {"strategy": "default", "reason": "question is sufficiently specific"}
    """
    # TODO Step 1: Validate input.
    # If question is None, raise ValueError.

    # TODO Step 2: Check for "branch" strategy first.
    # Condition: re.search(r'\b(best|top|highest|lowest|most|least)\b', question, re.I)
    #            AND re.search(r'\b(selling|sold|revenue|sales|customers|products|orders)\b', question, re.I)
    # If both match: return {"strategy": "branch",
    #                        "reason": "ambiguous ranking metric: could be by revenue or by units"}

    # TODO Step 3: Check for "ask" strategy.
    # Condition: re.search(r'\b(recent|latest|new|last|past)\b', question, re.I)
    #            AND NOT re.search(r'\b\d+\s*(days?|weeks?|months?|years?)\b', question, re.I)
    # If both conditions met: return {"strategy": "ask", "reason": "time period is ambiguous"}

    # TODO Step 4: Default strategy.
    # return {"strategy": "default", "reason": "question is sufficiently specific"}

    if not question:
        raise ValueError
    if re.search(r'\b(best|top|highest|lowest|most|least)\b', question, re.I) and re.search(r'\b(selling|sold|revenue|sales|customers|products|orders)\b', question, re.I):
        return {"strategy": "branch","reason": "ambiguous ranking metric: could be by revenue or by units"}
    elif re.search(r'\b(recent|latest|new|last|past)\b', question, re.I) and not re.search(r'\b\d+\s*(days?|weeks?|months?|years?)\b', question, re.I):
         return {"strategy": "ask", "reason": "time period is ambiguous"}
    return {"strategy": "default", "reason": "question is sufficiently specific"}



def generate_sql_candidates(question: str, schema_context: str, api_key: str, k: int = 3) -> list:
    """
    Call an LLM API to generate k SQL candidates for a given question.

    Args:
        question (str): The user's natural language question.
        schema_context (str): Formatted schema string from build_schema_context.
        api_key (str): Anthropic API key (starts with 'sk-ant-').
        k (int): Number of SQL candidates to generate. Default 3.

    Returns:
        list: List of dicts, each with at least "assumption" (str) and "sql" (str).

    Raises:
        ValueError: If question is None/empty, schema_context is None/empty, or k < 1.

    Examples:
        candidates = generate_sql_candidates(
            "top 5 selling products", schema_context, api_key, k=2
        )
        # Returns: [{"assumption": "by revenue", "sql": "SELECT ..."}, ...]
    """
    # TODO Step 1: Validate inputs.
    # If question is None or empty, raise ValueError.
    # If schema_context is None or empty, raise ValueError.
    # If k < 1, raise ValueError.

    # TODO Step 2: Build a prompt.
    # Include: schema_context, question, and instruction to return a JSON array
    # of k objects each with "assumption" and "sql" keys.

    # TODO Step 3: Call the API using requests.post.
    # response = requests.post(
    #     "https://api.anthropic.com/v1/messages",
    #     headers={
    #         "x-api-key": api_key,
    #         "anthropic-version": "2023-06-01",
    #         "content-type": "application/json"
    #     },
    #     json={
    #         "model": "claude-opus-4-6",
    #         "max_tokens": 2000,
    #         "messages": [{"role": "user", "content": prompt}]
    #     }
    # )
    # text = response.json()["content"][0]["text"]

    # TODO Step 4: Parse the response as JSON.
    # Try json.loads(text) first.
    # If that fails, try to extract a JSON array with:
    #   m = re.search(r'\[.*\]', text, re.DOTALL)
    #   if m: candidates = json.loads(m.group(0))
    # If all parsing fails, return:
    #   [{"assumption": "default", "sql": text.strip()}]

    # TODO Step 5: Return the parsed list of candidate dicts.
    if not question or not schema_context or k < 1:
        raise ValueError
    instructions = f"<schema_context>{schema_context} </schema_context> <prompt> {question} </prompt>  return a JSON array of k objects each with assumption and json keys"
    result = claude_api_call(instructions, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    text = result['content'][0]['text']
    try:
        candidates = json.loads(text)
    except:
        try:
            m = re.search(r'\[.*\]', text, re.DOTALL)
            if m: 
                candidates = json.loads(m.group(0))
            else:
                return [{"assumption": "default", "sql": text.strip()}]
        except:
            return [{"assumption": "default", "sql": text.strip()}]
    
    return candidates