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

def validate_sql_candidate(candidate: dict, schema: dict) -> dict:
    """
    Validate a SQL candidate for safety and schema correctness.

    Args:
        candidate (dict): Must contain a "sql" key with the SQL query string.
        schema (dict): Mapping of table names to column lists.

    Returns:
        dict: {"valid": bool, "issues": list[str]}
              issues is a sorted list of detected problem strings.

    Raises:
        ValueError: If candidate is None, candidate is missing "sql" key, or schema is None.

    Examples:
        schema = {"products": [...], "sales": [...]}
        validate_sql_candidate({"sql": "SELECT * FROM products"}, schema)
        # {"valid": True, "issues": []}

        validate_sql_candidate({"sql": "DROP TABLE products"}, schema)
        # {"valid": False, "issues": ["dangerous_operation", "not_select_query"]}

        validate_sql_candidate({"sql": "SELECT * FROM unknowntable"}, schema)
        # {"valid": False, "issues": ["unknown_table:unknowntable"]}
    """
    # TODO Step 1: Validate inputs.
    # If candidate is None, raise ValueError.
    # If "sql" not in candidate, raise ValueError.
    # If schema is None, raise ValueError.

    # TODO Step 2: Initialize issues = [] and get sql = candidate["sql"]

    # TODO Step 3: Check for dangerous operations.
    # if re.search(r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b', sql, re.I):
    #     issues.append("dangerous_operation")

    # TODO Step 4: Check that it is a SELECT query.
    # if not re.match(r'\s*SELECT\b', sql, re.I):
    #     issues.append("not_select_query")

    # TODO Step 5: Check referenced tables against schema.
    # matches = re.findall(r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', sql, re.I)
    # for groups in matches:
    #     table = groups[0] or groups[1]  # one group will be non-empty
    #     if table.lower() not in {k.lower() for k in schema.keys()}:
    #         issues.append(f"unknown_table:{table}")

    # TODO Step 6: Return {"valid": len(issues) == 0, "issues": sorted(issues)}
    
    if not candidate or "sql" not in candidate or schema is None:
        raise ValueError
    issues = []
    sql = candidate["sql"]
    if re.search(r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b', sql, re.I):
        issues.append("dangerous_operation")
    if not re.match(r'\s*SELECT\b', sql, re.I):
        issues.append("not_select_query")
    matches = re.findall(r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', sql, re.I)
    for groups in matches:
        table = groups[0] or groups[1]  # one group will be non-empty
        if table.lower() not in {k.lower() for k in schema.keys()}:
            issues.append(f"unknown_table:{table}")
    return {"valid": len(issues) == 0, "issues": sorted(issues)}



def execute_sql_candidates(candidates: list, run_sql) -> list:
    """
    Execute all SQL candidates and return merged result records.

    Args:
        candidates (list): List of dicts, each with at least "sql" and "assumption" keys.
        run_sql (callable): Function that takes a SQL string and returns a dict:
                            {"ok": bool, "rows": list, "columns": list, "error": str|None}

    Returns:
        list: New list of dicts. Each dict is the original candidate merged with the
              run_sql result. Keys in each output dict: assumption, sql, ok, rows, columns, error.

    Raises:
        ValueError: If candidates is None or empty.

    Examples:
        candidates = [
            {"assumption": "by revenue", "sql": "SELECT * FROM products"},
        ]
        results = execute_sql_candidates(candidates, mock_run_sql)
        # results[0] == {"assumption": "by revenue", "sql": "SELECT * FROM products",
        #                "ok": True, "rows": [...], "columns": [...], "error": None}
    """
    # TODO Step 1: Validate inputs.
    # If candidates is None or empty list, raise ValueError.

    # TODO Step 2: For each candidate dict:
    #   Try: result = run_sql(candidate["sql"])
    #   Except Exception as e:
    #       result = {"ok": False, "rows": [], "columns": [], "error": str(e)}
    #
    #   Merge: create a new dict with all keys from candidate + all keys from result.
    #   Append to output list.

    # TODO Step 3: Return the output list.
    if not candidates or candidates == []:
        raise ValueError
    lst = []
    for candidate in candidates:
        try:
            result = run_sql(candidates["sql"])
        except Exception as e:
            result = {"ok": False, "rows": [], "columns": [], "error": str(e)}
        merged = {**candidate, **result}
        lst.append(merged)
    return lst
        
    
def repair_sql_candidate(question: str, schema_context: str, failed_candidate: dict,
                         error_message: str, api_key: str) -> dict:
    """
    Use an LLM to repair a failed SQL candidate.

    Args:
        question (str): The original user question.
        schema_context (str): Formatted schema string.
        failed_candidate (dict): The candidate that failed (must have "sql" key).
        error_message (str): The error message from the failed execution.
        api_key (str): Anthropic API key (starts with 'sk-ant-').

    Returns:
        dict: {"assumption": str, "sql": str, "repaired": True}
              sql is extracted from the LLM response.

    Raises:
        ValueError: If question, schema_context, failed_candidate, or error_message is None.
        ValueError: If failed_candidate is missing the "sql" key.

    Examples:
        failed = {"assumption": "by revenue", "sql": "SELECT * FROM productz"}
        repaired = repair_sql_candidate(
            "top 5 products", schema_context, failed, "no such table: productz", api_key
        )
        # {"assumption": "by revenue", "sql": "SELECT ...", "repaired": True}
    """
    # TODO Step 1: Validate inputs.
    # If question is None, raise ValueError.
    # If schema_context is None, raise ValueError.
    # If failed_candidate is None, raise ValueError.
    # If error_message is None, raise ValueError.
    # If "sql" not in failed_candidate, raise ValueError.

    # TODO Step 2: Build the repair prompt.
    # Include: schema_context, original question, failed SQL, error message.
    # Instruct the LLM to return only the corrected SQL query.

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
    #         "max_tokens": 1000,
    #         "messages": [{"role": "user", "content": prompt}]
    #     }
    # )
    # text = response.json()["content"][0]["text"]

    # TODO Step 4: Extract SQL from the response in priority order:
    # 1. Try re.search(r'```sql\s*(.*?)```', text, re.DOTALL | re.IGNORECASE) → group(1).strip()
    # 2. Try re.search(r'```\s*(.*?)```', text, re.DOTALL) → group(1).strip()
    # 3. Fallback: text.strip()

    # TODO Step 5: Return the repaired candidate dict:
    # {"assumption": failed_candidate.get("assumption", "repaired"),
    #  "sql": extracted_sql.strip(),
    #  "repaire

    if not question or not schema_context or not failed_candidate or not error_message  or "sql" not in failed_candidate:
        raise ValueError
    prompt = f"return ONLY the corrected SQL query: <question> {question} </question> <schema_context> {schema_context} </schema_context> <failed_candidate> {failed_candidate} </failed_candidate> <error> {error_message} </error> "
    response = claude_api_call(prompt, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    text = response["content"][0]["text"]
    try:
        search = re.search(r'```sql\s*(.*?)```', text, re.DOTALL | re.IGNORECASE) 
        search = search.group(1).strip()
    except:
        search = text.strip()
    return {"assumption": failed_candidate.get("assumption", "repaired"),
     "sql": search.strip(),
     "repaired": True}


def cluster_results_by_equivalence(execution_records: list) -> list:
    """
    Group SQL execution records by result equivalence.

    Two records are equivalent if their result rows (after normalization) are identical.
    Normalization: convert all row values to str, then sort the rows.
    Failed executions (ok=False) are excluded.

    Args:
        execution_records (list): List of dicts with keys:
                                  assumption, sql, ok, rows, columns, error

    Returns:
        list: List of cluster dicts:
              {"result_key": tuple-of-tuples, "members": [record, ...], "size": int}
              Sorted by size descending; ties broken by first member's assumption (alphabetically).

    Raises:
        ValueError: If execution_records is None or empty.

    Examples:
        records = [
            {"assumption": "A", "sql": "...", "ok": True,
             "rows": [(1, "X"), (2, "Y")], "columns": [], "error": None},
            {"assumption": "B", "sql": "...", "ok": True,
             "rows": [(2, "Y"), (1, "X")], "columns": [], "error": None},
            {"assumption": "C", "sql": "...", "ok": False,
             "rows": [], "columns": [], "error": "error"},
        ]
        clusters = cluster_results_by_equivalence(records)
        # [{"result_key": ..., "members": [records[0], records[1]], "size": 2}]
        # Failed record C is excluded.
    """
    # TODO Step 1: Validate inputs.
    # If execution_records is None or empty, raise ValueError.

    # TODO Step 2: Filter to only successful records (ok=True).

    # TODO Step 3: Define a normalize function.
    # normalize(rows): convert each value in each row to str, make each row a tuple,
    # sort the rows, return tuple of row-tuples.

    # TODO Step 4: Group records by their normalized result key.
    # groups = {}  # key -> list of records
    # For each record in successful records:
    #   key = normalize(record["rows"])
    #   Append record to groups[key]

    # TODO Step 5: Build cluster dicts.
    # clusters = [{"result_key": key, "members": members, "size": len(members)}
    #             for key, members in groups.items()]

    # TODO Step 6: Sort clusters.
    # Sort by (-size, first_member_assumption_alphabetically).
    # clusters.sort(key=lambda c: (-c["size"], c["members"][0]["assumption"]))

    # TODO Step 7: Return the sorted clusters list.
    if not execution_records:
        raise ValueError
    successful_records = [record for record in execution_records if record.get("ok") is True]
    groups = {}
    for record in successful_records:
        normalized_sorted_record = sorted([tuple(str(rec) for rec in row) for row in record.get("rows", [])])
        key = tuple(normalized_sorted_record)
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    clusters = [{"result_key": key, "members": members, "size": len(members)} for key, members in groups.items()]
    clusters.sort(key=lambda c: (-c["size"], c["members"][0]["assumption"]))
    return clusters



def select_best_sql(execution_records: list) -> dict:
    """
    Select the single best SQL candidate using a consensus-based policy.

    Policy (in priority order):
    1. Filter to records where ok=True AND rows is non-empty.
    2. If none: filter to ok=True (empty results allowed).
    3. If still none: use all records (return the first in original list).
    4. Among filtered records, cluster by result equivalence (same as Q7).
    5. Return the first member of the largest cluster.
       Ties broken by earliest original list index.

    Args:
        execution_records (list): List of dicts with keys: assumption, sql, ok, rows, columns, error.

    Returns:
        dict: One record dict from execution_records.

    Raises:
        ValueError: If execution_records is None or empty.

    Examples:
        records = [
            {"assumption": "A", "sql": "q1", "ok": True, "rows": [(1,)], "columns": [], "error": None},
            {"assumption": "B", "sql": "q2", "ok": True, "rows": [(1,)], "columns": [], "error": None},
            {"assumption": "C", "sql": "q3", "ok": True, "rows": [(2,)], "columns": [], "error": None},
        ]
        winner = select_best_sql(records)
        # Returns records[0] or records[1] (both in the largest cluster of size 2)
    """
    # TODO Step 1: Validate inputs.
    # If execution_records is None or empty, raise ValueError.

    # TODO Step 2: Apply 3-tier filtering.
    # Tier 1: filtered = [r for r in execution_records if r.get("ok") and r.get("rows")]
    # Tier 2: if not filtered: filtered = [r for r in execution_records if r.get("ok")]
    # Tier 3: if not filtered: return execution_records[0]

    # TODO Step 3: Define normalize(rows) — same as Q7.
    # normalize(rows): tuple of sorted row-tuples, each value converted to str.

    # TODO Step 4: Track original indices for tie-breaking.
    # Build filtered_with_index = [(record, original_index), ...]
    # where original_index is the position in execution_records.

    # TODO Step 5: Cluster filtered records by normalized result.
    # groups = {}  # key -> list of (record, original_index)

    # TODO Step 6: Find the largest cluster.
    # Among all groups, pick the one with the highest len.
    # Ties broken by the smallest original_index in the cluster.

    # TODO Step 7: Return the first member (lowest original_index) in the largest cluster.
    if not execution_records:
        raise ValueError
    filtered = [r for r in execution_records if r.get("ok") and r.get("rows")]
    if not filtered: 
        filtered = [r for r in execution_records if r.get("ok")]
    if not filtered: 
        return execution_records[0]
    groups = {}
    filtered_with_index = [
        (record, i)
        for i, record in enumerate(execution_records)
        if record in filtered
    ]
    for record, index in filtered_with_index:
        normalized_sorted_record = sorted([tuple(str(rec) for rec in row) for row in record.get("rows", [])])
        key = tuple(normalized_sorted_record)
        if key not in groups:
            groups[key] = []
        groups[key].append((record, index))
    largest_cluter_size = 0
    for group in groups.values():
        group_len = len(group)
        if group_len > largest_cluter_size:
            best_cluster = group
            largest_cluter_size = group_len
    return min(best_cluster, key=lambda x: x[1])[0]



def produce_final_answer(question: str, selected_sql: str, selected_result: list,
                         api_key: str) -> dict:
    """
    Call an LLM to produce a natural language answer from SQL results.

    Args:
        question (str): The original user question.
        selected_sql (str): The winning SQL query.
        selected_result (list): List of rows (list of tuples or lists). Can be empty.
        api_key (str): Anthropic API key (starts with 'sk-ant-').

    Returns:
        dict: {"answer": str, "sql_used": selected_sql, "row_count": len(selected_result)}

    Raises:
        ValueError: If question, selected_sql, or api_key is None.

    Examples:
        result = produce_final_answer(
            question="What are the top 5 selling products?",
            selected_sql="SELECT product_name FROM products LIMIT 5",
            selected_result=[(1, "Laptop Pro"), (2, "USB-C Hub")],
            api_key=api_key
        )
        # {"answer": "The top selling products are...",
        #  "sql_used": "SELECT product_name FROM products LIMIT 5",
        #  "row_count": 2}
    """
    # TODO Step 1: Validate inputs.
    # If question is None, raise ValueError.
    # If selected_sql is None, raise ValueError.
    # If api_key is None, raise ValueError.

    # TODO Step 2: Format the result as a readable table string (max 10 rows).
    # rows_to_show = selected_result[:10]
    # table_str = "\n".join(str(row) for row in rows_to_show)
    # if not table_str: table_str = "(no rows returned)"

    # TODO Step 3: Build a prompt.
    # Include: original question, SQL used, and result rows.
    # Ask the LLM for a concise natural language answer.

    # TODO Step 4: Call the API using requests.post.
    # response = requests.post(
    #     "https://api.anthropic.com/v1/messages",
    #     headers={
    #         "x-api-key": api_key,
    #         "anthropic-version": "2023-06-01",
    #         "content-type": "application/json"
    #     },
    #     json={
    #         "model": "claude-opus-4-6",
    #         "max_tokens": 500,
    #         "messages": [{"role": "user", "content": prompt}]
    #     }
    # )
    # answer_text = response.json()["content"][0]["text"].strip()

    # TODO Step 5: Return:
    # {"answer": answer_text,
    #  "sql_used": selected_sql,
    #  "row_count": len(selected_result)}
    if not question or not selected_sql or not api_key:
        raise ValueError
    rows_to_show = selected_result[:10]
    table_str = "\n".join(str(row) for row in rows_to_show)
    if not table_str: 
        table_str = "(no rows returned)"
    prompt = f"provide a concise natural language answer <original_q>{question} </original_q> <sql> {selected_sql} </sql> <result> {selected_result if len(selected_result)>0 else "no rows"}</result>"
    response = claude_api_call(prompt, api_key, return_type='json', system_prompt = None, 
                    tools = None, stop_sequence = None, tool_choice = None, prefill = None, print_response = False,
                    temperature = None)
    return {"answer": response["content"][0]["text"].strip(),
     "sql_used": selected_sql,
     "row_count": len(selected_result)}


def text_to_sql_agent(question: str, schema: dict, api_key: str, run_sql) -> dict:
    """
    Full Text-to-SQL agent pipeline.

    Implements all pipeline steps inline (do NOT import helper functions from other questions):
    1. Build schema context (score and rank tables by relevance to question)
    2. Choose disambiguation strategy
    3. Generate SQL candidates via LLM (k=3)
    4. Validate each candidate; skip dangerous ones
    5. Execute valid candidates
    6. For each failed execution, call repair and re-execute
    7. Combine all executions; select best via consensus clustering
    8. Produce final natural language answer

    Args:
        question (str): The user's natural language question.
        schema (dict): Mapping of table name -> list of column names.
        api_key (str): Anthropic API key (starts with 'sk-ant-').
        run_sql (callable): Function that executes SQL and returns
                            {"ok": bool, "rows": list, "columns": list, "error": str|None}

    Returns:
        dict: {
            "final_sql": str,
            "final_answer": str,
            "row_count": int,
            "trace": {
                "schema_context": str,
                "disambiguation": dict,
                "candidates_generated": int,
                "candidates_valid": int,
                "executions_ok": int,
                "repairs_attempted": int,
                "selection_reason": str
            }
        }

    Raises:
        ValueError: If question is None/empty, schema is None/empty,
                    api_key is None, or run_sql is None.

    IMPORTANT: Do NOT import or call functions from other question files.
               Implement all logic inline in this function.
    """
    # TODO Step 1: Validate inputs.
    # If not question or not schema or api_key is None or run_sql is None:
    #     raise ValueError(...)

    # =========================================================================
    # TODO Step 2: BUILD SCHEMA CONTEXT
    # Tokenize question: tokens = [t for t in re.split(r'\W+', question.lower()) if t]
    # For each (table_name, columns) in schema.items():
    #   vocabulary = {table_name.lower()}
    #   For each col: vocabulary |= set(re.split(r'[_\s]+', col.lower())); vocabulary.add(col.lower())
    #   score = sum(1 for t in tokens if t in vocabulary)
    # Sort by (-score, table_name), take top 5, use first 8 columns per table.
    # Format: "Table: {name}\n  Columns: {col1}, ...\n"
    # =========================================================================

    # =========================================================================
    # TODO Step 3: CHOOSE DISAMBIGUATION STRATEGY
    # Check patterns in order:
    # 1. Branch: ranking + business entity words
    # 2. Ask: time words without specific duration
    # 3. Default
    # =========================================================================

    # =========================================================================
    # TODO Step 4: GENERATE SQL CANDIDATES via LLM (k=3)
    # Build prompt, call requests.post("https://api.anthropic.com/v1/messages", ...).
    # headers: x-api-key, anthropic-version: 2023-06-01, content-type: application/json
    # json: model claude-opus-4-6, max_tokens 2000, messages [{"role":"user","content":prompt}]
    # text = response.json()["content"][0]["text"]
    # Parse JSON response (try json.loads, then re.search r'\[.*\]', then fallback).
    # =========================================================================

    # =========================================================================
    # TODO Step 5: VALIDATE each candidate.
    # Check for dangerous operations, non-SELECT, unknown tables.
    # Skip (don't execute) candidates with "dangerous_operation" issue.
    # =========================================================================

    # =========================================================================
    # TODO Step 6: EXECUTE valid candidates.
    # For each valid candidate, call run_sql(candidate["sql"]).
    # Merge result into candidate dict.
    # =========================================================================

    # =========================================================================
    # TODO Step 7: REPAIR failed executions.
    # For each execution record where ok=False:
    #   Build repair prompt, call requests.post(...) with max_tokens=1000.
    #   Extract SQL from response (```sql```, ```, or raw text).
    #   Re-execute the repaired SQL.
    #   Append repaired+re-executed record to all_executions.
    # =========================================================================

    # =========================================================================
    # TODO Step 8: SELECT BEST via consensus.
    # Combine executions + re-executed repairs.
    # Filter: prefer ok=True + non-empty rows; fall back to ok=True; fall back to first.
    # Cluster by normalized result equivalence.
    # winner = first member of largest cluster.
    # selection_reason = f"largest consensus cluster ({cluster_size} members)"
    # =========================================================================

    # =========================================================================
    # TODO Step 9: PRODUCE FINAL ANSWER.
    # Format result rows (max 10), call requests.post(...) with max_tokens=500.
    # text = response.json()["content"][0]["text"].strip()
    # =========================================================================

    # TODO Step 10: Return full result dict.
    if not question or not schema or api_key is None or run_sql is None:
        raise ValueError(...)
    tokens = [t for t in re.split(r'\W+', question.lower()) if t]
    scores  = []
    for (table_name, columns) in schema.items():
        vocabulary = {table_name.lower()}
        for col in columns:
            vocabulary |= set(re.split(r'[_\s]+', col.lower()))
            vocabulary.add(col.lower())
            score = sum(1 for t in tokens if t in vocabulary)
            scores.append((table_name, columns, score))
    scores.sort(key=lambda x: (-x[2], x[0]))
    result_txt = ""
    for table_name, columns, _ in scores[:5]:
        result_txt += f"Table: {table_name}\n  Columns: {', '.join(columns[:8])}\n"