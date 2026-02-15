
from key import *
import json
import requests
import claude_api

def extract_search_queries(user_request):
    """
    Extract key search terms from user request for RAG retrieval.

    Args:
        user_request: String containing the user's natural language request

    Returns:
        List[str] of search terms
    """
    user_request = (user_request.lower())
    user_request = user_request.split()
    stopwords = ['a', 'the', 'an', 'to', 'for', 'is', 'do']
    search_terms = [word for word in user_request if word not in stopwords and len(word) >2 ]
    return search_terms
    
user_query = 'apples and oranges'
doc_ids = [0, 1, 2, 3]
doc_summaries = {0: 'apples', 1: 'oranges', 2: 'peaches', 3: 'blueberries and rubparb and apples'}


def filter_relevant_docs(user_query, doc_ids, doc_summaries):
    """
    Select which retrieved documents are relevant to user's specific question.

    Args:
        user_query: String containing the user's query
        doc_ids: List of document IDs
        doc_summaries: Dict mapping doc_id to summary text

    Returns:
        List[str] of relevant doc_ids (preserving original order)
    """
    result = []
    user_query = [word.lower() for word in user_query.split()]
    print(user_query, doc_summaries)
    for id in (doc_ids):
        summary = doc_summaries[id].lower()
        print(id, summary)
        if any(word in summary for word in user_query):
            result.append(id)
    print(result)
    return (result)

#filter_relevant_docs(user_query, doc_ids, doc_summaries)


def decompose_query_with_llm(complex_query, api_key):
    """
    Use Claude to decompose a complex query into multiple search queries.

    Args:
        complex_query: String containing complex user query
        api_key: Anthropic API key

    Returns:
        List[str] of 2-4 specific search queries
    """
    message = f"Decompose the following complex query into 2-4 specific search queries. Return as a comma separated list summary 1, summary 2, etc: . Query: {complex_query}"

    response = claude_api.claude_api_call(message, api_key)
    response = response['content'][0]['text']
    response = [response.strip() for response in response.split(',')]
    print(response)
    return response

complex_query = "search all the internet to find the closest associates of jeffery epstein and find information about the island he used and also his criminal activity"
# decompose_query_with_llm(complex_query, api_key)



query_terms = ['apples', 'oranges']
doc_ids = [0, 1, 2, 3]
documents = {0: 'apples', 1: 'oranges', 2: 'peaches', 3: 'blueberries and rubparb and apples and oranges'}

def score_documents_by_tf(query_terms, documents):
    """
    Score documents by how many query terms they contain.

    Args:
        query_terms: List of search terms
        documents: Dict[str, str] mapping doc_id to document content

    Returns:
        Dict[str, float] mapping doc_id to score (count of matching terms)
    """
    result = {}
    query_terms = [word.lower() for word in query_terms]
    print(query_terms)
    for id in (list(documents.keys())):
        summary = documents[id].lower()
        print(id, summary)
        score = 0 
        for word in summary.split():
            if word in query_terms:
                score += 1
        result[id] = score
    print(result)
    return (result)

#score_documents_by_tf(query_terms, documents)


def rerank_with_llm(query, documents, api_key):
    """
    Use Claude to re-rank documents by relevance to query.

    Args:
        query: String search query
        documents: Dict[str, str] mapping doc_id to document text/summary
        api_key: Anthropic API key

    Returns:
        List[str] of doc_ids ranked by relevance (most relevant first)
    """

    message = f"re-rank documents by relevance to query.Query: {query},documents: {documents}. return comma separated list of ids no additional text "

    response = claude_api.claude_api_call(message, api_key)
    response = response['content'][0]['text']
    response = [response.strip() for response in response.split(',')]
    print(response)
    return response


query ="i love apples and oranges"
documents = {0: 'apples', 1: 'oranges', 2: 'peaches', 3: 'blueberries and rubparb and apples and oranges'}
# rerank_with_llm(query, documents, api_key)


def select_docs_for_context(ranked_doc_ids, doc_token_counts, max_tokens):
    """
    Select top documents that fit within context window limit.

    Args:
        ranked_doc_ids: List of doc_ids in ranked order (best first)
        doc_token_counts: Dict[str, int] mapping doc_id to token count
        max_tokens: Integer maximum token limit

    Returns:
        List[str] of selected doc_ids in original ranked order
    """                 
    result = []
    tokens_used = 0                                                      
    for id in ranked_doc_ids:
        doc_token_count = doc_token_counts[id]
        if tokens_used + doc_token_count <= max_tokens:
            result.append(id)
            tokens_used += doc_token_count
    return result   


def refine_query_with_results(original_query, initial_doc_titles):
    """
    Refine search query based on initial retrieval results.

    Args:
        original_query: String containing original search query
        initial_doc_titles: List of document titles from initial retrieval

    Returns:
        String containing refined query
    """
    new_query = original_query.lower()
    for doc_title in initial_doc_titles:
        for word in doc_title.split(' '):
            if len(word) > 3 and word not in original_query:
                new_query += (' ' + word)
    return new_query


def extract_citation(query, document_text, doc_name):
    """
    Find relevant passage and create source citation.

    Args:
        query: String containing the search query
        document_text: String containing document content
        doc_name: String name of the document

    Returns:
        Dict with keys 'source', 'quote', 'relevance'
    """
    result = {}
    max_relevance = 0 
    for sentence in document_text.split('.'):
        sentence_relevance = 0 
        for word in query.split(' '):
            if word.lower() in sentence.lower():
                sentence_relevance += 1
        if sentence_relevance > max_relevance:
            result['quote'] = sentence
            result['relevance'] = sentence_relevance
            max_relevance = sentence_relevance
    result['source'] = doc_name
    return result



def select_tool_with_llm(user_request, tool_docs, api_key):
    """
    Use Claude to select the best tool for the user's request.

    Args:
        user_request: String containing user's request
        tool_docs: Dict[str, str] mapping tool_name to documentation
        api_key: Anthropic API key

    Returns:
        String name of the selected tool
    """
    message = f"Sekect the best tool given the user request and return only the name of the selected tool. User request: {user_request}, tools: {tool_docs}"
    response = claude_api.claude_api_call(message, api_key)
    response = response['content'][0]['text']
    return response


def verify_with_llm(agent_response, source_docs, api_key):
    """
    Use Claude to verify if agent response is supported by sources.

    Args:
        agent_response: String containing agent's response to verify
        source_docs: List[str] of source document texts
        api_key: Anthropic API key

    Returns:
        Dict with 'is_supported' (bool), 'confidence' (float), 'explanation' (str)
    """
    message = f"please verify if the agent response is supported based on the included docs. Return a comma separated str list with is_supported (bool True/False), confidence int (0-1) and explanation of type str. Values only. Don't include anything else.agent response: {agent_response}, docs: {source_docs}"
    response = claude_api.claude_api_call(message, api_key)
    response = response['content'][0]['text']
    response.split(',')
    is_supported = response[0]
    confidence = response[1]
    explanation = response[2]

    return {
        'is_supported' : is_supported,
        'confidence' : confidence,
        'explanation' : explanation

    }