"""
AI search engine for pyarts3

providing direct and cross-rank search capabilities across various workspace entities such as variables, methods, agendas, groups, and scenarios."""

def __getattr__(attr):
    if attr == "wsvs":
        import pyarts3_ai.wsvs as wsvs
        return wsvs

    if attr == "wsms":
        import pyarts3_ai.wsms as wsms
        return wsms

    if attr == "wsas":
        import pyarts3_ai.wsas as wsas
        return wsas

    if attr == "wsgs":
        import pyarts3_ai.wsgs as wsgs
        return wsgs

    if attr == "wssns":
        import pyarts3_ai.wssns as wssns
        return wssns

    if attr == "pafun":
        import pyarts3_ai.pafun as pafun
        return pafun

    if attr == "embedding":
        import pyarts3_ai.embedding as embedding
        return embedding

    assert False, f"module {__name__} has no attribute {attr}"


def __dir__():
    return [
        "wsvs",
        "wsms",
        "wsas",
        "wsgs",
        "wssns",
        "pafun",
        "embedding"
    ]


def exists(name: str) -> bool:
    """
    Checks if a workspace variable, method, agenda, group, or scenario exists.

    Args:
        name (str): The name of the workspace entity to check.

    Returns:
        bool: True if the entity exists, False otherwise.
    """

    from pyarts3_ai import wsas, wsgs, wsms, wssns, wsvs, pafun

    return \
        wsvs.exists(name) or \
        wsms.exists(name) or \
        wsas.exists(name) or \
        wsgs.exists(name) or \
        wssns.exists(name) or \
        pafun.exists(name)


def get_description(name: str) -> str:
    """
    Returns the description of a specific workspace variable, method, agenda, group, or scenario.

    Args:
        name (str): The name of the workspace entity to retrieve.

    Returns:
        str: The description of the entity, or an empty string if it doesn't exist.
    """

    from pyarts3_ai import wsas, wsgs, wsms, wssns, wsvs, pafun

    # Keep first as agendas are special and may collide with other names
    if wsas.exists(name):
        return wsas.get_description(name)
    
    if wsgs.exists(name):
        return wsgs.get_description(name)
    
    if wsms.exists(name):
        return wsms.get_description(name)
    
    if pafun.exists(name):
        return pafun.get_description(name)
    
    # Keep second to last as it may collide with wsas
    if wsvs.exists(name):
        return wsvs.get_description(name)
    
    # Keep last as it may collide with wsvs
    if wssns.exists(name):
        return wssns.get_description(name)

    return ""


def startup(model_name: str = 'all-mpnet-base-v2',
            cross_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            split_sentences: bool = True,
            clean_run: bool = False) -> None:
    """
    Initializes the search engines

    Args:
        model_name (str): The name of the embedding model to use.
        cross_model_name (str): The name of the cross-rank model to use.
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """

    from pyarts3_ai import wsas, wsgs, wsms, wssns, wsvs, pafun

    wsas.startup(model_name=model_name,
                 cross_model_name=cross_model_name,
                 split_sentences=split_sentences,
                 clean_run=clean_run)
    wsgs.startup(model_name=model_name,
                 cross_model_name=cross_model_name,
                 split_sentences=split_sentences,
                 clean_run=clean_run)
    wsms.startup(model_name=model_name,
                 cross_model_name=cross_model_name,
                 split_sentences=split_sentences,
                 clean_run=clean_run)
    wssns.startup(model_name=model_name,
                  cross_model_name=cross_model_name,
                  split_sentences=split_sentences,
                  clean_run=clean_run)
    wsvs.startup(model_name=model_name,
                 cross_model_name=cross_model_name,
                 split_sentences=split_sentences,
                 clean_run=clean_run)
    pafun.startup(model_name=model_name,
                  cross_model_name=cross_model_name,
                  split_sentences=split_sentences,
                  clean_run=clean_run)


def direct_search(user_query: str,
                 top_k: int = 5) -> list[dict]:
    """
    Performs a direct search on the WSAs based on the user query.

    The search is performed with the same top_k value for each engine, after
    which the results are combined and sorted by score, returning only the top_k results overall.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the top-k search results.
    """

    from pyarts3_ai import wsas, wsgs, wsms, wssns, wsvs, pafun

    results = []
    results.extend(wsas.direct_search(user_query=user_query, top_k=top_k))
    results.extend(wsgs.direct_search(user_query=user_query, top_k=top_k))
    results.extend(wsms.direct_search(user_query=user_query, top_k=top_k))
    results.extend(wssns.direct_search(user_query=user_query, top_k=top_k))
    results.extend(wsvs.direct_search(user_query=user_query, top_k=top_k))
    results.extend(pafun.direct_search(user_query=user_query, top_k=top_k))

    # Sort the results by score in descending order and return the top_k results
    results.sort(key=lambda x: x['direct_score'], reverse=True)
    return results[:top_k]


def cross_search(user_query: str,
                 top_k: int = 5) -> list[dict]:
    """
    Performs a cross-rank search on all of pyarts3 based on the user query.

    The search is performed with the same top_k value for each engine, after
    which the results are combined and sorted by score, returning only the top_k results overall.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the top-k search results.
    """

    from pyarts3_ai import wsas, wsgs, wsms, wssns, wsvs, pafun

    results = []
    results.extend(wsas.cross_search(user_query=user_query, top_k=top_k))
    results.extend(wsgs.cross_search(user_query=user_query, top_k=top_k))
    results.extend(wsms.cross_search(user_query=user_query, top_k=top_k))
    results.extend(wssns.cross_search(user_query=user_query, top_k=top_k))
    results.extend(wsvs.cross_search(user_query=user_query, top_k=top_k))
    results.extend(pafun.cross_search(user_query=user_query, top_k=top_k))

    # Sort the results by score in descending order and return the top_k results
    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]

__all__ = [s for s in dir() if not s.startswith("_")]
__version__ = "0.0.1"
version = __version__
