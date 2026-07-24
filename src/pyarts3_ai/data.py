import pyarts3_ai.embedding as embedding
import pyarts3 as pa
import os
import time


__all__ = ["startup",
           "direct_search",
           "cross_search",
           "exists",
           "get_description",
           "get_short_description",
           ]


_data = None
_index = None
_embed_model = None


def _digdeeper(start, paths=None, res=None) -> dict:
    if res is None:
        res = {}

    if (time.time() - start) >= 120:
        print("Timeout reached while digging deeper into pyarts3 functionality.")
        return res

    if paths is None:
        paths = pa.arts.globals.parameters.datapath

    for path in paths:
        if path.startswith('.'):
            continue

        if os.path.isdir(path):
            x = []
            for fname in os.listdir(path):
                if not fname.startswith('.'):
                    x.append(os.path.join(path, fname))
            res = _digdeeper(paths=x, res=res, start=start)
        else:
            if "readme" in path.lower():
                with open(path, 'r') as f:
                    content = f.read()
                res[path] = content

    return res


def _set_descriptions() -> None:
    """
    Initializes the pyarts3 functionality if they haven't been set yet.
    """
    global _data
    if _data is None:
        _data = _digdeeper(time.time())


def _set_index(split_sentences: bool,
               clean_run: bool) -> None:
    """
    Initializes the pyarts3 functionality index if it hasn't been set yet.

    Args:
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """
    global _index, _data
    _set_descriptions()

    assert _embed_model is not None, "Embeddings must be set before indexing."

    if _index is None or clean_run:
        _index = embedding.index(embed_model=_embed_model,
                                 descriptions=embedding.describe(names=[n for n in _data],
                                                                 descriptions=[_data[n] for n in _data]),
                                 split_sentences=split_sentences)


def startup(model_name: str = 'all-mpnet-base-v2',
            cross_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            split_sentences: bool = True,
            clean_run: bool = False) -> None:
    """
    Initializes the pyarts3 functionality, embeddings, and index.

    Args:
        model_name (str): The name of the embedding model to use.
        cross_model_name (str): The name of the cross-rank model to use.
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """
    global _embed_model, _index
    _set_descriptions()
    _embed_model = embedding.startup(
        model_name=model_name, cross_model_name=cross_model_name, clean_run=clean_run)
    _index = None  # Reset the index to ensure it is re-initialized
    _set_index(split_sentences, clean_run)


def direct_search(user_query: str,
                  top_k: int = 5) -> list[dict]:
    """
    Performs a direct search on the pyarts3 functionality based on the user query.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the top-k search results.
    """

    assert _embed_model is not None, "Embeddings must be set before performing a search."
    assert _index is not None, "Index must be set before performing a search."

    return embedding.direct_search(
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k, type="Data")


def cross_search(user_query: str,
                 top_k: int = 5) -> list[dict]:
    """
    Performs a cross-rank search on the pyarts3 functionality based on the user query.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the top-k search results.
    """

    assert _embed_model is not None, "Embeddings must be set before performing a search."
    assert _index is not None, "Index must be set before performing a search."

    return embedding.cross_search(
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k, type="Data")


def exists(name: str) -> bool:
    """
    Checks if a pyarts3 functionality with the given name exists.

    Args:
        name (str): The name of the pyarts3 functionality to check.

    Returns:
        bool: True if the pyarts3 functionality exists, False otherwise.
    """
    _set_descriptions()
    global _data
    return name in _data


def get_description(name: str) -> str:
    """
    Returns the description of a specific pyarts3 functionality.

    Args:
        name (str): The name of the pyarts3 functionality to retrieve.

    Returns:
        str: The description of the pyarts3 functionality, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    global _data
    return _data.get(name, "")


def get_short_description(name: str) -> str:
    """
    Returns the short description of a specific pyarts3 functionality.

    Args:
        name (str): The name of the pyarts3 functionality to retrieve.

    Returns:
        str: The short description of the pyarts3 functionality, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    global _data
    return _data.get(name, "").split('\n')[0]
