import pyarts3_ai.embedding as embedding
import pyarts3 as pa


__all__ = ["startup",
           "direct_search",
           "cross_search",
           "exists",
           "get_description",
           "get_short_description",
           "get_group_api",
           ]


_wsvs = None
_index = None
_embed_model = None


def _set_descriptions() -> None:
    """
    Initializes the WSVs if they haven't been set yet.
    """
    global _wsvs
    if _wsvs is None:
        _wsvs = pa.arts.globals.workspace_variables()


def _set_index(split_sentences: bool,
               clean_run: bool) -> None:
    """
    Initializes the WSVs index if it hasn't been set yet.

    Args:
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """
    global _index, _wsvs
    _set_descriptions()

    assert _embed_model is not None, "Embeddings must be set before indexing."

    if _index is None or clean_run:
        _index = embedding.index(embed_model=_embed_model,
                                 descriptions=embedding.describe(names=[n for n in _wsvs],
                                                                 descriptions=[_wsvs[n].desc for n in _wsvs]),
                                 split_sentences=split_sentences)


def startup(model_name: str = 'all-mpnet-base-v2',
            cross_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            split_sentences: bool = True,
            clean_run: bool = False) -> None:
    """
    Initializes the WSVs, embeddings, and index.

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
                  top_k: int = 5) -> list[dict[str, str, str]]:
    """
    Performs a direct search on the WSVs based on the user query.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict[str, str, str]]: A list of dictionaries containing the top-k search results.
    """

    global _embed_model, _index

    assert _embed_model is not None, "Embeddings must be set before performing a search."
    assert _index is not None, "Index must be set before performing a search."

    return embedding.direct_search(
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k, type="Workspace Variable")


def cross_search(user_query: str,
                 top_k: int = 5) -> list[dict[str, str, str]]:
    """
    Performs a cross-rank search on the WSVs based on the user query.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict[str, str, str]]: A list of dictionaries containing the top-k search results.
    """

    global _embed_model, _index

    assert _embed_model is not None, "Embeddings must be set before performing a search."
    assert _index is not None, "Index must be set before performing a search."

    return embedding.cross_search(
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k, type="Workspace Variable")


def exists(name: str) -> bool:
    """
    Checks if a WSV with the given name exists.

    Args:
        name (str): The name of the WSV to check.

    Returns:
        bool: True if the WSV exists, False otherwise.
    """
    _set_descriptions()
    global _wsvs
    return name in _wsvs


def get_description(name: str) -> str:
    """
    Returns the description of a specific WSV.

    Args:
        name (str): The name of the WSV to retrieve.

    Returns:
        str: The description of the WSV, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    global _wsvs
    if name not in _wsvs:
        return ""
    return eval(f"pa.Workspace.{name}.__doc__")


def get_short_description(name: str) -> str:
    """
    Returns the short description of a specific WSV.

    Args:
        name (str): The name of the WSV to retrieve.

    Returns:
        str: The description of the WSV, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    global _wsvs
    if name not in _wsvs:
        return ""
    return _wsvs[name].desc.split('\n')[0]

def get_group_api(name: str) -> dict:
    """
    Returns the python API of a specific WSV.

    Args:
        name (str): The name of the WSV to retrieve.

    Returns:
        dict: A dictionary containing the Workspace Group and Python API of the WSV, or an empty dictionary if it doesn't exist.
    """
    _set_descriptions()
    global _wsvs
    if name not in _wsvs:
        return dict()
    return {"Workspace Group": _wsvs[name].type,
            "Python API": str(dir(eval(f"pa.Workspace.{name}")))}