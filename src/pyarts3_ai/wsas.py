import pyarts3_ai.embedding as embedding
import pyarts3 as pa


__all__ = ["startup",
           "direct_search",
           "cross_search",
           "exists",
           "get_description",
           "get_short_description",
           ]


_wsas = None
_index = None
_embed_model = None


def _set_descriptions() -> None:
    """
    Initializes the WSAs if they haven't been set yet.
    """
    global _wsas
    if _wsas is None:
        _wsas = pa.arts.globals.workspace_agendas()


def _set_index(split_sentences: bool,
               clean_run: bool) -> None:
    """
    Initializes the WSAs index if it hasn't been set yet.

    Args:
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """
    global _index, _wsas
    _set_descriptions()

    assert _embed_model is not None, "Embeddings must be set before indexing."

    if _index is None or clean_run:
        _index = embedding.index(embed_model=_embed_model,
                                 descriptions=embedding.describe(names=[ws for ws in _wsas],
                                                                 descriptions=[_wsas[ws].desc for ws in _wsas]),
                                 split_sentences=split_sentences)


def startup(model_name: str = 'all-mpnet-base-v2',
            cross_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            split_sentences: bool = True,
            clean_run: bool = False) -> None:
    """
    Initializes the WSAs, embeddings, and index.

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
    Performs a direct search on the WSAs based on the user query.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the top-k search results.
    """

    global _embed_model, _index

    assert _embed_model is not None, "Embeddings must be set before performing a search."
    assert _index is not None, "Index must be set before performing a search."

    return embedding.direct_search(
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k, type="Workspace Agenda")


def cross_search(user_query: str,
                 top_k: int = 5) -> list[dict]:
    """
    Performs a cross-rank search on the WSAs based on the user query.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the top-k search results.
    """

    global _embed_model, _index

    assert _embed_model is not None, "Embeddings must be set before performing a search."
    assert _index is not None, "Index must be set before performing a search."

    return embedding.cross_search(
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k, type="Workspace Agenda")


def exists(name: str) -> bool:
    """
    Checks if a WSA with the given name exists.

    Args:
        name (str): The name of the WSA to check.

    Returns:
        bool: True if the WSA exists, False otherwise.
    """
    _set_descriptions()
    global _wsas
    return name in _wsas


def get_description(name: str) -> str:
    """
    Returns the description of a specific WSA.

    Args:
        name (str): The name of the WSA to retrieve.

    Returns:
        str: The description of the WSA, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    global _wsas
    if name not in _wsas:
        return ""
    return eval(f"pa.Workspace.{name}.__doc__")


def get_short_description(name: str) -> str:
    """
    Returns the short description of a specific WSA.

    Args:
        name (str): The name of the WSA to retrieve.

    Returns:
        str: The short description of the WSA, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    global _wsas
    if name not in _wsas:
        return ""
    return _wsas[name].desc.split('\n')[0]


# Global helper to design a workspace agenda
_ag = None


def init_ag() -> None:
    """
    Initializes the Workspace Agenda if it hasn't been set yet.
    """
    global _ag
    if _ag is None:
        _ag = pa.arts.Agenda(name="Custom Agenda")


def reset_ag() -> None:
    """
    Resets the Workspace Agenda if it hasn't been set yet.
    """
    global _ag
    _ag = None


def extend_ag(method_str: str, mode:str="prepend", name:str="None"):
    """
    Extends the Workspace Agenda with a new method.

    Args:
        method_str (str): The string representation of the method to add.
        mode (str): The mode of adding the method. Can be "append", "prepend", or "insert".
        name (str): The name of the method to insert before, if mode is "insert".
    
    Returns:
        str: A message indicating the result of the operation.
    """

    try:
        ag = pa.workspace.workspace.arts_method(method_str)
    except Exception as e:
        return f"Error creating method from string:\n{e}"

    global _ag
    if _ag is None:
        init_ag()

    if mode == "append":
        _ag.append_methods(ag)
        return f"Method '{method_str}' appended to the agenda."
    elif mode == "prepend":
        _ag.prepend_methods(ag)
        return f"Method '{method_str}' prepended to the agenda."
    elif mode == "insert":
        if name == "None":
            return "Error: 'name' must be provided when mode is 'insert'."
        try:
            _ag.insert_methods(ag, name=name)
            return f"Method '{method_str}' inserted before '{name}' in the agenda."
        except Exception as e:
            return f"Error inserting method:\n{e}"
    else:
        return "Error: Invalid mode. Use 'append', 'prepend', or 'insert'."


def view_ag() -> str:
    """ Returns a string representation of the current Workspace Agenda. """
    global _ag
    init_ag()
    return repr(_ag)


def check_ag() -> str:
    """ Returns a dict of inputs the Agenda needs to be set before it can be run. """
    global _ag
    init_ag()
    v = _ag.get_required_inputs()

    return {"pure inputs": v[0], "modified inputs": v[1]}