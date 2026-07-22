import pyarts3_ai.embedding as embedding
import pyarts3 as pa


__all__ = ["startup",
           "direct_search",
           "cross_search",
           "exists",
           "get_description",
           ]


_descriptions = None
_index = None
_embed_model = None


class _Type:
    def __init__(self, name, desc, ptr):
        self.name = name
        self.desc = desc
        self.ptr = ptr

    def __eq__(self, ptr):
        return isinstance(ptr, type(self.ptr)) and self.ptr == ptr

    def __repr__(self):
        return self.name


def _digdeeper(v, res=None) -> list[_Type]:
    if res is not None and v in res:
        return res

    if res is None:
        res = [_Type("pyarts3", pa.__doc__, pa)]

    if not hasattr(v, '__name__') or not v.__name__.startswith("pyarts3"):
        return res

    if v not in res:
        if v.__doc__ is None:
            print(f"Warning: {v.__name__} has no docstring - it is ignored.")
            return res
        res.append(_Type(v.__name__, v.__doc__, v))

    _mod = type(pa), type(pa.arts)
    _fun = type(_digdeeper), type(pa.arts.convert.deg2rad)
    _cls = type(_Type), type(pa.arts.Vector)

    for x in dir(v):
        if x.startswith('_') or x.endswith('_'):
            continue

        tp = getattr(v, x)

        if tp in res:
            continue

        if isinstance(tp, _mod):
            res = _digdeeper(tp, res)
        elif isinstance(tp, _cls) or isinstance(tp, _fun):
            if tp.__doc__ is None:
                print(f"Warning: {v.__name__ + "." + tp.__name__} has no docstring - it is ignored.")
                continue
            res.append(_Type(v.__name__ + "." + tp.__name__, tp.__doc__, tp))

    return res


def _set_descriptions() -> None:
    """
    Initializes the pyarts3 functionality if they haven't been set yet.
    """
    global _descriptions
    if _descriptions is None:
        paal = _digdeeper(pa)
        _descriptions = embedding.describe(names=[n.name for n in paal],
                                           descriptions=[n.desc for n in paal])


def _set_index(split_sentences: bool,
               clean_run: bool) -> None:
    """
    Initializes the pyarts3 functionality index if it hasn't been set yet.

    Args:
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """
    global _index
    _set_descriptions()

    assert _embed_model is not None, "Embeddings must be set before indexing."

    if _index is None or clean_run:
        _index = embedding.index(embed_model=_embed_model,
                                 descriptions=_descriptions,
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
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k)


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
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k)


def exists(name: str) -> bool:
    """
    Checks if a pyarts3 functionality with the given name exists.

    Args:
        name (str): The name of the pyarts3 functionality to check.

    Returns:
        bool: True if the pyarts3 functionality exists, False otherwise.
    """
    _set_descriptions()
    return name in _descriptions


def get_description(name: str) -> str:
    """
    Returns the description of a specific pyarts3 functionality.

    Args:
        name (str): The name of the pyarts3 functionality to retrieve.

    Returns:
        str: The description of the pyarts3 functionality, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    return _descriptions.get(name, {"desc": ""})['desc']
