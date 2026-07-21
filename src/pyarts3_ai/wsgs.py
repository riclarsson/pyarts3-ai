import pyarts3_ai.embedding as embedding
import pyarts3 as pa


__all__ = ["startup", "direct_search", "cross_search"]


_descriptions = None
_index = None
_embed_model = None


def _set_descriptions() -> None:
    """
    Initializes the WSGs if they haven't been set yet.
    """
    global _descriptions
    if _descriptions is None:
        wsgs = pa.arts.globals.workspace_groups()
        _descriptions = embedding.describe(names=[n for n in wsgs],
                                           descriptions=[wsgs[n].desc for n in wsgs])


def _set_index(split_sentences: bool,
               clean_run: bool) -> None:
    """
    Initializes the WSGs index if it hasn't been set yet.

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
    Initializes the WSGs, embeddings, and index.

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
    Performs a direct search on the WSGs based on the user query.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict[str, str, str]]: A list of dictionaries containing the top-k search results.
    """

    assert _embed_model is not None, "Embeddings must be set before performing a search."
    assert _index is not None, "Index must be set before performing a search."

    return embedding.direct_search(
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k)


def cross_search(user_query: str,
                 top_k: int = 5) -> list[dict[str, str, str]]:
    """
    Performs a cross-rank search on the WSGs based on the user query.

    Args:
        user_query (str): The user's search query.
        top_k (int): The number of top results to return.

    Returns:
        list[dict[str, str, str]]: A list of dictionaries containing the top-k search results.
    """

    assert _embed_model is not None, "Embeddings must be set before performing a search."
    assert _index is not None, "Index must be set before performing a search."

    return embedding.cross_search(
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k)
