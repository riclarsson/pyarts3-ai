import embedding
import pyarts3 as pa

_wsv_descriptions = None
_wsv_index = None
_wsv_embeddings = None


def _set_wsv_descriptions() -> None:
    """
    Initializes the WSVs if they haven't been set yet.
    """
    global _wsv_descriptions
    if _wsv_descriptions is None:
        wsvs = pa.arts.globals.workspace_variables()
        _wsv_descriptions = embedding.describe(names=[n for n in wsvs],
                                               descriptions=[wsvs[n].desc for n in wsvs])


def _set_wsv_index(split_sentences: bool,
                   clean_run: bool) -> None:
    """
    Initializes the WSVs index if it hasn't been set yet.

    Args:
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """
    global _wsv_index
    _set_wsv_descriptions()

    assert _wsv_embeddings is not None, "Embeddings must be set before indexing."

    if _wsv_index is None or clean_run:
        _wsv_index = embedding.index_descriptions(embed_model=_wsv_embeddings,
                                                   descriptions=_wsv_descriptions,
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
    global _wsv_embeddings, _wsv_index
    _set_wsv_descriptions()
    _wsv_embeddings = embedding.startup(
        model_name=model_name, cross_model_name=cross_model_name, clean_run=clean_run)
    _wsv_index = None  # Reset the index to ensure it is re-initialized
    _set_wsv_index(split_sentences, clean_run)


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
    return embedding.direct_search(
        embed_model=_wsv_embeddings, index=_wsv_index, user_query=user_query, top_k=top_k)


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
    return embedding.cross_search(
        embed_model=_wsv_embeddings, index=_wsv_index, user_query=user_query, top_k=top_k)
