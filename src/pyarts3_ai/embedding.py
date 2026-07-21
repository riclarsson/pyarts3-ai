
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

__all__ = ["startup", "describe", "index", "direct_search", "cross_search"]

_nltk_done = False
_cross_model = None


def _set_nltk_global_downloads() -> None:
    """
    Downloads necessary NLTK resources for sentence splitting.
    """
    from nltk import download

    global _nltk_done
    if not _nltk_done:
        download('punkt', quiet=True)
        download('punkt_tab', quiet=True)
        _nltk_done = True


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Sets the embedding model to be used for generating embeddings.

    Args:
        model_name (str): The name of the embedding model to use.
    Returns:
        SentenceTransformer: An instance of the SentenceTransformer model.
    """
    return SentenceTransformer(model_name)


def _set_cross_rank_model(model_name: str, clean_run: bool) -> CrossEncoder:
    """
    Sets the cross-rank model to be used for ranking embeddings.

    Args:
        model_name (str): The name of the cross-rank model to use.
        clean_run (bool): Whether to perform a clean run and re-initialize the model.
    Returns:
        CrossEncoder: An instance of the CrossEncoder model.
    """

    global _cross_model
    if _cross_model is None or clean_run:
        _cross_model = CrossEncoder(model_name)


def startup(model_name: str, cross_model_name: str, clean_run: bool) -> SentenceTransformer:
    """
    Initializes the embedding model and the cross-rank model.

    Args:
        model_name (str): The name of the embedding model to use.
        cross_model_name (str): The name of the cross-rank model to use.
        clean_run (bool): Whether to perform a clean run and re-initialize the models.

    Returns:
        SentenceTransformer: An instance of the SentenceTransformer model.
    """
    _set_cross_rank_model(cross_model_name, clean_run)
    return _get_embedding_model(model_name)


def describe(names: list[str], descriptions: list[str]) -> list[dict[str, str]]:
    """
    Creates a list of dictionaries mapping names to their descriptions.

    Args:
        names (list[str]): A list of names.
        descriptions (list[str]): A list of descriptions.

    Returns:
        list[dict[str, str]]: A list of dictionaries mapping names to descriptions.
    """
    assert len(names) == len(
        descriptions), "Names and descriptions must have the same length."
    return [{"name": v, "desc": d} for v, d in zip(names, descriptions)]


def index(embed_model: SentenceTransformer,
          descriptions: list[dict[str, str]],
          split_sentences: bool) -> list[dict[str, str, str]]:
    """
    Indexes the descriptions of the variables in the index using the embedding model.

    Args:
        embed_model (SentenceTransformer): The embedding model to use for generating embeddings.
        descriptions (list[dict[str, str]]): A list of dictionaries containing variable descriptions.
        split_sentences (bool): Whether to split descriptions into individual sentences.
    """

    index = []

    if split_sentences:
        from nltk import sent_tokenize
        _set_nltk_global_downloads()

        for var in descriptions:
            # Split description into individual sentences
            sentences = sent_tokenize(var["desc"])
            sentence_embeddings = embed_model.encode(sentences)
            index.append({
                "name": var["name"],
                "desc": var["desc"],
                "embeddings": sentence_embeddings
            })
    else:
        for var in descriptions:
            embedding = embed_model.encode([var["desc"]])
            index.append({
                "name": var["name"],
                "desc": var["desc"],
                "embeddings": embedding
            })

    return index


def direct_search(embed_model: SentenceTransformer,
                  index: list[dict[str, str, str]],
                  user_query: str,
                  top_k: int = 5) -> list[dict]:
    """
    Performs a direct search on the indexed descriptions using cosine similarity.

    Args:
        embed_model (SentenceTransformer): The embedding model to use for generating embeddings.
        index (list[dict[str, str, str]]): The indexed descriptions of the variables.
        user_query (str): The user's query to search for.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of the top_k most relevant variable descriptions based on the user's query.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = embed_model.encode([user_query])
    candidate_scores = []

    # Search across every sentence of every variable
    for var in index:
        # Calculate similarity between query and all sentences of this variable
        sims = cosine_similarity(query_vec, var["embeddings"])[0]

        # Take the MAX score among all sentences (the 'best' match)
        best_match_score = np.max(sims)
        # Store the best match score for later use
        var['direct_score'] = best_match_score
        candidate_scores.append(var)

    # Sort by the best sentence match
    candidate_scores.sort(key=lambda x: x['direct_score'], reverse=True)
    top_candidates = candidate_scores[:top_k]

    return top_candidates


def cross_search(embed_model: SentenceTransformer,
                 index: list[dict[str, str, str]],
                 user_query: str,
                 top_k: int = 5) -> list[dict]:
    """
    Performs a cross-search by first doing a direct search and then re-ranking the top candidates using the cross-encoder model.

    Effectively, this function calls `direct_search` to get the top candidates and then calls `partial_cross_search` to re-rank them.

    Direct search is used to narrow down the candidates, and cross-search is used to refine the ranking based on the cross-encoder model.

    Args:
        embed_model (SentenceTransformer): The embedding model to use for generating embeddings.
        index (list[dict[str, str, str]]): The indexed descriptions of the variables.
        user_query (str): The user's query to search for.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of the top_k most relevant variable descriptions based on the user's query, re-ranked by the cross-encoder model.
    """
    top_candidates = direct_search(embed_model=embed_model,
                                   index=index,
                                   user_query=user_query,
                                   top_k=top_k)

    pairs = [[user_query, cand["desc"]] for cand in top_candidates]
    cross_scores = _cross_model.predict(pairs)

    for i in range(len(top_candidates)):
        top_candidates[i]['final_score'] = cross_scores[i]

    return sorted(top_candidates, key=lambda x: x['final_score'], reverse=True)
