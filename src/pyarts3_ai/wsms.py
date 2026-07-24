import pyarts3_ai.embedding as embedding
import pyarts3 as pa


__all__ = ["startup",
           "direct_search",
           "cross_search",
           "exists",
           "get_description",
           "get_short_description",
           "get_interface",
           "get_options",
           "modified_by",
           "input_to",
           "output_of",
           ]


_wsms = None
_index = None
_embed_model = None


def _set_descriptions() -> None:
    """
    Initializes the WSMs if they haven't been set yet.
    """
    global _wsms
    if _wsms is None:
        _wsms = pa.arts.globals.workspace_methods()


def _set_index(split_sentences: bool,
               clean_run: bool) -> None:
    """
    Initializes the WSMs index if it hasn't been set yet.

    Args:
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """
    global _index, _wsms
    _set_descriptions()

    assert _embed_model is not None, "Embeddings must be set before indexing."

    if _index is None or clean_run:
        _index = embedding.index(embed_model=_embed_model,
                                 descriptions=embedding.describe(names=[ws for ws in _wsms],
                                                                 descriptions=[_wsms[ws].desc for ws in _wsms]),
                                 split_sentences=split_sentences)


def startup(model_name: str = 'all-mpnet-base-v2',
            cross_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            split_sentences: bool = True,
            clean_run: bool = False) -> None:
    """
    Initializes the WSMs, embeddings, and index.

    Args:
        model_name (str): The name of the embedding model to use.
        cross_model_name (str): The name of the cross-rank model to use.
        split_sentences (bool): Whether to split descriptions into individual sentences.
        clean_run (bool): Whether to perform a clean run and re-index all descriptions.
    """
    global _embed_model, _index
    _embed_model = embedding.startup(
        model_name=model_name, cross_model_name=cross_model_name, clean_run=clean_run)
    _index = None  # Reset the index to ensure it is re-initialized
    _set_index(split_sentences, clean_run)


def direct_search(user_query: str,
                  top_k: int = 5) -> list[dict]:
    """
    Performs a direct search on the WSMs based on the user query.

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
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k, type="Workspace Method")


def cross_search(user_query: str,
                 top_k: int = 5) -> list[dict]:
    """
    Performs a cross-rank search on the WSMs based on the user query.

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
        embed_model=_embed_model, index=_index, user_query=user_query, top_k=top_k, type="Workspace Method")


def exists(name: str) -> bool:
    """
    Checks if a WSM with the given name exists.

    Args:
        name (str): The name of the WSM to check.

    Returns:
        bool: True if the WSM exists, False otherwise.
    """
    _set_descriptions()
    global _wsms
    return name in _wsms


def get_description(name: str) -> str:
    """
    Returns the description of a specific WSM.

    Args:
        name (str): The name of the WSM to retrieve.

    Returns:
        str: The description of the WSM, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    global _wsms
    if name not in _wsms:
        return ""
    return eval(f"pa.Workspace.{name}.__doc__")


def get_short_description(name: str) -> str:
    """
    Returns the short description of a specific WSM.

    Args:
        name (str): The name of the WSM to retrieve.

    Returns:
        str: The short description of the WSM, or an empty string if it doesn't exist.
    """
    _set_descriptions()
    global _wsms
    if name not in _wsms:
        return ""
    return _wsms[name].desc.split('\n')[0]


def get_interface(name: str) -> dict:
    """
    Returns the interface of a specific WSM.

    There will be a dictionary with the following keys:
    - 'type': The type of the entity, which is 'Workspace Method' for WSMs.
    - 'input': A list of input variable names for the WSM.
    - 'output': A list of output variable names for the WSM.
    - 'modified': A list of input/output variable names for the WSM.
    - 'options': A list of non-workspace input variables for the WSM.

    Args:
        name (str): The name of the WSM to retrieve.

    Returns:
        dict: {'type': 'Workspace Method', 'input': list, 'output': list, 'modified': list, 'options': list}.
    """
    _set_descriptions()
    global _wsms

    if name not in _wsms:
        return {}

    wsm = _wsms[name]

    inputs = wsm.input
    outputs = wsm.output
    inout = set(inputs).intersection(set(outputs))
    input_only = list(set(inputs) - inout)
    output_only = list(set(outputs) - inout)
    ginputs = wsm.gin

    return {'type': 'Workspace Method',
            'input': list(input_only),
            'output': list(output_only),
            'modified': list(inout),
            'options': list(ginputs)}


def get_options(name: str, input: bool = True) -> dict:
    """
    Returns the local input variables of a specific WSM.

    Args:
        name (str): The name of the WSM to retrieve.
        input (bool): If True, returns the local input variables; if False, returns the local output variables.

    Returns:
        dict: {'type': 'Workspace Method', 'names': list, 'desc': list, 'type': list, 'has_default': list[bool]}
    """
    _set_descriptions()
    global _wsms

    if name not in _wsms:
        return []

    wsm = _wsms[name]

    if input:
        ginputs = list([str(x) for x in wsm.gin])
        ginputs_desc = list([str(x) for x in wsm.gin_desc])
        ginputs_type = list([str(x) for x in wsm.gin_type])
        ginputs_has_default = [wsm.gin_value is not None for g in ginputs]
        return {'type': 'Workspace Method',
                'names': ginputs,
                'desc': ginputs_desc,
                'type': ginputs_type,
                'has_default': ginputs_has_default}
    else:
        goutputs = list([str(x) for x in wsm.gout])
        goutputs_desc = list([str(x) for x in wsm.gout_desc])
        goutputs_type = list([str(x) for x in wsm.gout_type])
        return {'type': 'Workspace Method',
                'names': goutputs,
                'desc': goutputs_desc,
                'type': goutputs_type,
                'has_default': [False for _ in goutputs]}


def modified_by(name: str) -> list[str]:
    """
    Returns the workspace methods that modify a given workspace variable.

    Args:
        name (str): The name of the workspace variable to check.

    Returns:
        list[str]: A list of workspace method names that modify the given workspace variable.
    """
    _set_descriptions()
    global _wsms

    out = []

    for wsm_name, wsm in _wsms.items():
        inputs = wsm.input
        outputs = wsm.output
        inout = set(inputs).intersection(set(outputs))
        if name in inout:
            out.append(wsm_name)

    return out


def input_to(name: str) -> list[str]:
    """
    Returns the workspace methods that take a given workspace variable as input.

    Args:
        name (str): The name of the workspace variable to check.

    Returns:
        list[str]: A list of workspace method names that take the given workspace variable as input.
    """
    _set_descriptions()
    global _wsms

    out = []

    for wsm_name, wsm in _wsms.items():
        inputs = wsm.input
        outputs = wsm.output
        inout = set(inputs).intersection(set(outputs))
        inputs_only = list(set(inputs) - inout)
        if name in inputs_only:
            out.append(wsm_name)

    return out


def output_of(name: str) -> list[str]:
    """
    Returns the workspace methods that produce a given workspace variable as output.

    Args:
        name (str): The name of the workspace variable to check.

    Returns:
        list[str]: A list of workspace method names that produce the given workspace variable as output.
    """
    _set_descriptions()
    global _wsms

    out = []

    for wsm_name, wsm in _wsms.items():
        inputs = wsm.input
        outputs = wsm.output
        inout = set(inputs).intersection(set(outputs))
        outputs_only = list(set(outputs) - inout)
        if name in outputs_only:
            out.append(wsm_name)

    return out


def test_method(method_str: str) -> str:
    """
    Tests the validity of a method string.

    Args:
        method_str (str): The string representation of the method to test. Must follow the format of a Workspace Method, with ws as the workspace, e.g., 'ws.add_species(species=["O2"])'

    Returns:
        str: Confirmation that the method is valid, or an error message if it is not.
    """
    try:
        pa.workspace.workspace.arts_method(method_str)
        return f"'''\n{method_str}\n''' is a valid method string."
    except Exception as e:
        return f"Error parsing method: {e}"
