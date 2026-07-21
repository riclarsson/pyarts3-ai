import pyarts3 as pa


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
        "embedding"
    ]


__all__ = [s for s in dir() if not s.startswith("_")]
__version__ = pa.version
version = __version__
