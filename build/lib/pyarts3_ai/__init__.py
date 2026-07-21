import pyarts3 as pa

def __getattr__(attr):
    if attr == "wsvs":
        import pyarts3_ai.wsvs as wsvs
        return wsvs
    

def __dir__():
    return [
        "wsvs",
        ]

__all__ = [s for s in dir() if not s.startswith("_")]
__version__ = pa.version
version = __version__
