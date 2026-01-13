def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata
__version__ = metadata.version("kpimpc")

version: str = get_version()