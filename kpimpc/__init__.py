import importlib
from importlib import metadata


def get_version() -> str:
    try:
        return metadata.version(__name__)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__ = metadata.version("kpimpc")

version: str = get_version()
