from ._core import Session  # noqa: F401

__all__ = ["Session"]

try:
    from ._core import __version__  # type: ignore
except Exception:
    __version__ = "0.0.0"
