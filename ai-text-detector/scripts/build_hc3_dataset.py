"""Backward-compatible entry point for the HC3 dataset builder."""

try:
    from .prepare_hc3 import main
except ImportError:
    from prepare_hc3 import main


if __name__ == "__main__":
    main()
