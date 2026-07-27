"""Stage 2 — coincidence search.

Merges the per-detector timestamp streams and groups events that fall inside
the 200 ns coincidence window (DESIGN.md §5).
"""

from .search import (
    WINDOW_NS_DEFAULT as WINDOW_NS_DEFAULT,
    coincidence_stream as coincidence_stream,
)
