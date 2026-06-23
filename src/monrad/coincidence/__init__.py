"""Stage 2 — coincidence search.

Merges the per-detector timestamp streams and groups events that fall inside
the 200 ns coincidence window (DESIGN.md §5).
"""

from .search import coincidence_stream as coincidence_stream
