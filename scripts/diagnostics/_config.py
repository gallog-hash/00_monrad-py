"""
Shared constants for the testLab 20210723 one-off diagnostic scripts.

z-tel order for this dataset: file columns [0,1,2] -> z = [0, -1340, -670] mm.
This column-to-z mapping is dataset-specific and non-obvious (cf.
run_pipeline.py's --z-tel flag) — keep every diagnostic script importing
from here rather than redefining it.
"""

import numpy as np

Z_TEL = np.array([0.0, -1340.0, -670.0])
