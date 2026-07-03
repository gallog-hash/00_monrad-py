---
name: pipeline-out-is-curated
description: pipeline_out/ holds curated result subdirs — never rm -rf the directory itself
metadata:
  type: feedback
---

`pipeline_out/` is not disposable scratch. It holds the user's curated
result subdirectories (e.g. `2021/`, `2021_corrected/`, `testLab_20210723/`)
alongside any ad-hoc run output at its root.

**Why:** when asked to clean up throwaway `pipeline_out_*` runs, I proposed
`rm -rf pipeline_out`; the user rejected it and said "preserve pipeline_out/
directory and remove the others files" — those subdirs predate the session.

**How to apply:** Keep the `pipeline_out/` directory itself. To swap its
contents, `mv -f` individual files in/out; only delete sibling
`pipeline_out_*` dirs/files, never the base directory. Default `--out` for
`scripts/run_pipeline.py` is `./pipeline_out`, so fresh runs land at its root
(`summary.txt`, plus redirected `run.log`/`run.err`) — leave the subdirs alone.
See [[dataset-testlab-20210723]].
