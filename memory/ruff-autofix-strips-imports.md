---
name: ruff-autofix-strips-imports
description: The PostToolUse ruff hook removes a just-added import before its usage edit lands
metadata:
  type: feedback
---

A PostToolUse hook runs `ruff` (check --fix + format) after every Edit/Write
in this repo. Its F401 autofix **deletes an import you just added if nothing
uses it yet**. When you add `from x import y` in one Edit and the code that
uses `y` in a *later* Edit, the hook strips `y` in between, and the next edit
fails at runtime/lint with "undefined name".

**Why:** hit it 4+ times in one session wiring `disambiguate_telescope_hits`
into stage5 and adding `Hit`/`PlaneCorrection`/`disambiguate_telescope_hits`
to tests — each time the import vanished before the usage edit.

**How to apply:** add the import and its first use in the **same** Edit, or
add the usage edit first and the import second. If an import "disappears,"
this is why — just re-add it together with a use. (Also: the hook's ruff
*format* may reflow your edits, and `ruff check` is the pre-commit gate, so
keep diffs lint-clean.) Related: [[project_state]].
