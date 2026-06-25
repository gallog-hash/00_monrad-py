# OPTIMIZATION_NOTES — Step 0b (code-optimization search)

Branch: `feat/monitor-and-package-refactor`. Discovery engine: `/code-review
high` + a cleanup/efficiency finder over the Step 0a refactor diff
(`git diff main...HEAD`), plus `ty` diagnostics. Verification is **synthetic
only** (this track's choice).

**Gate (all green after the applied changes):**

- `uv run pytest` → **171 passed**.
- `uv run ruff check .` clean; `uv run ruff format --check .` clean.
- Synthetic regression `summary.txt` **byte-identical** to the Step 0a baseline
  (`sha256 a862537a…`) — `run_pipeline.py` sets `on_decode`, so every applied
  change is exercised on the unchanged diagnostic path and still reproduces it.
- `ty` errors on `src/monrad/` dropped **16 → 8** (remaining 8 are pre-existing,
  in `decoders/`, `timing/`, `alignment/` — out of Step 0b scope).

---

## Applied

### A1 — Narrow `decode_position` return type `list[Hit | None]` → `list[Hit]`
`src/monrad/reconstruction/hit.py`, `candidates.py`

`decode_position` documents (and its body guarantees) that **every element is a
`Hit`, never `None`** ("always present … rather than checking for None"). The
`| None` was a too-loose annotation that defeated type-narrowing and forced `ty`
to flag every `mh.quality` / `prb_hit.quality` / `decoded_hits[k].quality`
access in `_decode_cluster` as possible-`None`. Narrowing the annotation (and
the internal `hits` list) plus widening
`disambiguate_telescope_hits(z_tel: Sequence[float] | np.ndarray)` to accept the
`np.ndarray` callers actually pass resolves **12 of the 13** `ty` diagnostics in
`pose/fitter.py` + `reconstruction/`. No caller relied on a `None` element (the
two defensive `if h` filters in tests stay valid — a `Hit` is always truthy).
Pure annotation fix, zero runtime change. Addresses the handoff's
pre-existing-`Hit | None` cleanup item.

### A2 — Gate the diagnostic subset replay behind `on_decode`
`src/monrad/pose/fitter.py` `_decode_cluster`

> **Superseded by `4f36f55`.** The subset/combo replay this section gated has
> since been **deleted outright** (`disambiguate_telescope_hits`,
> `SubsetViolation`, `subset_ok`/`subset_violations`, the `"combo"` quality
> label). The gating optimization below is therefore moot — there is no longer a
> replay to gate. Retained as a record of the Step 0b reasoning; do not treat the
> machinery it describes as still present.

After the combinatorial χ² search has already found the winning telescope
triple, `_decode_cluster` was unconditionally **re-decoding all three telescope
planes** (`decode_position(n_cols=3)`) and **re-running the two-plane
`disambiguate_telescope_hits`** purely to (a) refine the per-plane
`tel_quality` label with `"combo"` and (b) compute the `main ⊄ combinatorial`
subset violations. Both outputs are **diagnostics-only**: the fit
(`optimize.py`) never reads `Coincidence.tel_quality`, and `subset_ok` /
`subset_violations` reach only `run_pipeline.py` via the `on_decode` callback.

Now a cheap `tel_quality` is taken straight from the winning candidates'
own `.quality`, and the replay runs **only when `self.on_decode is not None`**.
The streaming/monitoring path (Steps 1–3) leaves `on_decode` unset and skips the
extra decode + disambiguation entirely. `run_pipeline.py` sets `on_decode`, so
its `summary.txt` is byte-identical and every `tel_quality`/subset test (all use
`on_decode`) is unchanged. Implements memory *demote-disambiguation-replay* and
plan Step 0b candidate 3.

**Measured (synthetic, 54 clusters / 45 accepted, best of 50 passes over
`decode_cluster`):**

| path | decode time / pass |
|---|---|
| `on_decode=None` (replay gated off) | 13.5 ms |
| `on_decode=noop` (replay runs) | 15.9 ms |

≈ **15 % less per-cluster decode work** on the monitoring path (~53 µs saved per
accepted cluster — one `decode_position(n_cols=3)` + one `disambiguate` each).
The fraction grows on real fold-ambiguous data where the replay's two-plane fits
do more work. This also subsumes the "double block read"
(`reconstruct_plane_candidates` + `decode_position` reading the same block):
the second read is exactly this replay, so it's gone on the `on_decode=None`
path.

### A3 — Hoist loop-invariant alignment arrays out of the triple loop
`src/monrad/pose/optimize.py` (`TelAlignArrays` + `tel_align_arrays`),
`src/monrad/pose/fitter.py`

`_fit_triple` rebuilt four length-3 `np.array`s of per-plane
`delta_x/delta_y/tilt_x/tilt_y` on **every** call, i.e. up to 16³ = 4096 times
per mirror-fold-ambiguous cluster — even though they're constant across the
whole event. They're now built **once per cluster** (mirroring how `z_corr` was
already hoisted) and passed in as a `TelAlignArrays` bundle; the inner
expressions become vectorized elementwise ops. Identical numerics
(`summary.txt` byte-identical), fewer allocations on the combinatorial hot path.
The benefit is negligible for typical resolved clusters (product = 1 triple) and
scales with fold ambiguity.

---

## Deferred / advisory (logged, not changed)

### D1 — `cov = inv(JᵀJ)` applies no reduced-χ² scale — **decide in Step 1**
`src/monrad/pose/optimize.py` `fit_probe_pose`

The covariance is the inverse Gram matrix of normalised-residual Jacobian with
**no** reduced-χ² multiplier. Whether to rescale by reduced χ² is exactly what
Step 1's **pull test** `(fit − truth)/σ` (should be unit-Gaussian) is designed to
answer. Do **not** change blind in 0b (handoff candidate 1, plan §Step 1). If
pull std ≈ 1, leave as is; if not, apply the rescale there.

### D2 — One-pass Mahalanobis cut vs. iterative — **validate in Step 1**
`src/monrad/pose/optimize.py` `fit_probe_pose`

The outlier cut is one-pass (`d > 4`, single refit). A single high-leverage
accidental can drag `z_p` and survive a one-pass cut at thin statistics. This is
a **fit-behaviour change** (would move results), so it belongs with the Step 1
pull/resolution study that can measure whether it actually helps, not a blind
0b edit (handoff candidate 2).

### D3 — Decode-once reuse — **verified, no change needed**
`fit_probe_pose` operates only on pre-decoded `Coincidence` objects and never
calls `decode_position` / `_decode_cluster` / `reconstruct_plane_candidates`.
`PoseFitter.decode_cluster` runs the combinatorial search **once per cluster**.
So Step 1's repeated subsampling and Step 2's per-window refits re-run only the
cheap linear/LM fit, never the decode. The architecture already satisfies
"decode once" (handoff candidate 4) — confirmed, nothing to do.

### D4 — Stage-2 probe-only cluster filter — **defer to Step 3**
`src/monrad/coincidence/search.py` `coincidence_stream`

`coincidence_stream` emits any cluster spanning ≥ 2 distinct detectors; it does
**not** require the telescope specifically, so a probe-only multi-detector
cluster is emitted and later rejected as `ambiguous_cluster` in
`_decode_cluster`. Harmless today (single probe). Adding a telescope-membership
pre-filter would couple Stage 2 to a privileged detector id it is deliberately
agnostic to. Revisit in **Step 3 (multi-probe)**, where probe-only clusters
become common and the pre-filter pays off (plan candidate 5).

### D5 — `_decode_cluster` recovering `unresolved` telescope hits — **out of scope**
Optional efficiency-branch idea (handoff candidate 6) to fold the old
two-plane recovery (then `disambiguate_telescope_hits`, **deleted in
`4f36f55`**) into the accept path. The combinatorial χ² search already resolves
ambiguous planes globally, so this was a behaviour question for the efficiency
track, not a 0b cleanup. Not pursued.
