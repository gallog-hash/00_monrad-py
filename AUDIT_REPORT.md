# Architectural and Structural Audit — `monrad`

**Repository:** `monrad-py` (muon coincidence and probe-alignment pipeline)
**Branch reviewed:** `feat/combinatorial-track-finder` (vs. `main`)
**Scope:** full source tree (`src/monrad/`, `scripts/`, `tests/`), `DESIGN.md`, `README.md`, `handoff.md`
**Method:** static read-through of all stage modules and decoders, cross-check against
`DESIGN.md`, `git log`/`git diff` against `main`, `ruff check`, full test-suite run.

---

## 1. Executive summary

`monrad` reconstructs the relative pose of one or more small "probe" detectors
against a 3-plane muon telescope, using cosmic-ray tracks as a free alignment
source. The codebase is small (~8.7 kLOC including tests), single-purpose, and
unusually well specified: `DESIGN.md` is a 1,029-line algorithm reference that
the code visibly tracks line-by-line, including the rationale for each design
choice. This is the project's strongest asset — the audit below leans on it
as ground truth and reports where the implementation has now outpaced it.

**Overall assessment:** architecturally sound, the streaming/iterator design
is well-executed and tested against its own stated memory bound, and the
statistical core of stage 5 (§4 below) is correct and properly weighted. The
main risks are not bugs but **documentation lag** (a substantial new
combinatorial decoding path is live on this branch but not yet folded into
`DESIGN.md`) and a cluster of **known, self-documented algorithmic
limitations** (curvature-only alignment degeneracy, 4-fold rotation
ambiguity, real-hardware fold/cross-talk pathologies) that are correctly
flagged in the source but represent real precision ceilings once real data is
used.

| | |
|---|---|
| Tests | 163/163 passing |
| Lint | `ruff check .` clean |
| Source (non-test) | ~3,275 lines across 11 modules |
| Test code | ~3,490 lines (test:source ratio ≈ 1.07) |
| Diff vs. `main` | +1,512 / −263 lines across 9 files (not yet merged) |

---

## 2. System overview

```
*_header.txt + *_GPS.bin + *.bin   (per detector, n+1 detectors)
        │
        ▼
 stage 1  reconstruct_stream()      PPS-disciplined integer-ns timestamps
        │
        ▼
 stage 2  coincidence_stream()      200 ns transitive-closure clustering
        │
        ▼
 stage 3  decode_position() /       fiber×ribbon → (x, y, σ) per plane
          reconstruct_plane_        candidate enumeration (combinatorial path)
          candidates()
        │
   ┌────┴────┐
   ▼         ▼
stage 4    stage 5
(telescope (probe pose: t_x, t_y, θ, z_p)
 alignment)
```

Stages 1–3 are shared procedures; only the event selection upstream differs
between the telescope-alignment branch (stage 4, all telescope events) and
the probe-pose branch (stage 5, coincidence survivors only). This is exactly
the separation `DESIGN.md` §3 specifies, and the module boundaries
(`stage1.py` … `stage5.py`, `decoders/`) match `DESIGN.md` §9's module layout
table verbatim.

---

## 3. Architectural review

### 3.1 Streaming design — strength

Every stage boundary in stages 1–2 is a generator boundary; `stage1.py`
buffers at most ~1–2 s of events (`_pending`, `_pre_pps1`), and
`stage2.coincidence_stream` holds only the open transitive-closure cluster
plus the heap's per-stream lookahead of one event. `tests/test_pipeline_stream.py`
asserts a 512 MB peak-heap bound end-to-end, which is the kind of test that
actually defends an architectural property rather than just a behavioural
one — uncommon and valuable.

The two-stream rule for stages 4/5 (never `itertools.tee` a single
`reconstruct_stream()`; open two independent ones) is documented in both
`CLAUDE.md` and `DESIGN.md` §3, and both `README.md`'s usage example and
`scripts/run_pipeline.py` follow it correctly.

### 3.2 Stage 2 correctness — recent fix, now solid

`handoff.md`'s 2026-06-11 entry documents a real double-counting bug: the
original `coincidence_stream` yielded the *entire growing deque* on every
qualifying heap pop, so one event could appear in many overlapping clusters
and get double-fit in stage 5. The fix (current `stage2.py`) emits each
transitive-closure cluster exactly once, on close, producing disjoint
clusters — this matches `DESIGN.md` §5.1's "emit a cluster only when its
last-added event falls off the window" specification exactly, and is now
covered by `tests/test_stage2.py::TestTransitiveClosure`. `stage5._decode_cluster`
additionally enforces exactly-one-telescope / exactly-one-this-probe per
cluster, rejecting ambiguous clusters outright rather than guessing — a
defensible purity-over-efficiency choice that is explicitly tested in
`tests/test_corner_probe_edge_cases.py` (cases E2/E4).

### 3.3 Decoder layer — clean separation, one drift risk

`decoders/{header,gps,position}.py` are low-level, dependency-light, and
exposed both as a library (`BinDecoder`, `GPSDecoder`, `parse_header`) and as
CLI entry points (`monrad-decode-*`). `stage1.py` and `stage3.py` re-derive
the same bit masks (e.g. `tick = v & 0xFFFFFFFFFFFFF`, GEN at bits 52–62) in
their own code rather than calling `GPSDecoder._parse_u64` /
`BinDecoder._parse_u64` — harmless today since both copies agree, but it
means the bit layout is defined in two places. `BinDecoder._find_clusters`
and `_reconstruct_coord` *are* correctly reused by `stage3._axis_candidates`
and `_decode_axis`, so the duplication is partial, not total.

### 3.4 The combinatorial track finder — significant undocumented addition

This branch's commit sequence (`7aff0e5` … `3900946`) adds a materially new
decoding path to stage 5: `reconstruct_plane_candidates()` (stage3.py) emits
every plausible (x, y) candidate per telescope plane — one for an
already-resolved axis, the full ribbon×fiber cross-product for an ambiguous
one — and `PoseFitter._decode_cluster` performs a combinatorial search
(`itertools.product` over the 3 planes' candidate lists, capped at
16/plane) for the candidate triple that minimises the telescope line-fit χ².
This **replaces** `disambiguate_telescope_hits()` /
`recover_efficiency_hits()` on the stage-5 path. Of the two,
`disambiguate_telescope_hits` remains live — stage 4 still calls it directly
(`stage4.py:248`). `recover_efficiency_hits`, however, is now **orphaned**:
verified by grep, it is defined in `stage3.py`, exercised directly by
`tests/test_stage3.py`, and referenced only in a `stage5.py` comment — no
production code path (`stage4.py`, `stage5.py`, or any `scripts/*.py`) calls
it. It implements a real, DESIGN.md-undocumented feature (single-channel
"efficiency dropout" recovery, the `'efficiency'` quality flag) that was
seemingly superseded by the combinatorial search on its one call site and
never reconnected elsewhere. See §5.5 and §5.6 for the consequences.

This is a good design — it resolves the mirror-fold ambiguity (DESIGN.md §10
Deduction #4, ~83% unresolved rate from folded-fiber readout) by letting the
geometry pick the right candidate instead of requiring two already-clean
planes to bootstrap a third. However:

- `DESIGN.md` §8.2 still describes only the original two-plane-recovery
  algorithm; the combinatorial approach is documented solely in code
  comments (`stage5.py:578-611`) and commit messages. A reader who starts
  from `DESIGN.md` (the project's own stated authoritative reference) will
  not learn this path exists.
- The "require ≥1 already-resolved anchor plane" guard
  (`stage5.py:596-611`, `no_anchor_plane` gate) is a deliberate
  anti-pile-up safeguard with good reasoning in the comment and a regression
  test (`test_E2_pileup_same_window_unresolved_rejected`), but it is exactly
  the kind of policy decision `DESIGN.md` §8.4/§8.5 would normally host.
- `CLAUDE.md` states "When code and `DESIGN.md` disagree, the code wins" —
  appropriate for a living repo, but it means `DESIGN.md` is currently stale
  on the single most algorithmically interesting part of this branch.

**Recommendation:** before merging to `main`, fold §8.2's two-plane recovery
description into a description of the candidate-enumeration + χ²-minimising
triple search, including the anchor-plane requirement and its rationale.
This is the highest-value documentation fix available in the repo right now.

### 3.5 Module coupling

Dependency direction is clean and acyclic: `stage5 → stage4, stage3 → stage1`;
`stage4 → stage3 → stage1`; `stage2 → stage1`. No stage imports "downstream."
`scripts/run_pipeline.py` is the only consumer that wires all five stages
together end-to-end as a CLI; it is appropriately kept out of the package
(`src/monrad/`) since it is an application, not a library entry point.

---

## 4. Mathematical foundation — probe pose reconstruction

This is the algorithmic heart of the package (stage 5) and is implemented in
`src/monrad/stage5.py`, matching `DESIGN.md` §8 with the stage-3
combinatorial extension noted in §3.4 above.

### 4.1 Geometry and parameterisation

The telescope defines the reference frame: its three planes are mutually
parallel (validated/corrected by stage 4) and sit at fixed heights
`z₁, z₂, z₃` along the beam axis (nominally `z = [0, 400, 800]` mm, corrected
per-plane by stage 4's fitted `δz`, see §4.5). The probe is a rigid, planar
detector at unknown height `z = z_p`, related to the telescope frame by a
2-D rigid transform — translation `(t_x, t_y)` plus rotation `θ` about z:

```
x = t_x + u·cos θ − v·sin θ
y = t_y + u·sin θ + v·cos θ
z = z_p
```

where `(u, v)` are the probe's own (strip-defined) coordinates. Four unknowns
`(t_x, t_y, θ, z_p)`; each telescope–probe coincidence supplies two scalar
constraints (predicted telescope position at the probe plane vs. the probe's
measured hit), so the system is over-determined for any `N ≥ 3` and is
fit by weighted least squares / nonlinear refinement over the full sample.

### 4.2 Per-coincidence telescope line fit

For each coincidence, the (up to) three telescope-plane positions define a
3-D line via two **independent** weighted linear fits:

```
x(z) = a_x + b_x·z         y(z) = a_y + b_y·z
```

implemented as ordinary weighted least squares (`stage5._tel_line_fit`):
for plane `k` with measured `(x_k, y_k)` and per-axis sigma `σ_{x,k}, σ_{y,k}`,

```
       ⎡1  z₁⎤                ⎡1/σ_{x,1}²        ⎤
A_x =  ⎢1  z₂⎥ ,    W_x =      ⎢     1/σ_{x,2}²   ⎥ ,    p_x = (A_xᵀW_xA_x)⁻¹ A_xᵀW_x x
       ⎣1  z₃⎦                ⎣          1/σ_{x,3}²⎦
```

and identically for `y`. `(A_xᵀW_xA_x)⁻¹` is the 2×2 covariance of
`(a_x, b_x)`; ditto for `y`. Crucially, **X and Y are fit independently with
their own per-plane weights**, because `Hit.sigma_x` and `Hit.sigma_y` can
differ (a `cluster` hit's width, hence σ, is per-axis). This was the subject
of the `fix/per-axis-sigma` branch (`handoff.md`, 2026-06-12 entry) and is
verified by `tests/test_stage5.py::TestHeteroscedasticLineFit`.

When stage 4 has fitted a middle-plane out-of-plane tilt
`(tilt_x, tilt_y)`, the two axis fits additionally use *different* effective
z per plane (`stage5._fit_triple`):

```
z_x,k = z_k + tilt_y,k · x_k        z_y,k = z_k + tilt_x,k · y_k
```

removing the tilt's slope×lever-arm residual exactly, without iteration,
because the coordinate that sets the shift is itself the measured one
(`DESIGN.md` §8.2).

A **χ² track-quality cut** (`_CHI2_TRACK = 4.0`) on this fit's combined
x+y χ² rejects ghost tracks before they reach the pose objective.

### 4.3 The combinatorial extension (this branch)

Where each plane decodes unambiguously, the line fit above is applied
directly to the one candidate per plane. Where mirror-fold or pile-up
ambiguity leaves multiple candidate channels per axis,
`reconstruct_plane_candidates()` enumerates the full Cartesian product of
per-axis candidates for each plane (capped at 16, the worst case of a
2-ribbon × 2-fiber fold on both axes), and `_decode_cluster` evaluates the
§4.2 line fit for **every** candidate triple `(c₀, c₁, c₂) ∈
cands₀ × cands₁ × cands₂`, keeping the triple with minimum χ². This turns
plane-level mirror-fold disambiguation into a search over the discrete
candidate space jointly with the continuous geometric fit — i.e. χ² is
minimised over both the discrete candidate choice and (later, in §4.4–§4.6)
the continuous pose parameters.

### 4.4 The residual and its covariance

For coincidence `i`, the telescope line predicts the probe-plane position at
the (still unknown) `z_p`:

```
x_pred,i(z_p) = a_x,i + b_x,i·z_p        y_pred,i(z_p) = a_y,i + b_y,i·z_p
```

and the probe hit, mapped into the telescope frame, is

```
x_meas,i(θ, t_x) = t_x + u_i·cos θ − v_i·sin θ
y_meas,i(θ, t_y) = t_y + u_i·sin θ + v_i·cos θ
```

`r_i = (x_meas − x_pred, y_meas − y_pred)`. Because the x(z) and y(z) line
fits are independent (§4.2), the residual covariance is **diagonal in x/y**
rather than the general 2×2 block `DESIGN.md` §8.3 writes; in code
(`stage5._sigma_tel_at_z`):

```
Var(x_pred,i) = var_a,x + 2·z_p·cov_ab,x + z_p²·var_b,x
σ²_x,i(z_p)   = σ²_prb,x,i + Var(x_pred,i)
```

and identically for `y`. This is the correct specialisation of the general
`Σ_i = Σ_probe,i + JᵢΣ_line,iJᵢᵀ` form for the case where the x and y line
fits carry no cross-covariance — which holds here exactly, since they are
fit from independent weighted normal equations.

The objective is the standard weighted χ²:

```
χ²(θ, t_x, t_y, z_p) = Σᵢ [ r_{x,i}² / σ²_{x,i}(z_p)  +  r_{y,i}² / σ²_{y,i}(z_p) ]
```

### 4.5 Why the optimiser is well-behaved

Two structural properties make a brute 4-D nonlinear search unnecessary:

1. **Linear-in-`(t_x, t_y, z_p)` at fixed `θ`.** Re-arranging the residual
   equations at fixed `(c, s) = (cos θ, sin θ)`:

   ```
   t_x − b_x,i·z_p = a_x,i − (u_i·c − v_i·s)
   t_y − b_y,i·z_p = a_y,i − (u_i·s + v_i·c)
   ```

   is linear in `(t_x, t_y, z_p)` for every `i` — a standard weighted linear
   least-squares problem with 3 unknowns and `2N` equations, solved in
   closed form (`stage5._linear_solve_fixed_theta`, via `np.linalg.lstsq`
   on the probe-only-weighted system).

2. **The only nonlinearity is the single bounded scalar `θ ∈ [−π, π)`.**

### 4.6 The four-step optimiser (`fit_probe_pose`)

1. **Coarse θ scan**, 1° steps over `[−180°, 180°)`: at each θ, solve the
   linear sub-problem (probe-only weights, no `z_p`-dependence yet) and
   record `χ²_min(θ)`. This produces the **diagnostic χ²(θ) curve**
   (`PoseResult.chi2_curve`) — for a square probe, four equally deep minima
   90° apart are expected; unequal depths flag a wiring/axis fault
   (`DESIGN.md` §8.4 step 2, called out as "the most important consistency
   check in this stage").
2. **Fine θ scan**, 0.01° steps over the global minimum ± 2°, same linear
   solve.
3. **Levenberg–Marquardt polish** (`scipy.optimize.least_squares`,
   `method="lm"`) on all four parameters jointly, using `_weighted_residuals`
   — the *full* `σ²(z_p)`-dependent weights, normalised per-residual so the
   solver minimises a true sum of squared standardised residuals. The
   parameter covariance is read off the inverse Gram matrix of the
   normalised-residual Jacobian, `(JᵀJ)⁻¹` — the standard linearised
   covariance estimate at the optimum.
4. **One-pass Mahalanobis outlier cut**: `d_i = √(r_{x,i}²/σ²_{x,i} +
   r_{y,i}²/σ²_{y,i})`, drop `d_i > 4`, refit on the survivors and recompute
   covariance from the new Jacobian.

A stratified-half consistency check (split inliers by event-index parity,
refit each half at the shared θ, compare `(t_x, t_y, z_p)`) is computed and
returned in `PoseResult.half_params` as a drift/systematic diagnostic
(`DESIGN.md` §8.7).

### 4.7 Known, self-documented precision limits

- **4-fold rotation ambiguity** (`DESIGN.md` §8.5): a square probe with
  identical X/Y strip layouts is invariant under 90°/180°/270° rotation —
  the χ²(θ) curve has four equal minima and external information (mounting
  orientation, marked corner, X/Y asymmetry) is required to break it. This
  is architecture, not a bug.
- **Expected precision** scales as `σ_eff/√N` with
  `σ_eff² = σ_strip² + (3 mrad · z_p)²` (`DESIGN.md` §8.6) — at `z_p = 5 m`
  σ_eff ≈ 15 mm and `z_p` itself becomes degenerate with `(t_x, t_y)`,
  needing hundreds of coincidences and ideally an external tape-measure
  cross-check.
- **Stage-4 curvature-only degeneracy** propagates into stage 5: for the
  nominal evenly-spaced `z = [0, 400, 800]` mm geometry, the two-plane
  residual predictor can only recover the *curvature* (`Δ[0] = Δ[2] =
  −2·Δ[1]`) of a per-plane offset, not which individual plane is displaced
  (`DESIGN.md` §10, "Alignment curvature degeneracy"). Any stage-5 pose fit
  built on top of an uncorrected individual-plane offset will silently
  absorb it into its own `(t_x, t_y)` — this is *exactly* the failure mode
  `DESIGN.md` §7.1 motivates running stage 4 first to guard against, but the
  guard itself has a known blind spot.

---

## 5. Multi-perspective findings

### 5.1 Correctness / algorithmic perspective

| Area | Assessment |
|---|---|
| Integer-ns timestamping (stage 1) | Correct; `_linear()` keeps all arithmetic in integers as `DESIGN.md` §4.2 mandates, avoiding the float-precision trap explicitly called out there. |
| PPS interval acceptance | Residual-based (`res ≤ τ`) test matches spec; dropped-pulse and untrusted-interval paths are logged and propagated as `Quality.UNTRUSTED`/`DEGRADED`, not silently dropped. |
| Split-block / file-boundary stitching | Implemented eagerly (checks GEN continuity before yielding file `k`'s tail), matching the "must be eager" requirement in `DESIGN.md` §4.4. |
| Coincidence clustering | Correct disjoint transitive-closure semantics after the documented 2026-06-11 fix; verified by `TestTransitiveClosure`. |
| Stage 5 weighting | Per-plane/per-axis σ properly threaded end-to-end (§4.2/§4.4 above); this was a real prior bug (`handoff.md`), now fixed and tested. |
| Combinatorial candidate search | Sound in principle; the `no_anchor_plane` guard is a reasoned, tested defence against pile-up masquerading as fold ambiguity — but only tested via synthetic adversarial cases, not real folded-fiber data (see §5.6). |

### 5.2 Robustness / defensive design

- `decode_position` always returns a `Hit` (never bare `None`) specifically
  so callers can pattern-match on `quality` instead of null-checking — a
  good API decision that nonetheless leaves a documented loose end:
  `handoff.md` flags `_decode_cluster` accessing `.quality`/`.x_mm` on a
  `Hit | None` return without a guard in at least one historical code path,
  noted as a "pre-existing ty diagnostic." Worth a final sweep before this
  branch merges, since the type signature still says `Hit | None` in
  `decode_position`'s docstring while the body guarantees `Hit`.
- Malformed/truncated `*.bin` reads are zero-padded rather than raising
  (`stage3._read_block`), with the comment correctly noting a zero word has
  ribbon=0 and is rejected by `_is_valid` — a safe fail-closed default.
- GEN/row-count mismatches and PPS irregularities are logged via
  `logging.warning`, not raised — appropriate for a pipeline meant to run
  unattended over hours of data, though it does mean structural data
  corruption is observable only by grepping logs, not by an exit code or
  summary count. `scripts/run_pipeline.py` does NOT appear to aggregate
  these warnings into its printed summary; a corrupted run could complete
  "successfully" while having silently dropped a meaningful fraction of
  events. This is worth a dedicated counter surfaced in the pipeline
  summary.

### 5.3 Performance / scalability perspective

Already covered architecturally in §3.1; the one perspective worth adding
here is **algorithmic cost of the combinatorial search**: `_decode_cluster`'s
`itertools.product(*cands)` is bounded at 16 candidates/plane, so worst case
16³ = 4096 line fits per coincidence. Each fit is a closed-form 2×2 solve
(cheap), but for telescope planes that are *frequently* ambiguous (the
DESIGN.md §10 hardware findings report ~83% unresolved rates on folded
telescope readout before recovery), this could become the dominant per-event
cost at scale. No benchmark or complexity test currently exercises the
worst-case candidate count; given real hardware is reporting fold rates in
this range, a perf regression test at realistic candidate-count distributions
would be cheap insurance.

### 5.4 Testing perspective

- 163 tests, organized per-stage plus a dedicated end-to-end streaming test
  and a dedicated adversarial corner-probe suite
  (`test_corner_probe_edge_cases.py`) that names and asserts the handling of
  ten distinct physical scenarios (E1–E10) — an unusually thorough test
  taxonomy for a project this size.
- The synthetic generator (`monrad.synth.generate()`) is the sole input
  source for all tests, by design (`DESIGN.md` §11, `CLAUDE.md`). This is
  appropriate given no real detector files ship with the repo, but it means
  **no test currently exercises real folded-fiber / cross-talk statistics**
  — `synth.generate(fold=True)` encodes an idealised, perfectly periodic
  fold pattern (`y_rib = (1 << r) | (1 << (9-r))`), not the messier
  real-world fold-symmetry scores of 0.71–0.95 and the 89.5% all-bits-set
  failure documented for the 2023 BuS_Tracker plane 1 (`DESIGN.md` §10).
  The combinatorial track finder's real payoff is precisely on this messy
  data, which is untested.
- `handoff.md`'s "Findings worth acting on later" (corner-probe audit) is a
  good practice — advisory findings recorded without immediate code churn —
  but two of its three items (the χ²<4 cut's interaction with cluster-width
  σ, and the `z_p`/Mahalanobis interaction at low statistics) are still open
  as of this branch and are not yet tracked as `TODO`/issue markers in code,
  only in a markdown changelog. They risk being lost once `handoff.md` is
  pruned (the file's own housekeeping note flags it as scratch).

### 5.5 Code quality / maintainability perspective

- Consistent style: `ruff check .` and (per CI) `ruff format --check .` both
  pass; docstrings are detailed and consistently explain *why*, not just
  *what*, matching `CLAUDE.md`'s own comment-philosophy guidance even though
  that guidance is aimed at Claude, not the human authors — a good sign of a
  uniform house style.
- `stage5.py`'s `GATE_ORDER` tuple + `DecodeReport` NamedTuple
  (`stage5.py:61-93`) is a clean pattern: it makes the rejection funnel a
  single source of truth that diagnostics scripts import rather than
  re-deriving, explicitly to prevent drift between `_decode_cluster`'s real
  gates and `scripts/run_pipeline.py`'s/`track_coincidence_loss.py`'s
  reporting of them. This is good defensive engineering against a class of
  bug (silently-stale diagnostics) that is easy to introduce in pipelines
  with multiple consumers of the same internal logic.
- Minor duplication noted in §3.3 (bit-layout constants re-declared in
  `stage1.py`/`stage3.py` rather than imported from `decoders/`).
- **Orphaned code**: `stage3.recover_efficiency_hits()` (§3.4) is fully
  implemented, unit-tested in isolation, and undocumented in `DESIGN.md`, but
  is called by nothing in `src/` or `scripts/`. Confirmed by exhaustive grep
  across both directories. This is the kind of function that silently bit-rots
  — its assumptions (e.g. about `Hit.x_or`/`y_or` carrying raw masks) can
  drift out of sync with the rest of stage 3 without any test or caller
  noticing, since its only "caller" is its own test module. Either wire it
  back into a real path (e.g. as a fallback when the combinatorial search's
  `no_anchor_plane` gate rejects a cluster) or remove it and its tests.
- `scripts/run_pipeline.py` at 617 lines is now the largest file in the
  repo outside of stage3/stage5/tests, and mixes CLI parsing, pipeline
  orchestration, diagnostics aggregation, and a 3-D plotting routine
  (`_plot_pose_3d`, ~120 lines) in one file. Not a defect, but a natural
  split point (`_plot_pose_3d` into a `viz` module) if the script keeps
  growing — it already imports `matplotlib`/`plotly` as optional
  visualization deps that the core library doesn't need.

### 5.6 Documentation perspective

Already covered in depth in §3.4 (the combinatorial track finder vs.
`DESIGN.md` §8.2). Summarizing the broader picture:

- `DESIGN.md` §10 ("Open items and assumptions to verify") is exceptionally
  good practice — it is a living register of unverified assumptions
  (first-PPS handling, cross-file PPS continuity, probe channel count
  source, clock-frequency source, telescope plane z-coordinates) each
  phrased as a falsifiable claim to check against real hardware. Genuinely
  one of the better "things we assumed, please verify" sections seen in a
  project this size.
- The real-data fold/cross-talk findings embedded in §10 (three lab
  datasets, per-plane fold-symmetry scores, the 2023 plane-1 89.5%
  saturation finding) blur the line between *design document* and
  *lab notebook*. That is not wrong, but as the document grows it risks
  becoming two documents in one — a stable algorithm spec and a
  growing hardware-characterization log. Splitting §10's hardware findings
  into a separate `HARDWARE_NOTES.md` (referenced from `DESIGN.md`) would
  keep the algorithm spec reviewable on its own and let the hardware log
  grow without diluting it.
- `README.md`'s Python-API example (`alignment = accum.flush()`,
  `PoseFitter(... )`) is consistent with the current `stage4`/`stage5`
  signatures — verified by inspection, this is not stale.
- **Stale decoder filenames (fixed during this audit).** `DESIGN.md`'s intro
  and §2.1/§6.3 referred to the package's reference decoders by their
  pre-refactor flat-script names (`decode_header.py`, `decode_gps.py`,
  `decode_bin.py`), which no longer exist — the actual modules are
  `decoders/header.py`, `decoders/gps.py`, `decoders/position.py`. Since the
  document's own opening paragraph claims these files are authoritative over
  the prose, a reader following that claim literally would have hit three
  dead paths. Corrected in this pass (3 occurrences).
- **`decode_position()` interface drift (§6.1).** The documented signature is
  `decode_position(pos_ref, pos_paths, n_cols) -> list[Hit | None]`. The
  actual signature (`stage3.py:290`) additionally takes `tot_thresh` and
  `tot_weights` — both load-bearing for the `--tot-thresh`/`--tot-weights`
  CLI flags in `scripts/run_pipeline.py` — and its own docstring states it
  *never* returns `None` (every plane always gets a real `Hit`, with quality
  `'invalid'`/`'unresolved'` standing in for absence). So `DESIGN.md` §6.1
  and the function's own `list[Hit | None]` type hint both describe a
  contract the implementation no longer honours; the docstring is the only
  place the true (non-`None`) contract is written down.
- **`Hit.quality` literal drift (§6.4).** `DESIGN.md` step 5 states quality is
  one of `golden`, `cluster`, `unresolved`, `invalid` (4 values). The code's
  `Hit.quality` type (`stage3.py:37`) and `GOOD_QUALITIES` tuple add a 5th,
  `'efficiency'`, produced by the now-orphaned `recover_efficiency_hits()`
  (§5.5). Currently harmless in practice — nothing on the live path emits it
  — but the spec and the type disagree regardless.
- **§9 module-layout table is stale.** The `stage3.py` row lists only `Hit,
  decode_position()`; it omits `reconstruct_plane_candidates()`,
  `PlaneCandidate`, `disambiguate_telescope_hits()`,
  `recover_efficiency_hits()`, and `GOOD_QUALITIES` — all real, non-private
  names other modules import. The `stage5.py` row lists only `Coincidence,
  PoseResult, PoseFitter, fit_probe_pose()`, omitting `DecodeReport` and
  `GATE_ORDER`, which `scripts/run_pipeline.py` and
  `scripts/track_coincidence_loss.py` both import directly as the
  single-source-of-truth diagnostic surface (§5.5). The `Hit` row in the "Key
  types" table also still shows the original 5-field tuple
  `(x_mm, y_mm, sigma_x, sigma_y, quality)`; `Hit` now carries four more
  fields (`candidates_x`, `candidates_y`, `x_or`, `y_or`).

### 5.7 Operational readiness perspective

- The detector geometry (`_Z_TEL = [0, 400, 800] mm`) and the clock
  frequency (`F0_DEFAULT = 100_000_000` Hz, used unconditionally since real
  headers carry no frequency field) are both flagged in `DESIGN.md` §10 as
  unverified-against-hardware assumptions baked in as constants
  (`stage1.py:31`, `stage4.py:26`). These are exactly the kind of
  "looks like a constant, is actually an assumption" risk that should gate
  any first real-data run — both are easy to get wrong silently since
  nothing in the pipeline can detect a wrong `f0` or wrong plane spacing
  from the data alone (the fit would simply absorb the error into the pose).
- Real-hardware fold/cross-talk severity varies a lot by run (`DESIGN.md`
  §10: 2021 ~3% telescope cross-talk vs. 2023 plane-1 at 89.5%). The
  pipeline has no automated per-run health check that would flag "this
  plane is non-functional" before a full pipeline run is attempted —
  `scripts/diagnose_hits.py` exists as a manual diagnostic but isn't wired
  into `run_pipeline.py` as a pre-flight gate. Worth considering for
  production use, given the 2023 BuS_Tracker plane-1 case shows this isn't
  a hypothetical failure mode.

---

## 6. Summary of recommendations (priority order)

1. **Update `DESIGN.md` §8.2/§8.4** to document the combinatorial
   candidate-triple search and its anchor-plane requirement before merging
   this branch to `main` — currently the single largest gap between the
   stated authoritative spec and the actual stage-5 algorithm.
2. **Add a synthetic-data mode that reproduces real fold statistics**
   (partial fold-symmetry, mixed all-bits-set rates) rather than the
   idealised perfect-fold encoder, so the combinatorial track finder's
   primary use case is actually exercised by the test suite.
3. **Surface stage-1/stage-3 structural-warning counts** (GEN mismatches,
   row-count mismatches, untrusted PPS intervals) in
   `scripts/run_pipeline.py`'s printed summary, not just in logs, so a
   corrupted run doesn't read as a clean success.
4. **Resolve the `Hit | None` guard / docstring mismatch** flagged in
   `handoff.md` — small, but it's a latent crash site noted twice in the
   project's own history without being closed. While touching this, update
   `DESIGN.md` §6.1's interface block to the real `decode_position` signature
   (`tot_thresh`, `tot_weights`, always-`Hit` return) and §6.4 to add the
   `'efficiency'` quality value.
5. **Decide `recover_efficiency_hits`'s fate**: wire it into a live call site
   (e.g. as a fallback after the combinatorial search's `no_anchor_plane`
   rejection) or delete it with its tests. Untested-by-omission dead code in
   the decode path is a worse failure mode than no code at all, since it
   looks load-bearing.
6. Lower priority: refresh `DESIGN.md` §9's module-layout/key-types tables
   (missing `reconstruct_plane_candidates`, `PlaneCandidate`,
   `recover_efficiency_hits`, `GOOD_QUALITIES`, `DecodeReport`,
   `GATE_ORDER`, and `Hit`'s extra fields); deduplicate the GPS/position
   bit-layout constants between `decoders/` and `stage1.py`/`stage3.py`;
   consider splitting `scripts/run_pipeline.py`'s plotting code out;
   consider splitting `DESIGN.md` §10's hardware-characterization log from
   the algorithm spec. The stale `decode_header.py`/`decode_gps.py`/
   `decode_bin.py` filename references in `DESIGN.md`'s intro/§2.1/§6.3 were
   corrected to `decoders/{header,gps,position}.py` as part of this audit.

---

## 7. What this audit did not cover

No real detector data files were available in this environment (`data/` was
not inspected for content beyond its existence), so all claims above about
real-hardware behaviour are taken from `DESIGN.md` §10's recorded
measurements, not independently re-derived. `scripts/diagnose_hits.py`,
`scripts/investigate_single_axis.py`, and `scripts/plot_synth.py` were
identified but not read in full; they appear to be ad hoc analysis tools
rather than pipeline components and were judged out of scope for an
architectural review.
