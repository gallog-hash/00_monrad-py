---
name: monitor-window-rate
description: "testLab_20210723 accepted-coincidence rate and why fixed monitor windows must be minutes, not seconds"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0796d18b-f38b-4b89-9405-f728ca008b01
---

On `data/0_testLab_20210723`, the full-day pipeline accepts only ~8600
inlier-grade coincidences across the whole ~12 h run (~0.2 accepted
coincidences/s). A `monrad-monitor --window-s 60` therefore holds ~12
coincidences per window — below `PoseFitter.MIN_FIT = 30` — and fits 0 windows.

**Why:** the gate funnel (stage 3) rejects ~90% of the 106k raw coincidences;
accepted rate is what bounds the window, not the raw rate.

**How to apply:** size monitor windows by required z_p resolution, not time —
inspect a few hundred coincidences for approximate `z_p` + rate, look up
`σ_eff,z(z_p)` in `reports/resolution/n_required.csv`, then
`window_s = (σ_eff,z/target)² / rate`. See plan
`atomic-sleeping-stallman.md`. Fitted pose for this data: z_p≈840 mm,
t_x≈180, t_y≈233 mm. Run telescope with `--z-tel 0 -1340 -670` (see
[[testlab-20210723-plane-z-order]]).
