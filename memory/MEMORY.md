# Memory index

- [Project state](project_state.md) — pipeline implementation status and streaming redesign
- [Dataset: testLab 20210723](dataset-testlab-20210723.md) — real-data acquisition layout, scale, quirks
- [testLab 20210723 plane z-order](testlab-20210723-plane-z-order.md) — telescope columns not in z order; use `--z-tel 0 -1340 -670`
- [pipeline_out is curated](pipeline-out-is-curated.md) — never rm -rf pipeline_out/; it holds curated result subdirs
- [Ruff autofix strips imports](ruff-autofix-strips-imports.md) — add an import + its first use in the same edit
- [Per-axis recovery rejected](per-axis-recovery-rejected.md) — evaluated and dropped (999 vs 933 inliers, +7%, not worth it); branch deleted
- [Monitor window vs rate](monitor-window-rate.md) — ~0.2 accepted coincidences/s; size monitor windows by required z_p resolution, not fixed seconds
- [testLab 20210723 anomalous window is telescope-side](testlab-20210723-anomalous-window-telescope-side.md) — 17:08–18:26 UTC excursion is telescope-track degradation, not probe motion; SHIPPED residual-RMS gate (--max-resid-rms) uses all-coincidence RMS (inlier-only RMS is flat ~14 mm; anomaly 281 vs ~150 baseline, ~1.6× margin, threshold ~220 mm for this setup)
