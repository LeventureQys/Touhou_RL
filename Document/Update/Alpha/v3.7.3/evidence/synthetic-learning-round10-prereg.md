# Round10 synthetic rendered learning preregistration

This is a separate synthetic-pixel feature learnability experiment, NOT P3 real pixels, NOT official T21, NOT a deployable model. Existing real alignment/teacher gates stay pending.

- Data: bounded F1/F2/F3 causal toy worlds, existing teacherV2 finite full-episode verified trajectories; only exposed R8 development/validation sources used, no sealed tests or R9 new confirmation.
- Inputs: render current world, depth3 causal screenshots through existing ROI preprocessing; no rendered teacher targets/options/future countdown. State-dependent emissions follow actual actions. Actual previous candidate/hold are history, previous adopted option is BOS, never a teacher label.
- Single-route positive action labels identify one known finite-horizon witness, not all safe/optimal actions; unknown/failed episodes excluded from action supervision and retained as failures.
- Group split locked by renderer before collection. At least train8/validation4 complete events, no mirror/template leakage.
- Network: frozen real v3-st SHA-bound baseline, N1 hidden256, Adam1e-3, gradientclip1, FP32/noAMP. Seed7, maximum200 optimization steps for first run; fixed final step report, no test metric and no checkpoint selection on validation.
- Full-event80step memory used in this diagnostic (not the formal burnin16+seq32 protocol); batch1 event per update round-robin. This difference is reported, not silently treated as T21.
- Report before/after equal-event and row-weighted reliable action agreement, loss, GPU times, frozen backbone equality, source hashes. No claim of gameplay survival benefit.
- Artifact schema v373-synthetic-render-only-heads/1 cannot be loaded as formal tactical candidate. Do not replace default policy. No further epoch increase merely to force a good number; failed learnability is a result.
