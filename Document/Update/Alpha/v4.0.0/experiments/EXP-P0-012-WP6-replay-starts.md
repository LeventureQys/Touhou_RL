# EXP-P0-012 WP6 replay-start evidence

- Decoder: `touhou_rl/v400/gym/replay_starts.py`, using the read-only TH08 v6 codec (`T8RP` + LZSS) and the decoded stage/FPS offset layout.
- Index semantics: one lazy start reference every **600 input records** per stage (`stage_frame` is an input-record index, not an independently validated engine logic frame). No input stream is expanded into JSON. Engine playback is required to establish exact logical-frame alignment.
- Provenance: each row carries source absolute path, raw replay SHA-256, stage index/name, stage seed, input-record count, executable identity, codec/resource metadata, and `takeover_supported: false`.
- Real source `th08_replay/Th8_ud0059.rpy`: SHA-256 `62b02a172780d846f4109fd0dcc762692c5205692670c0f7400c2026c409a5e9`; 6 stages; 255 sampled starts; script exit code 0.
- Real source `th08_replay/Th8_ud0063.rpy`: SHA-256 `fca18e89b974d9947a794b05871f68ae6a73f8928cd9d8e9e7723308dac48c4b`; included in a two-source run; script exit code 0, 12 stage rows before 600-record sampling and corresponding sampled index output.
- Required test: `1 passed`, exit code 0.
- Mid-run takeover / `ReplayStart` attachment is not implemented; this work intentionally does not claim it.
