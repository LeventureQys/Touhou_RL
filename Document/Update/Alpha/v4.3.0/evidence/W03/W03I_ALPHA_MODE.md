# W03I — `alpha_mode`: an explicit opt-in for the real engine's blended alpha

Date: 2026-11-15
Repo: `D:\WorkShop\Touhou_RL`
Files changed (whitelist):

- `touhou_rl/v430/observation.py` — **additive only**: one keyword-only parameter,
  one recorded field, one accepted mode, identity/digest provenance.
- `tests/test_v430_observation.py` — **appended only**: 9 new tests, 47 existing
  tests untouched (no line of them changed).
- `Document/Update/Alpha/v4.3.0/evidence/W03/W03I_ALPHA_MODE.md` (this file, new)
- `Document/Update/Alpha/v4.3.0/evidence/W03/EXIT_CODES_W03I.txt` (new)

## 1. The gap: an over-strict clause, not an engine fault

The real engine readback (`glReadPixels` over the game window) does **not** return a
uniform alpha plane. Measured: **202..219**, blended. The strict gate refused it:

```
FrameAlphaError: alpha plane holds intermediate values (202..219);
the buffer is blended/premultiplied and dropping alpha is not lossless
```

The module's own `ALPHA_POLICY` already stated that *the alpha channel is dropped,
never blended*, and the pre-existing v4.2.0 path (`touhou_rl/v420/rl.py`
`_green_plane()`) reads `array[:, :, 1]` and never looks at alpha at all — the
28.8 GB corpus was produced that way. So "alpha must be uniform" protected **no
observation value**; it only locked the only real data source out. The clause was
over-strict in the original spec, and it is relaxed here **explicitly and by
opt-in**, not silently.

## 2. What changed (API)

```python
def encode_o2(frames, key_history=None, history_valid=None,
              elapsed_capture_s=None, *, alpha_mode: str = "strict",
              **unexpected) -> O2Observation
```

`alpha_mode` sits **before** `**unexpected`, so it is a real parameter and can no
longer be swallowed as an unknown keyword (the unknown-keyword guard is unchanged
for every other name). It is keyword-only: an existing four-positional call site
keeps its meaning, and a mistyped fifth positional is impossible.

`O2Observation` gained one **provenance** field:

```python
alpha_mode: str = "strict"   # observed gate: "any" or a strict verdict
```

It is deliberately **not** one of the six encoded fields: `as_dict()` still returns
exactly `O2_ENCODED_KEYS` (six fields, contract order), the actor input surface is
unchanged, `to_float_minibatch` keys are unchanged, and `attach_memory` still
produces the same seven-field `ActorObservation`. `check_o2_observation` keeps
refusing extra keys in a mapping, so nobody can smuggle `alpha_mode` into the
encoded data.

## 3. Final values and the default

| value | meaning | alpha gate |
| --- | --- | --- |
| `"opaque_255"` | constant 255 plane (normal opaque capture) | requires this exact verdict |
| `"unused_0"` | constant 0 plane (32bpp DIB alpha never written) | requires this exact verdict |
| `"binary_mixed"` | only 0/255 (a mask we do not apply) | requires this exact verdict |
| `"any"` | **do not inspect alpha at all** | none (alpha still dropped) |
| `"strict"` | **default policy name**, not a verdict | strict classifier; intermediate alpha refused |

- `ALPHA_MODES == ("opaque_255", "unused_0", "binary_mixed", "any")` — the four
  concrete modes; `ALPHA_STRICT_MODES` is the first three.
- `ALPHA_MODE_CHOICES == ALPHA_MODES + ("strict",)` — the four concrete values plus
  the default policy name. No other alias exists; `"strict"` is intentionally **not**
  in `ALPHA_MODES` so no code can mistake a policy for an observed plane.

**Why the default is `"strict"` and not `"opaque_255"`:** the pre-existing behaviour
accepts three lossless flavours (255, 0 and binary) and only refuses the undecidable
one. Defaulting the parameter to `"opaque_255"` would turn the mere act of adding a
parameter into a *tightening* — every `unused_0`/`binary_mixed` frame accepted today
would start to raise — which contradicts "默认行为一字不改". `"strict"` names the
existing policy exactly: whatever the strict classifier says, and nothing
intermediate. The record then stores the **actual verdict** (`opaque_255`,
`unused_0` or `binary_mixed`), so the three strict flavours stay distinguishable in
an audited dataset instead of collapsing into one name.

An explicit `"opaque_255"`/`"unused_0"`/`"binary_mixed"` is a **requirement**: a
frame that classifies as a different (still lossless) flavour is refused rather than
reclassified, because the caller asserted which buffer it was reading.

## 4. Semantics of `"any"` (as written into the module docstring)

1. The alpha channel is **still dropped and never blended**. `global_rgb`,
   `play_gray` and `hud_gray` are computed from the RGB channels alone; no alpha
   term exists anywhere in the arithmetic (the same `gray_bt601_int`,
   `downsample_2x2_int`, crop rectangles and key/history rules apply unchanged).
2. Therefore **any** alpha is accepted, including the real engine's blended
   202..219 band and premultiplied buffers the strict gate refuses.
3. **Cost:** the weak signal "this frame is translucent" is lost. The module offers
   **no** substitute: a caller using `"any"` **must** run its own
   placeholder/staleness detection (e.g. reject a frame whose `global_rgb` is
   all-zero or near-zero). The short-stack, duplicate-frame, uniform-RGB,
   history-valid-prefix, key-row, elapsed-interval and privileged-name gates all stay
   **active** in every mode — proven by
   `test_alpha_mode_any_does_not_relax_the_other_gates`.
4. No pre-multiply inverse, no upsampling, no zero-fill, no alpha composite: there
   is no conversion logic in this change, only a switch that skips an inspection.

## 5. Does `alpha_mode` enter `o2_digest`?

**No — and it does not need to, because it enters `encoder_digest` instead.**

- `o2_digest()` is the pass-through of the *contract's* parameter block and must
  stay byte-identical to `contracts.o2_digest()`
  (`cb1616e4cbfa928e691c2fb8497a84c813bded528064cf2b0aa1e30fa86b85b5`, unchanged by
  this work). The four crops/weights/rules do not depend on which gate a run chose,
  and an existing test pins `observation.o2_digest() == contracts.o2_digest()`.
- The provenance does enter the **encoder** identity:
  `encoder_parameters(alpha_mode=...)` / `encoder_digest(alpha_mode=...)` /
  `o2_identity(alpha_mode=...)` carry `alpha_mode_used` plus the accepted set, the
  default and the policy text. Measured:
  - strict: `37ebb8d40c45ee63d8e6d4d5f873cef00c9e34eff719af1343b8d8e49dc5a1aa`
  - `"any"`: `90e43af671e0c93bf333e26613345d5fe135e8ed8ae1c095b688bfd6993a54fc`
  So data captured through `"any"` **cannot** share an `encoder_digest` with
  strictly gated data — the property the task asked for — while the contract
  parameters keep one identity.
- Per decision, the record itself carries `alpha_mode`, which is the finest-grained
  audit trail (a mixed dataset can be split by it after the fact).

Consequence worth recording: the strict `encoder_digest` **changed** relative to
before this patch (`alpha_modes_accepted` grew by `"any"`, `encoder_steps[0]` and
`alpha_policy` now name the opt-in). No checkpoint or test pins the old literal
(`grep` over `touhou_rl/v430` and `tests/` finds no hardcoded encoder digest), and
the digest is self-consistently recomputed.

## 6. Tests added (9; the 47 existing ones are unchanged and still pass)

| # | test | what it pins |
| --- | --- | --- |
| 1 | `test_alpha_mode_values_and_default_stay_strict` | the four values, `"strict"` default outside `ALPHA_MODES` |
| 2 | `test_alpha_mode_default_still_refuses_intermediate_alpha` | no argument ⇒ alpha=128 still `FrameAlphaError` (also `rgba_to_rgb` default, also constant 200) |
| 3 | `test_alpha_mode_any_accepts_intermediate_alpha` | `"any"` + alpha=128 accepted; planes are the opaque ones |
| 4 | `test_alpha_mode_any_accepts_the_real_engine_blended_alpha_band` | the 202..219 band, per-column ramp + all 18 constant values |
| 5 | `test_alpha_mode_any_drops_alpha_bit_for_bit` | same RGB, alpha 255 vs 210 ⇒ `global_rgb`/`play_gray`/`hud_gray` **byte-identical** (`.tobytes()` and sha256) |
| 6 | `test_alpha_mode_refuses_an_unknown_value` | `"whatever"`, `"ANY"`, `"any "`, `""`, `None`, `255` ⇒ `ObservationError` naming `alpha_mode` (checked in `encode_o2`, `rgba_to_rgb`, `encoder_parameters`, `encoder_digest`) |
| 7 | `test_observation_records_the_alpha_mode_actually_used` | `obs.alpha_mode` ∈ {`opaque_255`,`unused_0`,`binary_mixed`,`any`}; `as_dict()` still six fields; minibatch/attach_memory unchanged |
| 8 | `test_alpha_mode_enters_the_encoder_identity_but_not_o2_digest` | `o2_digest` unchanged, `encoder_digest("any") != encoder_digest()`, identity records |
| 9 | `test_alpha_mode_any_does_not_relax_the_other_gates` | placeholder, duplicate, short stack and wrong shape still refused under `"any"` |

Test 4 construction (no engine started, synthetic numpy only): a `stack()` of four
marker-distinct `play_frame(90)` frames has its alpha plane set to
`202 + (np.arange(640) % 18)`, and the test asserts `min == 202`, `max == 219` and
that the unique values are exactly `{202..219}` **before** encoding, so the band is
demonstrably the measured one. The strict path is asserted to raise with both
`"202"` and `"219"` in the message (the real engine's error text); the `"any"` path is
asserted to accept it and to produce planes identical to the opaque reference.

## 7. Explicitly not done

- No engine launched, no `glReadPixels` call, no capture wired into
  `session.py`/`evaluate.py`: this slice is the **encoder contract only**. Passing
  `alpha_mode="any"` at the real call site is a separate change in files outside
  this task's whitelist (and `touhou_rl/v430/{session,evaluate}.py` do not call
  `encode_o2` today — `grep` over `touhou_rl/v430` finds `encode_o2` only inside
  `observation.py`).
- No corpus read or written; no training; no data touched; no asset move/add.
- No alpha un-premultiply, no upsampling, no zero-fill, no change to any other
  gate.
- `"any"` does not provide placeholder detection; a future `"any"` call site must
  add it (the `global_rgb` all/near-zero test) and record `alpha_mode` in its
  manifest.
