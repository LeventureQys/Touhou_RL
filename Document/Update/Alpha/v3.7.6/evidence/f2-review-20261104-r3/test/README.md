# Pending offline review pack

This pack is NOT accepted by f2_dataset build: pending schema, frozen=false,
null annotations, missing independent player/track measurements. Audit success
only establishes raw identity, never F2 presence or visual validity. Structural
mode is diagnosis only. No label, split, training or F2-pass claim is generated.

Open index.html locally (file://); PNGs stay at their original file URIs.
Do not move/change the raw dataset. asset-index.json binds source bytes by SHA.
Candidates are deterministic source-order windows, not detected F2 events.
Stride counts stage-filtered sample indices, not elapsed frames. Each row has
exact f-4, f-2, f images; missing past rejects the entire window.
All occurrences of a duplicated (stage,stageFrames) are excluded as ambiguous;
no first/last representative is chosen. Windows citing ambiguous current/past
frames are rejected; candidates.json records every excluded sample index.

Independent reviewer workflow:
1. Review original pixels first, independently from privileged audit-state.
   Fill observation.player_px (full image coordinates), bullet_tracks samples
   with frame/xy/visible=true/image_sha256. Images must remain <= current frame.
2. For a supported F2 aimed stream, replace annotation=null with the complete
   f2_labels.validate_event_annotation contract: observation_sha256 (computed
   AFTER observation editing), family=F2, event_type=aimed_stream, distinct
   reviewer.id/method=independent_pixel_annotation and teacher.producer_id,
   explicit tolerances, pixel_tracks, teacher.tracks, and emissions. Teacher
   artifact_sha256 is the original audit-state SHA. Pixel tracks must be measured
   from PNGs, NEVER copied from the memory file. Retain unknown/context_only for
   unresolved rows; bullet_tracks=[] is deliberately incomplete, not a negative.
3. Match independently measured pixel tracks to frame-local memory slot:N
   (e.g. slot:17). Slot reuse is NOT a persistent lifetime. Use multiple distinct
   slot IDs for distinct emitted bullets when applicable; independently document
   which samples belong to each visible lifecycle. Do not concatenate recycled
   slot lifetimes to invent motion. Current validator requires slot:N for teacher
   tracks; unsupported/reused identity ambiguity stays unknown and is handed off.
   Prove at least two distinct visible emissions from the SAME emitter: emission
   frame/track_id/emitter_id, origin_px, velocity_px_per_frame derived from two
   visible samples, causal aim_player_frame/aim_image_sha256/aim_player_px and
   independent player_pixel_annotation. Every cited frame must be in that row's
   causal images (extend ONLY with exact past/current images, never future).
4. Privileged audit-state supplies game-space teacher coordinates; convert to
   full native PNG coordinates (+32,+16 per current f2_dataset contract). Its
   references/player hints are explicitly memory audit, not visual annotations.
5. A separate reviewer must check the independent evidence and bindings. Export
   each selected event as its own {frozen:true, rows:[...]} review only AFTER
   review, alongside a separately preregistered/frozen multi-source plan. Do not
   merely flip frozen: this template intentionally lacks valid observations and
   annotations. Event selection, split assignment and final freeze are a separate
   handoff; no acceptance or independence is asserted by this generator.
