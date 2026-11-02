# 镜像：研究用帧栅栏变体构建说明（快照）

- 快照日期：2026-11-02
- **来源**：`third_party/th08/docs/FRAME_FENCE.md`（git submodule，子模块 HEAD `08c7f00`）。
- **为什么要镜像**：该文件在子模块里**尚未提交**（`?? docs/FRAME_FENCE.md`），子模块指针指向的
  commit 里没有它，所以外部读者通过 GitHub 看不到。为了让"同帧证明"这项已达成能力在仓库内
  可复现，这里原样快照一份；**正文未做任何修改**。
- **权威副本仍是子模块里的那份**：若两者不一致，以子模块 + 实际构建为准，并应重新生成本快照。
- 相关：采集报告 [P0-frame-fence-round1.md](<reports/P0-frame-fence-round1.md>)、读取端实现
  [touhou_rl/framefence.py](<../../../../touhou_rl/framefence.py>)、能力边界
  [DATA-ACQUISITION.md](<../../../../DATA-ACQUISITION.md>)。

---

# Research-only frame fence variant (`--build-type fence`)

**Status: research variant. It is not a reconstruction artifact, it backs no
exact-match claim, and its bytes are not the canonical target image.** The
canonical `bugfix` image is unchanged by anything described here; see
[Verify the canonical image is untouched](#verify-the-canonical-image-is-untouched).

## Why this exists

`DATA-ACQUISITION.md` (repo root) records the open problem: the GDI capture path
has no present fence, so a screenshot and a `ReadProcessMemory` state read cannot
be *proven* to describe the same drawn frame, which is why precise geometric
labels stay frozen.  The relevant fact is in `GameWindow::Render`
(`src/main.cpp`): the calculation chain advances the logic frame, the draw chain
runs afterwards, `Present()` follows, and only then is the frame on screen.  A
read-only observer outside the process has no synchronization point.

This variant adds one: a **frame fence** published from inside the game, at the
end of the present, that a reader can hold and verify.  It is deliberately built
as a separate image so the canonical reconstruction keeps its exact bytes.

## What the fence publishes

`src/framefence.cpp`, compiled only when `TH08_FRAME_FENCE` is defined, creates
the named shared section `Local\th08.frame_fence.v1` on the first drawn frame:

| offset | contents | writer |
|---|---|---|
| `0x000` | control: enable flags, hold policy (`holdEvery`, `holdMinUs`, `holdMaxUs`), reader acknowledgement, reset request | reader |
| `0x020` | telemetry: magic/version/size, pid, ticket, logic frame, stage/run frames, player position/velocity/state/focus, inputs, active bullet count, bullet + enemy FNV-1a digests, digest ticket, hold phase/counters, pixel ticket/geometry, QPC stamp | game |
| `0x1000` | `640*480*3` canonical RGB24 copy of the presented back buffer | game |

Per drawn present the game fills the telemetry, computes the digests, bumps
`ticket`, and — when the reader asked for holds — sets `holdPhase = 1` and spins
until the reader acknowledges that ticket or `holdMaxUs` expires.  While
`holdPhase == 1` no logic frame can advance and no new present can occur.

A reader that observes `ticket == T` together with `holdPhase == 1` therefore
knows: (a) every telemetry field belongs to `T`; (b) the game is stopped; (c) a
`BitBlt` and a `ReadProcessMemory` issued now describe the same drawn frame; and
(d) the digests let it *verify* that its own bulk reads (6.3 MB bullet pool,
10 MB enemy array) are bit-identical to what the game published for `T`.
`FencePixels` additionally lets it compare its capture against the pixels the
game actually drew.

The game never writes game state from this file, and the reader never writes to
the game process: the acknowledgement travels through the shared section.

## Build

```bash
python scripts/build.py --build-type fence -j 1     # -> build_fence/th08.exe
python scripts/fence-addresses.py build_fence/th08.map build_fence/frame-fence-addresses.json
```

* `build_fence/` is git-ignored.  Nothing under `build/` is touched.
* The variant is compiled with `/DFIX_REALLY_BAD_BUGS` (same as `bugfix`) plus
  `/DTH08_FRAME_FENCE`.  **A `normal` build is non-functional** (the
  `th08_0100d.ver` identity path aborts at boot), so the fence must never be
  layered onto `normal`.
* Adding a translation unit grows `.text`/`.rdata` and therefore moves every
  global by a page multiple (observed `+0x3000`).  Struct *offsets* are
  unchanged, but image *addresses* are not: readers must load the generated
  address table (`TH08_ADDRESS_FILE`, see `touhou_rl/spec.py`).
  `scripts/fence-addresses.py` reads the linker map that the build already
  produces; run against the canonical `build/th08.map` it reproduces the
  hard-coded addresses in `touhou_rl/spec.py` exactly.

## Protocol for a reader

1. `OpenFileMapping`/`MapViewOfFile` the section, validate magic/version/size and
   that `pid` is the session you expect (stale sections from a dead process are
   detectable this way).
2. Enable publishing (`flags = ENABLE`), `holdEvery = 0`, then read `ticket`.
   Acknowledgement is only meaningful once a ticket exists, so acknowledge one
   ordinary ticket first (`ackTicket = ticket`).
3. Enable holds (`holdEvery = n`, `holdMaxUs` = frame budget you can afford).
4. Wait for `ticket != last` **and** `holdPhase == 1`.
5. Read telemetry, capture pixels (`BitBlt`), read state, recompute the digests.
6. Re-check `ticket == T && holdPhase == 1`; if it still holds, the sample is
   exact, then write `ackTicket = T`.  If the hold expired, discard and retry —
   never accept a sample whose hold ended mid-read.
7. Validate: player bits, gameplay frame, stage frames, input mask, active
   bullets, bullet digest, enemy digest, `pixels_frozen` (two captures equal) and
   `pixels_match_backbuffer`.

`touhou_rl/framefence.py` implements exactly this (`FrameFence`,
`FenceFrameSampler`, `FenceStateDigest`, `legacy_sample`).

## Kill switches and diagnostics

* An SEH guard around the publication path disables the fence and logs instead of
  ever taking the game down.
* A first-chance vectored exception handler logs `eip`/`esp`/`ebp` to
  `framefence.log` (resolvable against the linker map).
* `Step()` breadcrumbs are stored in the telemetry `flags` field while a frame is
  being published, so a crash leaves the last completed step visible.

## Verify the canonical image is untouched

```bash
python scripts/build.py --build-type bugfix -j 1
# .text must hash to 7B5B3C8AB5023BFA7960DA309338A01982F651DC6B1E7734BEE92437202ED6DD
# (same value as tools/th08/Th08Pipeline.psm1 Get-Th08CodeHash on the shipped image)
```

The whole-file SHA of a fresh link differs only in the PE `TimeDateStamp`
(3 bytes at file offsets `0x118..0x11A`); the canonical artifact shipped in
`build/play/th08.exe` still hashes to
`CDD23BCA0C724941866EFDA5DA280BE67EB6CDD935F83548045C2A8E037F1DDF`.  The
`#ifdef TH08_FRAME_FENCE` blocks in `src/main.cpp` are compiled out of every
canonical build, so code generation is bit-identical.

## Measured (2026-11-02, this host)

240 held frames in the stage-5 attract demo: 240/240 frames passed all 13
same-frame checks, 240/240 GDI captures were byte-identical to the published back
buffer, and the game still presented at 59.87 Hz (hold p50 7.44 ms, max 8.97 ms,
0 expired holds).  Results and evidence: `Document/Update/Alpha/v3.7.4/`.
