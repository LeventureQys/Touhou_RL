#!/usr/bin/env python3
"""v8.29 native 4-frame source collector -- the training host makes its own ancestors.

为什么要这个文件
================
`v8.29` 的 pair/trace 只能从**已有录像**里取前缀：`v450_cf_label.masks_for` 把
`target = N + 1` 交给 `v431_dense_driver.build_masks`，后者只认 S1c 的四个 window
（`.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed<w>-shard/labels.jsonl`）。也就是说，
新的祖先要么来自旧语料（已被 `configs/v430/splits.json` 的 lineage 记账），要么没有。
本文件补的就是这一块：**在本机现场采一段 4 帧一决策的 tape**，格式与真实 tape
逐字段一致，于是 `v829_pair.read_tape` / `--prefix-tape` / `v829_manifest_build`
不需要任何改动就能把它当作祖先前缀。

本轮**不启动引擎**：真实采集由主代理执行，本文件只交付可离线验证的部分。
所有验收都在 WSL `/root/v400-venv/bin/python` = Python 3.10.12 上做（Python 3.10
是本项目真实运行解释器，见 `v829_pair.py` 的 ``--workers`` 注释）。

Tape 格式（逐字段与真实 tape 对照）
===================================
真实 tape 首行实测
``.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed11-shard/labels.jsonl``::

    {"kind": "decision", ..., "decision": 0, "frame": 0, "decision_frame": 0,
     "frames_per_decision": 4, "decision_frames": 2, "stage": 0, "difficulty": 0,
     "hash_pre": "7c8705fc88f94c17", "hash_post": "85f4f78c96ba7684",
     "step_frames": [65, 65, 65, 65], ...}

本文件写的行（键顺序即写出顺序）::

    {"kind": "decision", "schema": "v829-source-tape/1", "decision": 0,
     "frame": 0, "decision_frame": 0, "frames_per_decision": 4,
     "stage": 0, "difficulty": 1, "step_frames": [1, 1, 1, 1],
     "hash_pre": "<frame 0 的 16 hex>", "hash_post": "<frame 4 的 16 hex>",
     "frame_end": 4, "policy": "stay", "source": "tools/rl/v829_collect_source.py"}

`v829_pair.read_tape` 只要求（`v829_pair.py:2012-2155`）：

1. 每行 4 个 `step_frames`（= 真实 tape 的四个逐帧掩码，`step_frames` 就是 wire 掩码，
   与 `v431_dense_driver.build_masks` 的 ``masks`` 完全同义）；
2. `decision_frame` 从 0 起、步长 4、无重复/无空洞；
3. 每行 `hash_pre` / `hash_post` 齐全，锚点集合 = ``{0, 4, ..., 4*rows}``；
4. ``target`` 是 4 的倍数且在 ``hashes`` 里。

`frames_per_decision` / `stage` / `difficulty` 是格式要求（本文件必写，`verify` 必查），
`read_tape` 不看它们。``frame`` 的语义与真实 tape 相同：**决策起始帧**
（决策 0 = 帧 0），块末帧另记在 ``frame_end``。

子命令
======
``collect``   真实采集（本轮不跑）：HELLO -> RESET -> blocks x 4 x STEP -> labels.jsonl
              + shard.json + collect.json。``--fake-parent`` 用确定性假 parent 走**同一
              条**代码路径（无引擎），只允许写到 ``.tmp_v829_info/`` 下。
``verify``    不启引擎，用与 `v829_pair.read_tape` 等价的逻辑读 tape（更严格）并逐条
              打印检查结果；全过 exit 0，否则打印第几条不合规并 exit 1。
``selftest``  不启引擎：跑 12 块（48 帧）离线端到端 + 三种注入故障 + 路径围栏 + 音频闸门，
              打印 ``SELFTEST: ok N/N``。

采集循环（核心，与工单伪代码一致）
==================================
::

    parent.start()                      # 真机：HarnessClient(role="shadow")
    f, _ = parent.reset(seed)           # prev_hash = f["hash"]  (= 帧 0)
    for block in range(blocks):
        masks = [policy(4*block + i) for i in range(4)]
        for m in masks: st = parent.step(m)      # 每步必须 advanced==1
        row = {decision_frame: 4*block, step_frames: masks,
               hash_pre: prev_hash, hash_post: st["hash"],
               frames_per_decision: 4, stage: S, difficulty: D,
               decision: block, frame: <决策起始帧>, frame_end: st["frame"]}
        prev_hash = st["hash"]
        if st["exited"] or st["game_over"]: 记录 death_frame 并 break

对伪代码的两处**显式**偏离，都写进报告：

* ``frame`` 记的是**决策起始帧**（真实 tape 的语义；决策 0 = 0），块末帧另记
  ``frame_end``——伪代码里的 ``st["frame"]`` 是块末帧，两个值都保留会丢掉信息；
* 死亡步允许 ``advanced=0``：`v432_fork_arm.child_trace`（`:686-691`）的冻结写法就是
  ``advanced == 1`` 记帧、``elif not (exited or game_over)`` 才报错。**活着的步**一律
  必须 ``advanced == 1``，否则 ``STEP_NOT_ADVANCED``。

策略（``--policy``）
====================
``stay``          常量 cand 0（不按方向、仍射击），mask = `contracts.cand_to_mask(0)` = 1；
                  默认值，采集时会打印一行提醒（原地不动的祖先信息量很低）。
``const:N``      常量 cand N（0..17），mask = `cand_to_mask(N)`。
``random``       `random.Random(--seed)` 逐帧抽 cand，可复现。
``policy:CKPT``  `v860_bombdag.load_policy36` + `v830_bombpol.sample_action36`
                 （stochastic，36 动作 = cand*2+bomb，bomb 位只打在决策首帧，
                 与 `v830.run_episode36` 的 emit 规则一致）；特征来自
                 `v431_arm_a.build_features`，状态/几何来自 parent 的
                 ``STATE`` / ``HAZARDS``；torch.Generator 种子 = ``seed*1000+7``
                 （`v8r0_urgency.generator_seed` 的冻结族公式，GEN_CONST=7）。

失败即关闭（每条一个错误码，exit 码见下表）
==========================================
=============================  ====  ============================================
code                           exit  触发条件
=============================  ====  ============================================
``OUT_DIR_FORBIDDEN``          1     ``--out-dir``/``--work-dir`` 落在
                                     ``.tmp_v400_*`` / ``.tmp_v410_*`` /
                                     ``.tmp_v829_pair`` / ``Document`` / 源码树，
                                     或 ``--fake-parent`` 的输出不在
                                     ``.tmp_v829_info/`` 下
``OUT_DIR_NOT_EMPTY``          1     目录已存在且非空（``--overwrite-empty`` 只放过
                                     "只有本工具三个产物且上一轮不是 ok" 的残骸）
``ENGINE_MISSING``             2     引擎二进制不是文件，或 ``--data-source`` 不是目录
``ENGINE_SHA_MISMATCH``        1     引擎 sha256 != ``--engine-sha256``（默认 691bfd77…）
``HELLO_FAILED``               1     HELLO 不是 ``OK role=shadow``（假 parent 无 hello 时跳过）
``RESET_FAILED``               1     RESET 抛错 / 不是 OK / 回复里没有合法 ``hash``
``STEP_NOT_ADVANCED``          1     活着的步 ``advanced != 1``；或四帧都 advanced 但
                                     ``hash`` 没变/与更早锚点重复（链断）；或帧号回退
``HASH_MISSING``               1     STEP（非死亡）没有 hash，或 hash 不是 8..64 位小写 hex
``AUDIO_NOT_MUTED``            1     私有 data 目录的 ``th08.cfg`` 四个字段不全为 0
                                     （跑前、跑后各查一次）
``TAG_REQUIRED``               1     ``--tag`` 给了但不是合法 slug（``[A-Za-z0-9][…]{2,63}``）
``TAG_NOT_UNIQUE``             1     ``--tag`` 已在本 work-dir 的 ledger 里绑定到**另一个**
                                     cell（同 cell 重跑放行）
``OK_UNEXPECTED``              2     兜底：未分类异常、缺 checkpoint、写完的 tape 过不了
                                     本工具自己的 ``verify``、内部不变量破裂
=============================  ====  ============================================

写入围栏（AGENTS.md §8.2）
==========================
只往两处写：``--out-dir``（三个产物）与 ``--work-dir``（``data/<tag>`` 私有 data 目录、
``logs/<tag>``、``tags.jsonl`` ledger）。历史证据树（``.tmp_v400_*``、``.tmp_v410_*``、
``.tmp_v829_pair``、``Document/``）与源码树一律拒绝（``OUT_DIR_FORBIDDEN``）。本轮默认
work-dir 是 ``.tmp_v829_info/s15/work``。

音频（AGENTS.md §9，全局硬约束）
================================
三层，全部记账：`v450_cf_label.prepare_env()` 设 ``SDL_AUDIODRIVER=dummy`` 并摘掉
``DISPLAY``/``WAYLAND_DISPLAY``；`v432_fork_arm.build_private_data_dir` 每次启动前
用 `tools/rl/v400_make_cfg.py` 重新生成 60 字节 ``th08.cfg`` 并复核四个字段为 0；
本文件再用 `v431_audio_cfg_audit.decode` 独立解码一次（跑前 + 跑后），四个字段
（musicMode / playSounds / musicVolume / sfxVolume）必须全 0，否则 ``AUDIO_NOT_MUTED``。
`collect.json.audio` 存的就是这份证据（四字段 + ``all_muted`` + 两次的摘要）。

用法
====
::

    # 真机（主代理执行；本轮不跑）
    python -B tools/rl/v829_collect_source.py collect \
        --out-dir .tmp_v829_source/s3101 --seed 3101 --stage 0 --difficulty 1 \
        --blocks 600 --policy policy:.tmp_v829_pair/ckpt/s3101.pt --tag src-s3101

    # 离线（本文件的自测路径，无引擎）
    python -B tools/rl/v829_collect_source.py collect --fake-parent \
        --out-dir .tmp_v829_info/s15/e2e12 --seed 77101 --stage 0 --difficulty 1 \
        --blocks 12 --policy random
    python -B tools/rl/v829_collect_source.py verify \
        --tape .tmp_v829_info/s15/e2e12/labels.jsonl --expect-rows 12
    python -B tools/rl/v829_collect_source.py selftest

本轮**没有**验证的部分（报告里也必须这么写）
==========================================
引擎没起过一次，所以 ``HELLO`` 的真实回复、``RESET`` 的帧号字段、真机 ``STEP``
的 ``advanced``/``hash`` 行为、``policy:`` 分支在真状态上的采样，全部只有代码路径与
离线假 parent 的覆盖，没有真机证据。真机首跑的断言清单见 REPORT.md。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import re
import sys
import time
import traceback
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:                                      # noqa: SIM103
    sys.path.insert(0, str(REPO))

TOOL = "tools/rl/v829_collect_source.py"
TOOL_VERSION = "v8.29.0"
PRODUCER = f"{TOOL} ({TOOL_VERSION})"

#: One recorded hash anchor every 4 frames (`v829_pair.ANCHOR_STRIDE`,
#: `v431_dense_driver.build_masks`); the engine's own decision span.
FRAMES_PER_DECISION = 4
ANCHOR_STRIDE = 4
#: `touhou_rl.v420.contracts.N_MOVE_FOCUS` (frozen 18-candidate table).
N_CAND = 18
#: `v8r0_urgency.GEN_CONST` / `v8o0:147` / `v830:259` -- all three are 7.
GEN_CONST = 7

#: The rl_v3 build every pinned runner starts (`v432:185`, `v450:123`,
#: `v829_pair:478`, `ASSET_SYNC.md`).
ENGINE_DEFAULT = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern"
ENGINE_SHA256 = "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6"
DATA_SOURCE_DEFAULT = "/root/v400-data/game"
#: This round may only write under ``.tmp_v829_info/s15``; the default work dir
#: is inside it so a smoke run cannot touch anybody else's evidence tree.
WORK_DEFAULT = REPO / ".tmp_v829_info" / "s15" / "work"

SCHEMA_TAPE = "v829-source-tape/1"
SCHEMA_SHARD = "v829-source-shard/1"
SCHEMA_COLLECT = "v829-source-collect/1"
SCHEMA_VERIFY = "v829-source-verify/1"

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_UNUSABLE = 2

#: The whole fail-closed table of this file: one code, one exit status.
CODE_EXIT: dict[str, int] = {
    "OUT_DIR_FORBIDDEN": EXIT_REFUSED,
    "OUT_DIR_NOT_EMPTY": EXIT_REFUSED,
    "ENGINE_MISSING": EXIT_UNUSABLE,
    "ENGINE_SHA_MISMATCH": EXIT_REFUSED,
    "HELLO_FAILED": EXIT_REFUSED,
    "RESET_FAILED": EXIT_REFUSED,
    "STEP_NOT_ADVANCED": EXIT_REFUSED,
    "HASH_MISSING": EXIT_REFUSED,
    "AUDIO_NOT_MUTED": EXIT_REFUSED,
    "TAG_REQUIRED": EXIT_REFUSED,
    "TAG_NOT_UNIQUE": EXIT_REFUSED,
    "OK_UNEXPECTED": EXIT_UNUSABLE,
}

#: Engine state hash as the harness emits it: FNV-1a over the 25 tracked fields,
#: printed as 16 lowercase hex digits (`rlharness.cpp:170-179`).  The width is
#: checked loosely so a future build with a wider digest is still readable.
HASH_RE = re.compile(r"^[0-9a-f]{8,64}$")
#: A tag is a path-free slug and it is also the private data dir's name.
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")

#: Historical evidence trees + source trees: never a write target (AGENTS.md §8.2).
FORBIDDEN_PREFIXES: tuple[str, ...] = (".tmp_v400_", ".tmp_v410_")
FORBIDDEN_NAMES: tuple[str, ...] = (
    ".tmp_v829_pair", "Document", ".git", "tools", "touhou_rl", "configs",
    "tests", "third_party", "third_party_data",
)
#: The only directory a synthetic (``--fake-parent``) tape may land in.
FAKE_PARENT_ROOT_NAME = ".tmp_v829_info"
#: Entries this tool owns inside ``--out-dir``.
KNOWN_ARTIFACTS: tuple[str, ...] = ("labels.jsonl", "shard.json", "collect.json")

LOG_PREFIX = "[v829src]"
_PROGRESS_EVERY = 50
_CACHE: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# errors / logging / small helpers
# --------------------------------------------------------------------------- #
class CollectError(RuntimeError):
    """A fail-closed refusal.  ``code`` is stable, greppable and mapped to an exit."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"[{code}] {detail}")
        self.code = str(code)
        self.detail = str(detail)

    @property
    def exit_code(self) -> int:
        return int(CODE_EXIT.get(self.code, EXIT_REFUSED))


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def load_module(name: str, rel: str):
    """Import a ``tools/rl`` script by path (never registered in ``sys.modules``)."""

    module = _CACHE.get(name)
    if module is not None:
        return module
    path = REPO / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CollectError("OK_UNEXPECTED", f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CACHE[name] = module
    return module


def m432():
    return load_module("v432_fork_arm", "tools/rl/v432_fork_arm.py")


def m450():
    return load_module("v450_cf_label", "tools/rl/v450_cf_label.py")


def m830():
    return load_module("v830_bombpol", "tools/rl/v830_bombpol.py")


def m860():
    return load_module("v860_bombdag", "tools/rl/v860_bombdag.py")


def m431a():
    return load_module("v431_arm_a", "tools/rl/v431_arm_a.py")


def m_audio():
    return load_module("v431_audio_cfg_audit", "tools/rl/v431_audio_cfg_audit.py")


def m_cfg():
    return load_module("v400_make_cfg", "tools/rl/v400_make_cfg.py")


def m829():
    return load_module("v829_pair", "tools/rl/v829_pair.py")


def _contracts():
    from touhou_rl.v420 import contracts as C

    return C


def reply_fields(reply: str) -> dict[str, str]:
    """``key=value`` tokens of one protocol line (same split as `v432.fields`).

    Inlined on purpose: the offline path (``--fake-parent`` / ``selftest``) must
    not import the engine-side module just to split a reply string.
    """

    out: dict[str, str] = {}
    for part in str(reply).split()[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value), 10)
    except (TypeError, ValueError):
        return None


def _torch_version() -> dict[str, object]:
    """The torch this process would use (imported: the real run needs it anyway)."""

    try:
        import torch
    except Exception as exc:                                      # noqa: BLE001
        return {"imported": False, "version": None,
                "detail": f"{type(exc).__name__}: {exc}"}
    return {"imported": True, "version": str(torch.__version__),
            "cuda_available": bool(getattr(torch.cuda, "is_available", lambda: False)()),
            "file": str(getattr(torch, "__file__", ""))}


def _torch_probe() -> dict[str, object]:
    """Cheap (no-import) torch probe for the paths that must not pay for torch."""

    try:
        available = importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        available = False
    return {"imported": False, "available": available, "version": None,
            "note": ("torch is only imported when the run needs it (the fake parent "
                     "and the tag/fence refusals do not); the real path records "
                     "collect.json.torch.version from the interpreter that ran")}


# --------------------------------------------------------------------------- #
# write confinement
# --------------------------------------------------------------------------- #
def forbidden_part(path: pathlib.Path) -> str | None:
    """The first path component that names a historical/source tree, else ``None``."""

    for part in pathlib.Path(path).parts:
        if part in FORBIDDEN_NAMES:
            return part
        for prefix in FORBIDDEN_PREFIXES:
            if part.startswith(prefix):
                return part
    return None


def resolve_write_dir(value: object, *, what: str) -> pathlib.Path:
    """Absolute path of a directory this tool may create, or ``OUT_DIR_FORBIDDEN``."""

    raw = str(value).strip()
    if not raw:
        raise CollectError("OUT_DIR_FORBIDDEN", f"{what} is empty")
    path = pathlib.Path(raw).expanduser()
    path = (path if path.is_absolute() else pathlib.Path.cwd() / path)
    path = path.resolve()
    if not path.parts or path == pathlib.Path(path.anchor):
        raise CollectError("OUT_DIR_FORBIDDEN", f"{what} {path} is a filesystem root")
    if path == REPO.resolve():
        raise CollectError("OUT_DIR_FORBIDDEN",
                           f"{what} is the repository root itself; this tool writes "
                           f"only inside a fresh run directory")
    bad = forbidden_part(path)
    if bad is not None:
        raise CollectError(
            "OUT_DIR_FORBIDDEN",
            f"{what} {path} is inside {bad!r}, which is a historical evidence tree or "
            f"a source tree (AGENTS.md §8.2: never rewrite somebody's evidence). "
            f"Use a fresh path such as {REPO / '.tmp_v829_source'}/<cell>")
    return path


def is_under(path: pathlib.Path, root: pathlib.Path) -> bool:
    target = pathlib.Path(path).resolve()
    base = pathlib.Path(root).resolve()
    return target == base or base in target.parents


def assert_out_dir_usable(out_dir: pathlib.Path, *, overwrite_empty: bool) -> dict:
    """``OUT_DIR_NOT_EMPTY`` unless the directory is absent/empty or a failed leftover."""

    out_dir = pathlib.Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise CollectError("OUT_DIR_NOT_EMPTY",
                           f"{out_dir} exists and is not a directory")
    if not out_dir.is_dir():
        return {"state": "absent", "entries": []}
    entries = sorted(entry.name for entry in out_dir.iterdir())
    if not entries:
        return {"state": "empty", "entries": []}
    if overwrite_empty and set(entries) <= set(KNOWN_ARTIFACTS):
        prior = out_dir / "collect.json"
        status = None
        if prior.is_file():
            try:
                document = json.loads(prior.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                document = None
            status = document.get("status") if isinstance(document, dict) else None
        if status == "ok":
            raise CollectError(
                "OUT_DIR_NOT_EMPTY",
                f"{out_dir} holds a finished tape (collect.json status=ok); "
                f"--overwrite-empty never overwrites complete evidence -- pick a new "
                f"--out-dir or a new --tag")
        return {"state": "leftover", "entries": entries,
                "prior_status": status, "overwrite_empty": True}
    raise CollectError(
        "OUT_DIR_NOT_EMPTY",
        f"{out_dir} is not empty ({entries[:6]}{'...' if len(entries) > 6 else ''}); "
        f"refusing to write a tape over it. Use a fresh --out-dir, or "
        f"--overwrite-empty when the only entries are {list(KNOWN_ARTIFACTS)} from an "
        f"earlier FAILED run")


# --------------------------------------------------------------------------- #
# tags
# --------------------------------------------------------------------------- #
def policy_key(spec: dict) -> str:
    kind = str(spec.get("kind"))
    if kind == "stay":
        return "stay"
    if kind == "random":
        return "random"
    if kind == "const":
        return f"c{int(spec['cand'])}"
    digest = hashlib.sha256(str(spec.get("spec")).encode("utf-8")).hexdigest()[:6]
    return f"ckpt{digest}"


def default_tag(*, seed: int, stage: int, difficulty: int, blocks: int,
                spec: dict, out_dir: pathlib.Path) -> str:
    """A tag that is unique per cell **and** per output directory."""

    digest = hashlib.sha256(str(out_dir).encode("utf-8")).hexdigest()[:8]
    return (f"v829c-s{int(seed)}-st{int(stage)}-d{int(difficulty)}"
            f"-b{int(blocks)}-{policy_key(spec)}-{digest}")


def validate_tag(tag: object) -> str:
    text = str(tag or "").strip()
    if not TAG_RE.match(text):
        raise CollectError(
            "TAG_REQUIRED",
            f"--tag {text!r} is not a usable tag: it must match "
            f"{TAG_RE.pattern} (3..64 chars, letters/digits/._-, must start "
            f"alphanumeric). The tag names the private data dir and the ledger "
            f"cell, so it has to be path-free and non-empty")
    return text


def tag_ledger_path(work: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(work) / "tags.jsonl"


def load_tag_ledger(work: pathlib.Path) -> list[dict]:
    path = tag_ledger_path(work)
    if not path.is_file():
        return []
    entries: list[dict] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CollectError(
                "OK_UNEXPECTED",
                f"{path}: line {index + 1} is not JSON ({exc}); the tag ledger is "
                f"this tool's own file, so a corrupt one means the unique-tag rule "
                f"cannot be evaluated") from exc
        if isinstance(row, dict):
            entries.append(row)
    return entries


def claim_tag(work: pathlib.Path, *, tag: str, cell: dict) -> dict:
    """Register ``tag`` for ``cell``; ``TAG_NOT_UNIQUE`` if it names another cell."""

    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    ledger = load_tag_ledger(work)
    for entry in ledger:
        if str(entry.get("tag")) != tag:
            continue
        if entry.get("cell") != cell:
            raise CollectError(
                "TAG_NOT_UNIQUE",
                f"tag {tag!r} is already bound to another cell in "
                f"{tag_ledger_path(work)}: "
                f"{json.dumps(entry.get('cell'), ensure_ascii=False, sort_keys=True)} "
                f"!= {json.dumps(cell, ensure_ascii=False, sort_keys=True)}. A tag "
                f"names exactly one tape; pass a new --tag")
        return {"tag": tag, "ledger": str(tag_ledger_path(work)),
                "entries": len(ledger), "reused_same_cell": True}
    record = {"tag": tag, "cell": cell, "claimed_at": now_iso(), "pid": os.getpid(),
              "producer": PRODUCER}
    with tag_ledger_path(work).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {"tag": tag, "ledger": str(tag_ledger_path(work)),
            "entries": len(ledger), "reused_same_cell": False}


# --------------------------------------------------------------------------- #
# engine / data dir / audio
# --------------------------------------------------------------------------- #
def engine_gate(path: object, expected: str) -> dict:
    """``ENGINE_MISSING`` (exit 2) / ``ENGINE_SHA_MISMATCH`` (exit 1)."""

    engine = pathlib.Path(str(path)).expanduser()
    if not engine.is_file():
        raise CollectError("ENGINE_MISSING",
                           f"--engine {engine} is not a file on this host, so no "
                           f"frame can be produced (exit 2: cannot evaluate)")
    digest = sha256_file(engine)
    wanted = str(expected or "").strip().lower()
    gate = {"path": str(engine), "bytes": engine.stat().st_size, "sha256": digest,
            "sha256_expected": wanted or None, "match": True}
    if wanted and digest != wanted:
        raise CollectError(
            "ENGINE_SHA_MISMATCH",
            f"--engine {engine} digests to {digest} but --engine-sha256 is {wanted}. "
            f"The frozen rl_v3 build is {ENGINE_SHA256}; a different binary is a "
            f"different tape (recorded prefix hashes would not transfer)")
    return gate


def data_source_gate(path: object) -> dict:
    source = pathlib.Path(str(path)).expanduser()
    if not source.is_dir():
        raise CollectError(
            "ENGINE_MISSING",
            f"--data-source {source} is not a directory; the private engine data "
            f"dir cannot be built, so the engine cannot start at all (exit 2)")
    return {"data_source": str(source)}


def audio_evidence(cfg_path: pathlib.Path) -> dict:
    """Decode the 60-byte ``th08.cfg`` and insist on four zeros (AGENTS.md §9)."""

    cfg_path = pathlib.Path(cfg_path)
    raw = cfg_path.read_bytes()
    decoded = m_audio().decode(cfg_path) or {}
    fields = {key: decoded.get(key) for key in
              ("musicMode", "playSounds", "musicVolume", "sfxVolume")}
    all_muted = bool(
        cfg_path.stat().st_size == 60
        and all(isinstance(value, int) and not isinstance(value, bool) and value == 0
                for value in fields.values()))
    return {
        "path": str(cfg_path), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "fields": fields,
        "all_muted": all_muted,
        "decoder": "tools/rl/v431_audio_cfg_audit.py::decode (offset 0x1F/0x20/0x27/0x28)",
        "rule": ("musicMode == playSounds == musicVolume == sfxVolume == 0, "
                 "file size 60 B (AGENTS.md §9.3)"),
    }


def assert_audio_muted(evidence: dict, *, when: str) -> dict:
    if not evidence.get("all_muted"):
        raise CollectError(
            "AUDIO_NOT_MUTED",
            f"{when}: {evidence.get('path')} decodes to "
            f"{json.dumps(evidence.get('fields'), ensure_ascii=False)} "
            f"({'size ' + str(evidence.get('bytes')) + ' B'}), which is not the "
            f"four-zero muted config AGENTS.md §9.3 requires. Refusing to run the "
            f"engine audible")
    return evidence


# --------------------------------------------------------------------------- #
# policies (the per-frame mask function)
# --------------------------------------------------------------------------- #
def policy_spec_arg(text: str) -> dict:
    """argparse ``type=`` for ``--policy`` (syntax errors become usage exit 2)."""

    raw = str(text).strip()
    if raw == "stay":
        return {"spec": "stay", "kind": "stay", "cand": 0}
    if raw == "random":
        return {"spec": "random", "kind": "random"}
    if raw.startswith("const:"):
        value = raw.split(":", 1)[1].strip()
        try:
            cand = int(value, 10)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"const: needs an integer candidate, got {value!r}") from exc
        if not 0 <= cand < N_CAND:
            raise argparse.ArgumentTypeError(
                f"const:{cand} is outside the frozen candidate range [0,{N_CAND})")
        return {"spec": f"const:{cand}", "kind": "const", "cand": cand}
    if raw.startswith("policy:"):
        ckpt = raw.split(":", 1)[1].strip()
        if not ckpt:
            raise argparse.ArgumentTypeError(
                "policy: needs a checkpoint path, e.g. policy:.tmp_v829_pair/ckpt/s3101.pt")
        return {"spec": raw, "kind": "policy", "ckpt": ckpt}
    raise argparse.ArgumentTypeError(
        f"unknown policy {raw!r}: expected policy:<ckpt> | random | stay | const:N")


class _ConstPolicy:
    """One fixed candidate on every frame (``stay`` is ``const:0``)."""

    kind = "const"

    def __init__(self, cand: int, *, kind: str | None = None) -> None:
        self.cand = int(cand)
        if not 0 <= self.cand < N_CAND:
            raise CollectError("OK_UNEXPECTED", f"cand {self.cand} outside [0,{N_CAND})")
        if kind:
            self.kind = str(kind)
        self.mask = int(_contracts().cand_to_mask(self.cand))
        self.counts = [0] * N_CAND
        self.frames = 0

    def bind(self, parent: object) -> None:
        self.parent = parent

    def __call__(self, frame_index: int) -> int:
        self.counts[self.cand] += 1
        self.frames += 1
        return self.mask

    def audit(self) -> dict:
        return {"kind": self.kind, "cand": self.cand, "mask": self.mask,
                "cand_counts": list(self.counts), "bomb_count": 0,
                "masks_sent": self.frames}


class _RandomPolicy:
    """``random.Random(--seed)`` candidate per frame -- reproducible, no torch."""

    kind = "random"

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(int(seed))
        self.seed = int(seed)
        self.counts = [0] * N_CAND
        self.frames = 0

    def bind(self, parent: object) -> None:
        self.parent = parent

    def __call__(self, frame_index: int) -> int:
        cand = int(self.rng.randrange(N_CAND))
        self.counts[cand] += 1
        self.frames += 1
        return int(_contracts().cand_to_mask(cand))

    def audit(self) -> dict:
        return {"kind": self.kind, "rng": f"random.Random({self.seed})",
                "cand_counts": list(self.counts), "bomb_count": 0,
                "masks_sent": self.frames}


class _CkptPolicy:
    """``policy:<ckpt>`` -- the 36-way student, sampled once per decision.

    Frame 0 of a decision carries the sampled bomb bit, frames 1..3 do not
    (`v830.run_episode36`'s emit rule, `v830.action_masks`).
    """

    kind = "policy"

    def __init__(self, spec: dict, *, seed: int, holder: dict) -> None:
        self.holder = holder
        self.seed = int(seed)
        self.ckpt = pathlib.Path(str(spec.get("ckpt"))).expanduser()
        if not self.ckpt.is_file():
            raise CollectError("OK_UNEXPECTED",
                               f"policy checkpoint {self.ckpt} is not a file")
        import torch

        self.torch = torch
        self.policy = m860().load_policy36(self.ckpt, seed=self.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed * 1000 + GEN_CONST)
        self.counts = [0] * N_CAND
        self.bomb_count = 0
        self.frames = 0
        self.decisions = 0
        self._masks: list[int] | None = None

    def bind(self, parent: object) -> None:
        self.holder["parent"] = parent

    def __call__(self, frame_index: int) -> int:
        index = int(frame_index)
        block, offset = divmod(index, FRAMES_PER_DECISION)
        if offset == 0 or self._masks is None:
            self._masks = self._sample(block)
        self.frames += 1
        return int(self._masks[offset])

    def _sample(self, decision: int) -> list[int]:
        from touhou_rl.v400.gym import protocol as P

        parent = self.holder.get("parent")
        if parent is None or not callable(getattr(parent, "ask", None)):
            raise CollectError(
                "OK_UNEXPECTED",
                f"policy:{self.ckpt} needs a live shadow parent to read STATE / "
                f"HAZARDS from, and the parent in hand cannot answer them")
        state, _reply = parent.state()
        hazards = P.parse_hazards_reply(parent.ask("HAZARDS"))
        feats, _detail = m431a().build_features(
            state=state, hazards=hazards, decision_index=int(decision))
        action = int(m830().sample_action36(self.policy, feats, "stochastic",
                                            self.generator))
        cand, bomb = m830().split_action(action)
        contracts = _contracts()
        first = int(contracts.cand_to_mask(cand, int(bomb)))
        rest = int(contracts.cand_to_mask(cand, 0))
        self.counts[int(cand)] += 1
        self.bomb_count += int(bomb)
        self.decisions += 1
        return [first, rest, rest, rest]

    def audit(self) -> dict:
        return {"kind": self.kind, "ckpt": str(self.ckpt),
                "ckpt_sha256": sha256_file(self.ckpt),
                "ckpt_bytes": self.ckpt.stat().st_size,
                "generator_seed": self.seed * 1000 + GEN_CONST,
                "sampler": ("v830_bombpol.sample_action36(kind='stochastic') over 36 "
                            "actions = cand*2+bomb; bomb bit on the decision's first "
                            "frame only"),
                "features": "v431_arm_a.build_features(state, hazards, decision)",
                "cand_counts": list(self.counts), "bomb_count": self.bomb_count,
                "decisions": self.decisions, "masks_sent": self.frames}


def build_policy(spec: dict, *, seed: int, holder: dict) -> object:
    kind = str(spec.get("kind"))
    if kind == "stay":
        return _ConstPolicy(0, kind="stay")
    if kind == "const":
        return _ConstPolicy(int(spec["cand"]))
    if kind == "random":
        return _RandomPolicy(seed)
    if kind == "policy":
        return _CkptPolicy(spec, seed=seed, holder=holder)
    raise CollectError("OK_UNEXPECTED", f"unknown policy kind {kind!r}")


# --------------------------------------------------------------------------- #
# the hash chain: the invariant `verify` re-checks on disk
# --------------------------------------------------------------------------- #
class HashChain:
    """``hash_pre[n] == hash_post[n-1]`` plus "the state really advanced".

    A repeat is impossible in a genuine run: the engine's 25-field hash starts
    with ``f0 = FrameCount()`` (`touhou_rl/v400/labels/state.py`, FIELD_NAMES[0]),
    so two different frames can never share a digest.  A repeat therefore means
    the recorded chain does not continue -- the injected ``chain`` fault in
    ``selftest`` is exactly that.
    """

    def __init__(self, origin: str) -> None:
        self.origin = str(origin)
        self.prev = str(origin)
        self.anchors: dict[int, str] = {0: str(origin)}
        self.seen: dict[str, int] = {str(origin): 0}
        self.rows = 0
        self.mismatches = 0

    def accept(self, *, decision_frame: int, hash_pre: str, hash_post: str) -> None:
        if hash_pre != self.prev:
            raise CollectError(
                "OK_UNEXPECTED",
                f"internal: decision_frame {decision_frame} hash_pre {hash_pre} is "
                f"not the previous hash_post {self.prev}; the writer is broken")
        if hash_post == hash_pre:
            raise CollectError(
                "STEP_NOT_ADVANCED",
                f"decision_frame {decision_frame}: the state hash did not change over "
                f"four steps the engine reported as advanced ({hash_pre}); the frozen "
                f"25-field hash includes f0=FrameCount, so the state did not advance")
        earlier = self.seen.get(hash_post)
        if earlier is not None:
            raise CollectError(
                "STEP_NOT_ADVANCED",
                f"decision_frame {decision_frame}: hash_post {hash_post} repeats the "
                f"anchor already recorded at frame {earlier}; the hash chain is broken "
                f"(the engine claims frame {decision_frame + FRAMES_PER_DECISION} but "
                f"the state hash belongs to frame {earlier})")
        self.anchors[int(decision_frame) + FRAMES_PER_DECISION] = str(hash_post)
        self.seen[str(hash_post)] = int(decision_frame) + FRAMES_PER_DECISION
        self.prev = str(hash_post)
        self.rows += 1

    def summary(self) -> dict:
        return {"hash_checks": len(self.anchors), "hash_mismatches": self.mismatches,
                "anchors": len(self.anchors), "rows": self.rows,
                "origin_hash": self.origin, "last_hash": self.prev,
                "chain_ok": True}


# --------------------------------------------------------------------------- #
# the collection loop (shared by the real parent, the fake parent and selftest)
# --------------------------------------------------------------------------- #
def _hello_gate(parent: object) -> dict:
    hello = getattr(parent, "hello", None)
    if not callable(hello):
        return {"called": False, "ok": True,
                "note": ("parent exposes no hello(); the HELLO gate cannot be "
                         "evaluated and is recorded as skipped (only the offline "
                         "fake parent takes this path)")}
    try:
        reply = hello()
    except Exception as exc:                                      # noqa: BLE001
        raise CollectError("HELLO_FAILED",
                           f"HELLO raised {type(exc).__name__}: {exc}") from exc
    got = reply_fields(reply) if isinstance(reply, str) else {}
    if not (isinstance(reply, str) and reply.startswith("OK")
            and got.get("role") == "shadow"):
        raise CollectError(
            "HELLO_FAILED",
            f"HELLO must answer 'OK ... role=shadow', got {reply!r}. A role=env "
            f"engine cannot PARK, and a parent that cannot PARK cannot be a source")
    return {"called": True, "ok": True, "reply": reply, "role": got.get("role")}


def _reset(parent: object, *, stage: int, difficulty: int, seed: int):
    """RESET and return ``(fields, reply, mode)``; ``RESET_FAILED`` on any gap."""

    frozen = (0, 1)
    wanted = (int(stage), int(difficulty))
    ask = getattr(parent, "ask", None)
    if wanted == frozen or not callable(ask):
        if wanted != frozen:
            raise CollectError(
                "OK_UNEXPECTED",
                f"--stage {stage} --difficulty {difficulty} cannot be sent: the frozen "
                f"`v432_fork_arm.ForkParent.reset(seed)` issues RESET 0 1 0 <seed> 0 "
                f"(stage 0, difficulty 1) and this parent has no raw `ask`. Recording "
                f"stage={stage}/difficulty={difficulty} would be a false label")
        try:
            fields, reply = parent.reset(int(seed))
        except CollectError:
            raise
        except Exception as exc:                                  # noqa: BLE001
            raise CollectError("RESET_FAILED",
                               f"RESET raised {type(exc).__name__}: {exc}") from exc
        mode = "frozen"
    else:
        try:
            reply = ask(f"RESET {int(stage)} {int(difficulty)} 0 {int(seed)} 0")
        except Exception as exc:                                  # noqa: BLE001
            raise CollectError("RESET_FAILED",
                               f"extended RESET raised {type(exc).__name__}: {exc}") from exc
        try:
            setattr(parent, "reset_reply", reply)
        except Exception:                                          # noqa: BLE001, S110
            pass
        fields = reply_fields(reply) if isinstance(reply, str) else {}
        mode = "extended"
    if not (isinstance(reply, str) and reply.startswith("OK")):
        raise CollectError("RESET_FAILED", f"RESET is not OK: {reply!r}")
    hash_hex = str((fields or {}).get("hash") or "").lower()
    if not HASH_RE.match(hash_hex):
        raise CollectError(
            "RESET_FAILED",
            f"RESET reply carries no usable state hash ({hash_hex!r}): the tape's "
            f"frame-0 anchor would be missing, so no chain could be checked. reply="
            f"{reply!r}")
    return dict(fields or {}), reply, mode


def _step_once(parent: object, mask: int, *, frame_cursor: int | None,
               decision_frame: int, offset: int) -> dict:
    """One STEP with every single-frame invariant this tool relies on."""

    where = f"decision_frame {decision_frame} frame+{offset} (mask {int(mask)})"
    try:
        step = parent.step(int(mask))
    except CollectError:
        raise
    except Exception as exc:                                      # noqa: BLE001
        # A raise here is a transport/decode failure (HarnessError: ERR reply, pipe
        # closed, engine gone; EpisodeError: reply is not OK / has no advanced=),
        # i.e. "this run cannot be evaluated", not a verdict on the frame -- so it
        # is OK_UNEXPECTED (exit 2).  `STEP_NOT_ADVANCED` is reserved for a STEP that
        # *did* answer with advanced != 1, which arrives through the dict below.
        text = str(exc).lower()
        code = "HASH_MISSING" if ("neither a hash" in text or "no hash" in text) \
            else "OK_UNEXPECTED"
        raise CollectError(
            code, f"{where}: STEP raised {type(exc).__name__}: {exc}") from exc
    if not isinstance(step, dict):
        raise CollectError("OK_UNEXPECTED", f"{where}: STEP did not return a dict")
    advanced = step.get("advanced")
    if isinstance(advanced, bool) or not isinstance(advanced, int):
        raise CollectError("OK_UNEXPECTED",
                           f"{where}: advanced={advanced!r} is not an int")
    exited = bool(step.get("exited"))
    game_over = bool(step.get("game_over"))
    hash_hex = step.get("hash")
    if hash_hex is not None:
        hash_hex = str(hash_hex).strip().lower()
    frame = _as_int(step.get("frame"))
    reply = step.get("reply")
    if not (exited or game_over):
        if advanced != 1:
            raise CollectError(
                "STEP_NOT_ADVANCED",
                f"{where}: advanced={advanced} on a live frame; every step must "
                f"advance exactly one frame (reply={reply!r})")
        if not hash_hex:
            raise CollectError(
                "HASH_MISSING",
                f"{where}: the step returned no state hash and no exit/game_over "
                f"flag, so the block's hash_post anchor cannot be recorded "
                f"(reply={reply!r})")
    elif advanced not in (0, 1):
        raise CollectError("STEP_NOT_ADVANCED",
                           f"{where}: death step reports advanced={advanced}")
    if hash_hex is not None and not HASH_RE.match(hash_hex):
        raise CollectError("HASH_MISSING",
                           f"{where}: hash {hash_hex!r} is not a lowercase hex digest")
    if isinstance(reply, str) and reply.startswith("OK ") and hash_hex is not None:
        wire = str(reply_fields(reply).get("hash") or "").strip().lower()
        if wire and wire != hash_hex:
            raise CollectError(
                "OK_UNEXPECTED",
                f"{where}: the reply line carries hash={wire} but the decoded STEP "
                f"returned {hash_hex}; the transport and the decoder disagree, so no "
                f"hash from this step can be trusted (reply={reply!r})")
    if frame is not None and frame_cursor is not None and frame != frame_cursor + advanced:
        raise CollectError(
            "STEP_NOT_ADVANCED",
            f"{where}: the engine reports frame {frame} but the cursor was "
            f"{frame_cursor} and advanced={advanced} (expected "
            f"{frame_cursor + advanced}); a rewound or stalled frame counter cannot "
            f"label an anchor (reply={reply!r})")
    return {"advanced": advanced, "hash": hash_hex, "exited": exited,
            "game_over": game_over, "frame": frame, "reply": reply,
            "mask": int(mask)}


def run_collect_loop(*, parent: object, seed: int, stage: int, difficulty: int,
                     blocks: int, policy: object, stop_on_death: bool,
                     emit: object, tag: str = "") -> dict:
    """start -> reset -> blocks x 4 x step, emitting one tape row per block.

    ``emit(row)`` is called once per **complete** block, in order.  A death that
    happens inside a block drops that block (a tape is a run of complete 4-frame
    blocks) and ends the loop; the death is described in the returned record.
    """

    if int(blocks) < 1:
        raise CollectError("OK_UNEXPECTED", f"--blocks must be >= 1, got {blocks}")
    bind = getattr(policy, "bind", None)
    if callable(bind):
        bind(parent)
    started = time.perf_counter()
    parent.start()
    hello = _hello_gate(parent)
    log(f"HELLO: {hello.get('reply') or hello.get('note')}")
    fields0, reset_reply, reset_mode = _reset(parent, stage=stage,
                                              difficulty=difficulty, seed=seed)
    origin = str(fields0["hash"]).lower()
    cursor = _as_int(fields0.get("frame"))
    log(f"RESET {reset_mode}: seed={seed} stage={stage} difficulty={difficulty} "
        f"frame0_hash={origin} reply={reset_reply!r}")
    chain = HashChain(origin)
    prev = origin
    rows = 0
    death = None
    frame_first = cursor if cursor is not None else 0
    frame_last = frame_first
    for block in range(int(blocks)):
        decision_frame = block * FRAMES_PER_DECISION
        masks: list[int] = []
        for offset in range(FRAMES_PER_DECISION):
            index = decision_frame + offset
            try:
                mask = int(policy(index))
            except CollectError:
                raise
            except Exception as exc:                              # noqa: BLE001
                raise CollectError(
                    "OK_UNEXPECTED",
                    f"policy raised at frame {index}: {type(exc).__name__}: {exc}"
                ) from exc
            if not 0 <= mask <= 0xFFFF:
                raise CollectError(
                    "OK_UNEXPECTED",
                    f"policy returned mask {mask} at frame {index}; the engine's "
                    f"button mask is 16-bit")
            masks.append(mask)
        block_frame_start = cursor
        steps: list[dict] = []
        block_death = None
        for offset, mask in enumerate(masks):
            step = _step_once(parent, mask, frame_cursor=cursor,
                              decision_frame=decision_frame, offset=offset)
            steps.append(step)
            if step["frame"] is not None:
                cursor = step["frame"]
            elif cursor is not None:
                cursor = cursor + step["advanced"]
            if step["exited"] or step["game_over"]:
                block_death = {
                    "block": block, "decision_frame": decision_frame,
                    "offset": offset, "mask": mask,
                    "exited": step["exited"], "game_over": step["game_over"],
                    "frame": cursor if cursor is not None else decision_frame + offset + 1,
                    "hash": step["hash"], "reply": step["reply"],
                    "advanced": step["advanced"],
                    "block_complete": len(steps) == FRAMES_PER_DECISION,
                }
                break
        complete = len(steps) == FRAMES_PER_DECISION
        if complete:
            for offset, step in enumerate(steps):
                if step["advanced"] != 1:
                    raise CollectError(
                        "STEP_NOT_ADVANCED",
                        f"decision_frame {decision_frame} frame+{offset}: advanced="
                        f"{step['advanced']} inside a complete block")
                if not step["hash"]:
                    raise CollectError(
                        "HASH_MISSING",
                        f"decision_frame {decision_frame} frame+{offset}: no hash "
                        f"inside a complete block")
            post = str(steps[-1]["hash"])
            chain.accept(decision_frame=decision_frame, hash_pre=prev, hash_post=post)
            start_frame = block_frame_start if block_frame_start is not None else decision_frame
            end_frame = cursor if cursor is not None else decision_frame + FRAMES_PER_DECISION
            row = {
                "kind": "decision",
                "schema": SCHEMA_TAPE,
                "decision": int(block),
                "frame": int(start_frame),
                "decision_frame": int(decision_frame),
                "frames_per_decision": FRAMES_PER_DECISION,
                "stage": int(stage),
                "difficulty": int(difficulty),
                "step_frames": list(masks),
                "hash_pre": prev,
                "hash_post": post,
                "frame_end": int(end_frame),
                "policy": str(getattr(policy, "kind", "unknown")),
                "tag": str(tag),
                "source": TOOL,
            }
            emit(row)
            rows += 1
            prev = post
            frame_last = end_frame
            if _PROGRESS_EVERY and (rows % _PROGRESS_EVERY == 0 or rows == int(blocks)):
                log(f"block {rows}/{blocks} frames={rows * FRAMES_PER_DECISION} "
                    f"frame_end={end_frame} hash={post} "
                    f"wall={time.perf_counter() - started:.1f}s")
        if block_death is not None:
            death = block_death
            break
        if not complete:
            raise CollectError(
                "OK_UNEXPECTED",
                f"block {block} stopped after {len(steps)} steps with no death record")
    wall = time.perf_counter() - started
    summary = chain.summary()
    out = {
        "rows": rows, "frames": rows * FRAMES_PER_DECISION,
        "origin_hash": origin, "last_hash": prev,
        "death": death, "death_frame": (death or {}).get("frame"),
        "hash_checks": summary["hash_checks"],
        "hash_mismatches": summary["hash_mismatches"],
        "anchors": summary["anchors"], "chain_ok": True,
        "frame_first": frame_first, "frame_last": frame_last,
        "wall_s": round(wall, 3),
        "hello": hello, "reset": {"mode": reset_mode, "reply": reset_reply,
                                  "frame0_hash": origin,
                                  "frame0_frame": _as_int(fields0.get("frame"))},
        "stop_on_death_requested": bool(stop_on_death),
        "stop_reason": "death" if death is not None else "blocks_complete",
        "policy": getattr(policy, "audit", lambda: {})() if callable(
            getattr(policy, "audit", None)) else {},
    }
    if death is not None:
        out["stop_reason"] = "death"
        log(f"death: block {death['block']} offset {death['offset']} "
            f"frame={death['frame']} exited={death['exited']} "
            f"game_over={death['game_over']} block_complete="
            f"{death['block_complete']}")
    return out


# --------------------------------------------------------------------------- #
# tape reader / verifier (no engine; same rules as v829_pair.read_tape, stricter)
# --------------------------------------------------------------------------- #
def verify_tape(path: object, *, expect_rows: int | None = None,
                expect_origin: str | None = None) -> dict:
    """Read a 4-frame tape and judge every rule this file guarantees.

    Returns a document with ``ok``, ``checks`` (numbered), ``rows``, ``frames``,
    ``anchors``, ``hashes``, ``masks`` and ``first_failure``.  It never writes.
    """

    tape = pathlib.Path(str(path))
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"n": len(checks) + 1, "check": str(name), "ok": bool(ok),
                       "detail": str(detail)})
        return bool(ok)

    doc: dict = {"schema": SCHEMA_VERIFY, "tape": str(tape), "ok": False,
                 "checks": checks, "rows": 0, "frames": 0, "anchors": 0,
                 "hashes": {}, "masks": [], "origin_hash": None,
                 "first_failure": None, "present": False}
    ok_exists = check("tape_readable", tape.is_file(),
                      "" if tape.is_file() else f"{tape} is not a file")
    if not ok_exists:
        doc["first_failure"] = {"n": 1, "check": "tape_readable"}
        return doc
    doc["present"] = True
    doc["bytes"] = tape.stat().st_size
    doc["sha256"] = sha256_file(tape)
    lines = tape.read_text(encoding="utf-8").splitlines()
    parsed: list[dict] = []
    parse_errors: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            parse_errors.append(f"line {index}: {exc}")
            continue
        if not isinstance(row, dict):
            parse_errors.append(f"line {index}: not a JSON object")
            continue
        parsed.append(row)
    check("rows_parse", not parse_errors, "; ".join(parse_errors[:3]))
    if not parsed:
        check("rows_present", False, f"{tape} has no rows")
        doc["first_failure"] = next((c for c in checks if not c["ok"]), None)
        return doc

    # -- per-row structure ---------------------------------------------------
    bad_frame: list[str] = []
    bad_steps: list[str] = []
    bad_masks: list[str] = []
    bad_hashes: list[str] = []
    bad_fpd: list[str] = []
    bad_decision: list[str] = []
    masks: list[int] = []
    hashes: dict[int, str] = {}
    rows = 0
    for index, row in enumerate(parsed):
        frame = row.get("decision_frame")
        if isinstance(frame, bool) or not isinstance(frame, int) \
                or frame != rows * FRAMES_PER_DECISION:
            bad_frame.append(f"row {index}: decision_frame={frame!r} "
                             f"(expected {rows * FRAMES_PER_DECISION})")
        steps = row.get("step_frames")
        steps = steps if isinstance(steps, list) else []
        if len(steps) != FRAMES_PER_DECISION:
            bad_steps.append(f"row {index}: {len(steps)} step_frames "
                             f"(expected {FRAMES_PER_DECISION})")
            ints: list[int] = []
        else:
            try:
                ints = [int(value) for value in steps]
            except (TypeError, ValueError) as exc:
                bad_steps.append(f"row {index}: non-integer step_frames {steps!r}: {exc}")
                ints = []
        if len(ints) == FRAMES_PER_DECISION:
            for offset, mask in enumerate(ints):
                if not 0 <= mask <= 0xFFFF:
                    bad_masks.append(f"row {index} +{offset}: mask {mask} outside 16-bit")
        pre = row.get("hash_pre")
        post = row.get("hash_post")
        if not (isinstance(pre, str) and HASH_RE.match(str(pre).lower())):
            bad_hashes.append(f"row {index}: hash_pre={pre!r} is not a hex digest")
        if not (isinstance(post, str) and HASH_RE.match(str(post).lower())):
            bad_hashes.append(f"row {index}: hash_post={post!r} is not a hex digest")
        fpd = row.get("frames_per_decision")
        if fpd != FRAMES_PER_DECISION:
            bad_fpd.append(f"row {index}: frames_per_decision={fpd!r} "
                           f"(expected {FRAMES_PER_DECISION})")
        decision = row.get("decision")
        if decision is not None and decision != index:
            bad_decision.append(f"row {index}: decision={decision!r} (expected {index})")
        frame_end = row.get("frame_end")
        frame_start = row.get("frame")
        if isinstance(frame_end, int) and isinstance(frame_start, int) \
                and not isinstance(frame_end, bool) and not isinstance(frame_start, bool):
            if frame_end != frame_start + FRAMES_PER_DECISION:
                bad_decision.append(
                    f"row {index}: frame_end={frame_end} != frame={frame_start} + "
                    f"{FRAMES_PER_DECISION}")
        if len(ints) == FRAMES_PER_DECISION:
            masks.extend(ints)
        if isinstance(pre, str) and isinstance(post, str):
            hashes[int(frame) if isinstance(frame, int) else rows * FRAMES_PER_DECISION] \
                = str(pre).lower()
            hashes[(int(frame) if isinstance(frame, int) else rows * FRAMES_PER_DECISION)
                   + FRAMES_PER_DECISION] = str(post).lower()
        rows += 1
    check("decision_frame_contiguous", not bad_frame, "; ".join(bad_frame[:3]))
    check("step_frames_4", not bad_steps, "; ".join(bad_steps[:3]))
    check("masks_16bit", not bad_masks, "; ".join(bad_masks[:3]))
    check("hash_fields_present", not bad_hashes, "; ".join(bad_hashes[:3]))
    check("frames_per_decision_4", not bad_fpd, "; ".join(bad_fpd[:3]))
    check("row_optional_fields", not bad_decision, "; ".join(bad_decision[:3]))

    # -- chain ---------------------------------------------------------------
    breaks: list[str] = []
    for index in range(1, len(parsed)):
        prev_post = str(parsed[index - 1].get("hash_post", "")).lower()
        this_pre = str(parsed[index].get("hash_pre", "")).lower()
        if prev_post != this_pre:
            breaks.append(f"row {index}: hash_pre {this_pre} != row {index - 1} "
                          f"hash_post {prev_post}")
    check("hash_chain", not breaks, "; ".join(breaks[:3]))
    origin = str(parsed[0].get("hash_pre", "")).lower()
    doc["origin_hash"] = origin
    ok_origin = bool(HASH_RE.match(origin))
    detail = "" if ok_origin else f"first row hash_pre {origin!r} is not a hex digest"
    if ok_origin and expect_origin is not None:
        wanted = str(expect_origin).strip().lower()
        ok_origin = origin == wanted
        detail = "" if ok_origin else f"origin {origin} != --expect-origin {wanted}"
    check("origin_hash", ok_origin, detail)

    # -- anchors -------------------------------------------------------------
    frames_built = len(masks)
    expected_keys = set(range(0, frames_built + 1, ANCHOR_STRIDE))
    missing = sorted(expected_keys - set(hashes))
    extra = sorted(set(hashes) - expected_keys)
    check("anchor_coverage", not missing and not extra,
          f"frames_built={frames_built} anchors={len(hashes)} "
          f"missing={missing[:6]} extra={extra[:6]}")
    values = list(hashes.values())
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    check("anchor_distinct", not duplicates,
          f"repeated anchor digests {duplicates[:4]} (the 25-field hash carries "
          f"f0=FrameCount, so a repeat means the chain does not continue)")
    repeats: list[str] = []
    for index, row in enumerate(parsed):
        if str(row.get("hash_pre", "")).lower() == str(row.get("hash_post", "")).lower():
            repeats.append(f"row {index}: hash_pre == hash_post")
    check("row_hash_advances", not repeats, "; ".join(repeats[:3]))

    # -- rows / target -------------------------------------------------------
    doc.update({"rows": len(parsed), "frames": frames_built, "masks": masks,
                "anchors": len(hashes),
                "hashes": {str(key): value for key, value in sorted(hashes.items())}})
    if expect_rows is not None:
        check("expect_rows", len(parsed) == int(expect_rows),
              f"{len(parsed)} rows, --expect-rows {int(expect_rows)}")
    target = frames_built
    check("target_is_anchor",
          target % ANCHOR_STRIDE == 0 and target in hashes,
          f"target {target} ({'a multiple of' if target % ANCHOR_STRIDE == 0 else 'NOT a multiple of'} "
          f"{ANCHOR_STRIDE}, "
          f"{'recorded' if target in hashes else 'not recorded'}); "
          f"v450.masks_for passes target = N + 1, so a tape must have a hash at "
          f"its own end frame")
    doc["target"] = target

    failed = [entry for entry in checks if not entry["ok"]]
    doc["ok"] = not failed
    doc["first_failure"] = ({"n": failed[0]["n"], "check": failed[0]["check"],
                             "detail": failed[0]["detail"]} if failed else None)
    doc["checks_run"] = len(checks)
    doc["checks_ok"] = len(checks) - len(failed)
    return doc


def format_verify(doc: dict) -> list[str]:
    lines = [f"v829 collect verify: {doc.get('tape')}"]
    if doc.get("present"):
        lines.append(f"  bytes       {doc.get('bytes')}  sha256 {doc.get('sha256')}")
        lines.append(f"  rows        {doc.get('rows')}  frames {doc.get('frames')}  "
                     f"anchors {doc.get('anchors')}  target {doc.get('target')}")
        lines.append(f"  origin      {doc.get('origin_hash')}")
        lines.append(f"  checks      {doc.get('checks_ok')}/"
                     f"{doc.get('checks_run')} ok")
        for entry in doc.get("checks", []):
            if not entry["ok"]:
                lines.append(f"  FAIL #{entry['n']} {entry['check']}: {entry['detail']}")
    else:
        lines.append(f"  FAIL #{1} tape_readable: {doc.get('tape')} is not a file")
    if doc.get("ok"):
        lines.append("VERIFY: ok")
    else:
        failure = doc.get("first_failure") or {}
        lines.append(f"VERIFY: failed (first non-compliant check: #{failure.get('n')} "
                     f"{failure.get('check')})")
    return lines


# --------------------------------------------------------------------------- #
# the offline fake parent (deterministic pseudo hashes, no engine, no torch)
# --------------------------------------------------------------------------- #
class FakeParent:
    """``start``/``reset``/``step`` (+ ``ask`` for RESET) with injected faults.

    The hash sequence is ``sha256("v829-fake-source:<seed>:<frame>")[:16]``, i.e.
    the same 16-hex shape the harness emits, and it is a function of the frame
    only -- so a repeat is always a chain break, never a coincidence.
    """

    def __init__(self, *, seed: int, fault: str = "none", fault_block: int = 1,
                 fault_step: int = 1) -> None:
        self.seed = int(seed)
        self.fault = str(fault or "none")
        self.fault_block = int(fault_block)
        self.fault_step = int(fault_step)
        self.frame = 0
        self.calls = 0
        self.started = False
        self.stage = 0
        self.difficulty = 1
        self.reset_mode = "frozen"

    def hash_at(self, frame: int) -> str:
        raw = f"v829-fake-source:{self.seed}:{int(frame)}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def start(self) -> None:
        self.started = True

    def hello(self) -> str:
        if self.fault == "hello_fail":
            return "ERR role=env (fake parent asked to fail HELLO)"
        return "OK harness=fake role=shadow pid=0"

    def close(self) -> None:
        self.started = False

    def _reset_reply(self) -> str:
        return (f"OK reset=1 ready=1 mode=cold stage={self.stage} "
                f"difficulty={self.difficulty} frame=0 hash={self.hash_at(0)}")

    def reset(self, seed: int):
        self.seed = int(seed)
        self.frame = 0
        self.calls = 0
        reply = self._reset_reply()
        return reply_fields(reply), reply

    def ask(self, line: str) -> str:
        parts = str(line).split()
        if parts and parts[0].upper() == "RESET" and len(parts) == 6:
            self.stage = int(parts[1])
            self.difficulty = int(parts[2])
            self.frame = 0
            self.calls = 0
            self.reset_mode = "extended"
            return self._reset_reply()
        raise RuntimeError(f"FakeParent cannot answer {line!r}")

    def step(self, mask: int) -> dict:
        index = self.calls
        self.calls += 1
        block, offset = divmod(index, FRAMES_PER_DECISION)
        at = (block, offset) == (self.fault_block, self.fault_step)
        if at and self.fault == "advanced_zero":
            return self._step(advanced=0, hash_hex=self.hash_at(self.frame))
        if at and self.fault == "hash_none":
            return self._step(advanced=1, hash_hex=None)
        if at and self.fault == "death":
            self.frame += 1
            return self._step(advanced=1, hash_hex=self.hash_at(self.frame),
                              exited=True)
        if at and self.fault == "death_nohash":
            return self._step(advanced=0, hash_hex=None, exited=True)
        if at and self.fault == "chain":
            self.frame += 1
            # Replay the frame-4 anchor at this later block's end: the chain does
            # not continue (frames 4 and 8/12 can never share a 25-field hash,
            # because f0=FrameCount is part of it).
            stale = self.hash_at(FRAMES_PER_DECISION)
            return self._step(advanced=1, hash_hex=stale)
        self.frame += 1
        return self._step(advanced=1, hash_hex=self.hash_at(self.frame))

    def _step(self, *, advanced: int, hash_hex: str | None,
              exited: bool = False) -> dict:
        advanced = int(advanced)
        fields = [f"advanced={advanced}", f"frame={self.frame}"]
        if hash_hex is not None:
            fields.append(f"hash={hash_hex}")
        if exited:
            fields.append("exit=1")
        reply = "OK " + " ".join(fields)
        return {"advanced": advanced, "hash": hash_hex, "exited": bool(exited),
                "game_over": False, "frame": self.frame, "reply": reply}


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def _collect_record(*, args, out_dir: pathlib.Path, work: pathlib.Path, tag: str,
                    rows_path: pathlib.Path) -> dict:
    spec = args.policy
    return {
        "schema": SCHEMA_COLLECT, "producer": PRODUCER, "tool_version": TOOL_VERSION,
        "created_at": now_iso(), "status": "running",
        "argv": [sys.executable] + list(sys.argv),
        "tag": tag,
        "interpreter": {"executable": sys.executable, "version": sys.version,
                        "version_info": list(sys.version_info[:3]),
                        "implementation": sys.implementation.name},
        "torch": _torch_probe(),
        "out_dir": str(out_dir), "work_dir": str(work),
        "labels_path": str(rows_path),
        "requested": {"seed": int(args.seed), "stage": int(args.stage),
                      "difficulty": int(args.difficulty), "blocks": int(args.blocks),
                      "policy": spec["spec"], "stop_on_death": bool(args.stop_on_death),
                      "fake_parent": bool(args.fake_parent),
                      "overwrite_empty": bool(args.overwrite_empty)},
        "engine": None, "data_dir": None, "audio": None, "policy": None,
        "hello": None, "reset": None, "death": None, "rows_written": 0,
        "hash_checks": 0, "hash_mismatches": 0, "hash_chain_ok": None,
        "wall_s": None, "tape": None, "tape_self_check": None, "error": None,
        "writes": [str(out_dir / "labels.jsonl"), str(out_dir / "shard.json"),
                   str(out_dir / "collect.json"), str(work / "tags.jsonl"),
                   str(work / "data" / tag), str(work / "logs" / tag)],
        "writes_scope": ("the three <out-dir> artifacts plus the work-dir ledger, "
                         "private data dir and log dir; a fake-parent run touches "
                         "neither data/ nor logs/"),
    }


def _shard_document(*, args, tag: str, run: dict, tape: dict,
                    audio: dict | None, engine: dict | None) -> dict:
    return {
        "schema": SCHEMA_SHARD, "producer": PRODUCER, "created_at": now_iso(),
        "tag": tag,
        "seed": int(args.seed), "stage": int(args.stage),
        "difficulty": int(args.difficulty),
        "blocks": int(args.blocks), "rows": int(run["rows"]),
        "frames": int(run["frames"]),
        "frames_per_decision": FRAMES_PER_DECISION,
        "policy": run.get("policy") or {},
        "engine": engine,
        "engine_sha256": (engine or {}).get("sha256"),
        "data_cfg_sha256": (audio or {}).get("sha256"),
        "audio_all_muted": (audio or {}).get("all_muted"),
        "created_at_utc_offset": time.strftime("%z"),
        "wall_s": run.get("wall_s"),
        "death_frame": run.get("death_frame"),
        "death": run.get("death"),
        "hash_chain_ok": bool(run.get("chain_ok")),
        "hash_checks": int(run.get("hash_checks", 0)),
        "hash_mismatches": int(run.get("hash_mismatches", 0)),
        "anchors": int(run.get("anchors", 0)),
        "origin_hash": run.get("origin_hash"), "last_hash": run.get("last_hash"),
        "frame_first": run.get("frame_first"), "frame_last": run.get("frame_last"),
        "tape": tape,
        "stop_on_death": bool(args.stop_on_death),
        "stop_reason": run.get("stop_reason"),
        "fake_parent": bool(args.fake_parent),
        "source": TOOL,
        "hash_algorithm": ("engine HASH: FNV-1a over the 25 tracked fields "
                           "(f0=FrameCount), 16 lowercase hex digits"),
        "walk": {"hello": run.get("hello"), "reset": run.get("reset")},
    }


def _drive(args, *, out_dir: pathlib.Path, work: pathlib.Path, tag: str,
           rows_path: pathlib.Path, record: dict) -> dict:
    """Start the parent, run the loop, write ``labels.jsonl``; returns the run."""

    seed = int(args.seed)
    parent = None
    data_dir = None
    audio = None
    engine = None
    if args.fake_parent:
        parent = FakeParent(seed=seed, fault=args.fake_fault,
                            fault_block=int(args.fake_fault_block),
                            fault_step=int(args.fake_fault_step))
        record["engine"] = {"skipped": "fake parent: no engine is started"}
        record["audio"] = {"skipped": "fake parent: no engine, no private data dir",
                           "all_muted": None}
        record["parent"] = {"kind": "fake", "seed": seed, "fault": args.fake_fault,
                            "fault_block": int(args.fake_fault_block),
                            "fault_step": int(args.fake_fault_step),
                            "hash_sequence": ("sha256('v829-fake-source:<seed>:"
                                              "<frame>')[:16]"),
                            "note": ("NO ENGINE: this tape is synthetic and can never "
                                     "be used as a real ancestor")}
        record["torch"] = _torch_probe()
        log("parent: fake (no engine, no audio, deterministic pseudo hashes)")
    else:
        launch_env = m450().prepare_env()
        record["launch_env"] = launch_env
        log(f"launch env: {json.dumps(launch_env, ensure_ascii=False)}")
        engine = engine_gate(args.engine, args.engine_sha256)
        record["engine"] = engine
        log(f"engine: {engine['path']} sha256={engine['sha256']} "
            f"match={engine['match']}")
        data_source = data_source_gate(args.data_source)
        record["data_source"] = data_source
        data_dir = work / "data" / tag
        log(f"private data dir: {data_dir} (from {data_source['data_source']})")
        try:
            info = m432().build_private_data_dir(data_dir, pathlib.Path(
                data_source["data_source"]))
        except CollectError:
            raise
        except Exception as exc:                                  # noqa: BLE001
            text = str(exc)
            code = "AUDIO_NOT_MUTED" if "not muted" in text else "OK_UNEXPECTED"
            raise CollectError(
                code, f"could not build the private data dir {data_dir}: "
                      f"{type(exc).__name__}: {text[:900]}") from exc
        record["data_dir"] = {"data_dir": info["data_dir"],
                              "generated_from": info["generated_from"],
                              "cfg": info["cfg"]}
        audio = audio_evidence(pathlib.Path(info["cfg"]["path"]))
        record["audio"] = audio
        assert_audio_muted(audio, when="before the engine starts")
        log(f"audio: four fields {json.dumps(audio['fields'])} all_muted="
            f"{audio['all_muted']} cfg_sha256={audio['sha256'][:16]}")
        record["torch"] = _torch_version()
        log(f"torch: {record['torch'].get('version')}")
        parent = m432().ForkParent(slot=0, window=seed, engine=str(args.engine),
                                   data_dir=data_dir, log_dir=work / "logs" / tag)
        record["parent"] = {"kind": "ForkParent", "window": seed, "slot": 0,
                            "engine": str(args.engine),
                            "data_dir": str(data_dir),
                            "log_dir": str(work / "logs" / tag)}

    holder = {"parent": parent}
    policy = build_policy(args.policy, seed=seed, holder=holder)
    log(f"policy: {args.policy['spec']} ({getattr(policy, 'kind', '?')})")
    rows_written = {"n": 0}
    try:
        with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
            def emit(row: dict) -> None:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                rows_written["n"] += 1
                record["rows_written"] = rows_written["n"]

            run = run_collect_loop(
                parent=parent, seed=seed, stage=int(args.stage),
                difficulty=int(args.difficulty), blocks=int(args.blocks),
                policy=policy, stop_on_death=bool(args.stop_on_death),
                emit=emit, tag=tag)
    finally:
        closer = getattr(parent, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:                                      # noqa: BLE001, S110
                pass
    run["rows_written"] = rows_written["n"]
    if data_dir is not None:
        post = audio_evidence(data_dir / "th08.cfg")
        record["audio"]["post_run"] = post
        record["audio"]["post_run_muted"] = bool(post["all_muted"])
        assert_audio_muted(post, when="after the run (the engine rewrites th08.cfg "
                                       "on exit)")
        log(f"audio after the run: all_muted={post['all_muted']} "
            f"sha256={post['sha256'][:16]}")
    return run


def mode_collect(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    out_dir = resolve_write_dir(args.out_dir, what="--out-dir")
    work = resolve_write_dir(args.work_dir, what="--work-dir")
    fake_root = REPO / FAKE_PARENT_ROOT_NAME
    if args.fake_parent and not is_under(out_dir, fake_root):
        raise CollectError(
            "OUT_DIR_FORBIDDEN",
            f"--fake-parent writes a SYNTHETIC tape (no engine), so its --out-dir "
            f"must stay under {fake_root}/; got {out_dir}. A synthetic tape must "
            f"never be reachable from a real asset path")
    state = assert_out_dir_usable(out_dir, overwrite_empty=bool(args.overwrite_empty))
    if args.fake_parent and str(args.policy.get("kind")) == "policy":
        raise CollectError(
            "OK_UNEXPECTED",
            "--fake-parent cannot drive --policy policy:<ckpt>: there is no engine "
            "to read STATE/HAZARDS from")
    spec = dict(args.policy)
    tag = validate_tag(args.tag) if str(args.tag).strip() else default_tag(
        seed=int(args.seed), stage=int(args.stage), difficulty=int(args.difficulty),
        blocks=int(args.blocks), spec=spec, out_dir=out_dir)
    cell = {"out_dir": str(out_dir), "seed": int(args.seed),
            "stage": int(args.stage), "difficulty": int(args.difficulty),
            "blocks": int(args.blocks), "policy": spec["spec"]}
    claim = claim_tag(work, tag=tag, cell=cell)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "labels.jsonl"
    record = _collect_record(args=args, out_dir=out_dir, work=work, tag=tag,
                             rows_path=rows_path)
    record["out_dir_state_before"] = state
    record["tag_claim"] = claim
    log(f"out-dir {out_dir} (before: {state['state']})  work-dir {work}")
    log(f"tag {tag} (ledger {claim['ledger']}, entries {claim['entries']}, "
        f"reused_same_cell={claim['reused_same_cell']})")
    if spec["kind"] == "stay":
        log("NOTE: --policy stay holds candidate 0 (no direction) for every frame; "
            "pass --policy policy:<ckpt> for a real ancestor")
    run = None
    try:
        run = _drive(args, out_dir=out_dir, work=work, tag=tag,
                     rows_path=rows_path, record=record)
        tape = {"path": str(rows_path), "sha256": sha256_file(rows_path),
                "bytes": rows_path.stat().st_size, "rows": int(run["rows"]),
                "frames": int(run["frames"])}
        shard = _shard_document(args=args, tag=tag, run=run, tape=tape,
                               audio=record.get("audio"),
                               engine=record.get("engine"))
        write_json(out_dir / "shard.json", shard)
        log(f"shard.json: {out_dir / 'shard.json'}")
        self_check = verify_tape(rows_path, expect_rows=int(run["rows"]),
                                 expect_origin=run["origin_hash"])
        record["tape_self_check"] = {
            "ok": bool(self_check["ok"]),
            "checks_ok": self_check.get("checks_ok"),
            "checks_run": self_check.get("checks_run"),
            "first_failure": self_check.get("first_failure"),
        }
        if not self_check["ok"]:
            failure = self_check.get("first_failure") or {}
            raise CollectError(
                "OK_UNEXPECTED",
                f"the tape just written ({rows_path}) fails this tool's own verify at "
                f"check #{failure.get('n')} {failure.get('check')}: "
                f"{failure.get('detail')}. The writer and the reader disagree, so the "
                f"tape cannot be trusted even though the loop reached the end")
        record.update({
            "status": "ok", "wall_s": round(time.perf_counter() - started, 3),
            "run_wall_s": run.get("wall_s"),
            "rows_written": int(run["rows"]),
            "hash_checks": int(run["hash_checks"]),
            "hash_mismatches": int(run["hash_mismatches"]),
            "hash_chain_ok": bool(run["chain_ok"]),
            "hello": run.get("hello"), "reset": run.get("reset"),
            "death": run.get("death"), "death_frame": run.get("death_frame"),
            "policy": run.get("policy"), "tape": tape, "shard": str(out_dir / "shard.json"),
            "stop_reason": run.get("stop_reason"),
        })
    except CollectError as exc:
        record.update({"status": "failed", "wall_s": round(
            time.perf_counter() - started, 3),
            "error": {"code": exc.code, "detail": exc.detail},
            "rows_written": int(record.get("rows_written") or 0)})
        try:
            write_json(out_dir / "collect.json", record)
        except OSError:
            pass
        log(f"REFUSED [{exc.code}] {exc.detail}")
        raise
    except Exception as exc:                                      # noqa: BLE001
        # Everything unclassified is OK_UNEXPECTED (exit 2) -- but the failure is
        # still recorded next to the partial tape, so nobody has to guess.
        detail = f"{type(exc).__name__}: {exc}"
        record.update({"status": "failed", "wall_s": round(
            time.perf_counter() - started, 3),
            "error": {"code": "OK_UNEXPECTED", "detail": detail},
            "rows_written": int(record.get("rows_written") or 0)})
        try:
            write_json(out_dir / "collect.json", record)
        except OSError:
            pass
        log(f"REFUSED [OK_UNEXPECTED] {detail}")
        raise CollectError("OK_UNEXPECTED", detail) from exc
    write_json(out_dir / "collect.json", record)
    log(f"labels.jsonl: {rows_path} rows={run['rows']} frames={run['frames']} "
        f"sha256={record['tape']['sha256']}")
    log(f"collect.json: {out_dir / 'collect.json'}")
    log(f"done: {run['rows']}/{int(args.blocks)} blocks, "
        f"death_frame={run.get('death_frame')}, "
        f"hash_checks={run['hash_checks']}, mismatches={run['hash_mismatches']}, "
        f"wall={record['wall_s']}s")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def mode_verify(args: argparse.Namespace) -> int:
    doc = verify_tape(args.tape, expect_rows=args.expect_rows,
                      expect_origin=args.expect_origin)
    for line in format_verify(doc):
        print(line, flush=True)
    if args.out:
        target = pathlib.Path(str(args.out)).expanduser().resolve()
        bad = forbidden_part(target)
        if bad is not None:
            raise CollectError("OUT_DIR_FORBIDDEN",
                               f"--out {target} is inside {bad!r}")
        write_json(target, doc)
        print(f"wrote {target}", flush=True)
    if not doc["present"]:
        # exit 2 = the input could not be evaluated at all (same policy as the rest
        # of the family: a missing input file is not a verdict on a tape).
        return EXIT_UNUSABLE
    if not doc["ok"]:
        return EXIT_REFUSED
    return EXIT_OK


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def _fresh_subdir(base: pathlib.Path, prefix: str) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    for counter in range(1000):
        candidate = base / f"{prefix}-{os.getpid()}-{counter}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise CollectError("OK_UNEXPECTED", f"cannot find a free name under {base}")


def _collect_namespace(**overrides) -> argparse.Namespace:
    base = {
        "command": "collect", "out_dir": "", "seed": 77101, "stage": 0,
        "difficulty": 1, "blocks": 12, "policy": policy_spec_arg("random"),
        "engine": ENGINE_DEFAULT, "engine_sha256": ENGINE_SHA256,
        "data_source": DATA_SOURCE_DEFAULT, "work_dir": "", "tag": "",
        "stop_on_death": True, "overwrite_empty": False, "fake_parent": True,
        "fake_fault": "none", "fake_fault_block": 1, "fake_fault_step": 1,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _try_collect(**overrides) -> tuple[int, str | None]:
    namespace = _collect_namespace(**overrides)
    try:
        return mode_collect(namespace), None
    except CollectError as exc:
        return exc.exit_code, exc.code


def mode_selftest(args: argparse.Namespace) -> int:
    base = resolve_write_dir(args.dir, what="--dir")
    run_dir = _fresh_subdir(base, "selftest")
    work = run_dir / "work"
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: object = "") -> bool:
        checks.append({"name": str(name), "ok": bool(ok), "detail": str(detail)})
        return bool(ok)

    print(f"v829 collect selftest: {run_dir}", flush=True)
    print("  (no engine is started: the fake parent below is the only 'engine')",
          flush=True)

    # -- A. offline end to end: 12 blocks = 48 frames -------------------------
    tape_dir = run_dir / "tape12"
    rc, code = _try_collect(out_dir=str(tape_dir), work_dir=str(work),
                            blocks=12, seed=77101, tag="selftest-tape12")
    check("collect: 12 blocks exit 0", rc == 0 and code is None, f"rc={rc} code={code}")
    rows_path = tape_dir / "labels.jsonl"
    lines = ([line for line in rows_path.read_text(encoding="utf-8").splitlines()
              if line.strip()] if rows_path.is_file() else [])
    check("collect: labels.jsonl has 12 rows", len(lines) == 12, f"{len(lines)} rows")
    rows = [json.loads(line) for line in lines]
    verify_doc = verify_tape(rows_path, expect_rows=12)
    check("verify accepts the 48-frame tape", verify_doc["ok"],
          verify_doc.get("first_failure"))
    check("verify: rows=12 frames=48 anchors=13",
          (verify_doc.get("rows"), verify_doc.get("frames"),
           verify_doc.get("anchors")) == (12, 48, 13),
          f"rows={verify_doc.get('rows')} frames={verify_doc.get('frames')} "
          f"anchors={verify_doc.get('anchors')}")
    check("verify: target 48 is a recorded anchor",
          verify_doc.get("target") == 48 and "48" in (verify_doc.get("hashes") or {}),
          f"target={verify_doc.get('target')}")
    check("verify: every check ran", verify_doc.get("checks_ok") ==
          verify_doc.get("checks_run") and (verify_doc.get("checks_run") or 0) >= 12,
          f"{verify_doc.get('checks_ok')}/{verify_doc.get('checks_run')}")

    pair = None
    try:
        pair = m829()
        masks, hashes, meta = pair.read_tape(str(rows_path), 48)
        check("v829_pair.read_tape: 48 masks", len(masks) == 48, f"{len(masks)}")
        check("v829_pair.read_tape: 13 recorded hash frames", len(hashes) == 13,
              f"{len(hashes)}")
        check("v829_pair.read_tape: frames_built=48", meta.get("frames_built") == 48,
              f"{meta.get('frames_built')}")
        check("v829_pair.read_tape: target 48 is an anchor",
              48 in hashes and meta.get("target") == 48,
              f"target={meta.get('target')}")
        same = masks == [int(mask) for row in rows for mask in row["step_frames"]]
        check("v829_pair.read_tape: masks equal the tape's step_frames", same)
        check("v829_pair.read_tape: hash 48 == last hash_post",
              hashes.get(48) == rows[-1]["hash_post"], f"{hashes.get(48)}")
    except Exception as exc:                                       # noqa: BLE001
        check(f"v829_pair.read_tape usable ({type(exc).__name__}: {exc})", False)

    shard_path = tape_dir / "shard.json"
    shard = json.loads(shard_path.read_text(encoding="utf-8")) if shard_path.is_file() else {}
    check("shard.json: blocks=12 frames=48 death_frame=null hash_chain_ok=true",
          (shard.get("blocks"), shard.get("frames"), shard.get("death_frame"),
           shard.get("hash_chain_ok")) == (12, 48, None, True),
          f"{shard.get('blocks')}/{shard.get('frames')}/{shard.get('death_frame')}/"
          f"{shard.get('hash_chain_ok')}")
    check("shard.json: schema + stage/difficulty + fake_parent",
          (shard.get("schema"), shard.get("stage"), shard.get("difficulty"),
           shard.get("fake_parent")) == (SCHEMA_SHARD, 0, 1, True))
    collect_path = tape_dir / "collect.json"
    collect = json.loads(collect_path.read_text(encoding="utf-8")) \
        if collect_path.is_file() else {}
    check("collect.json: status=ok rows_written=12 hash_checks=13 mismatches=0",
          (collect.get("status"), collect.get("rows_written"),
           collect.get("hash_checks"), collect.get("hash_mismatches")) ==
          ("ok", 12, 13, 0),
          f"{collect.get('status')}/{collect.get('rows_written')}/"
          f"{collect.get('hash_checks')}/{collect.get('hash_mismatches')}")
    check("collect.json: tape_self_check ok",
          (collect.get("tape_self_check") or {}).get("ok") is True,
          collect.get("tape_self_check"))
    origin = rows[0]["hash_pre"] if rows else ""
    check("tape row 0: hash_pre is the reset frame-0 hash",
          origin == (collect.get("reset") or {}).get("frame0_hash"), origin)

    # -- B. three injected faults must each be refused -----------------------
    for label, key, kwargs, wanted_code in (
        ("fault advanced=0 -> STEP_NOT_ADVANCED", "advanced_zero",
         {"fake_fault": "advanced_zero", "fake_fault_block": 1, "fake_fault_step": 1},
         "STEP_NOT_ADVANCED"),
        ("fault hash=None -> HASH_MISSING", "hash_none",
         {"fake_fault": "hash_none", "fake_fault_block": 1, "fake_fault_step": 1},
         "HASH_MISSING"),
        ("fault chain break at block 2 -> STEP_NOT_ADVANCED", "chain",
         {"fake_fault": "chain", "fake_fault_block": 2, "fake_fault_step": 3},
         "STEP_NOT_ADVANCED"),
        ("fault HELLO role=env -> HELLO_FAILED", "hello_fail",
         {"fake_fault": "hello_fail"}, "HELLO_FAILED"),
    ):
        fault_rc, fault_code = _try_collect(
            out_dir=str(run_dir / f"fault-{key}"),
            work_dir=str(work), blocks=6, tag=f"selftest-{key}",
            **kwargs)
        check(f"{label} (exit 1)", fault_rc == 1 and fault_code == wanted_code,
              f"rc={fault_rc} code={fault_code}")
    chain_collect = (run_dir / "fault-chain" / "collect.json")
    chain_detail = ""
    if chain_collect.is_file():
        chain_detail = str(((json.loads(chain_collect.read_text("utf-8"))
                             .get("error") or {}).get("detail")) or "")
    check("fault chain break: the refusal names the broken hash chain",
          "chain is broken" in chain_detail, chain_detail[:160])

    # -- C. death handling ---------------------------------------------------
    death_rc, death_code = _try_collect(
        out_dir=str(run_dir / "death"), work_dir=str(work), blocks=8,
        tag="selftest-death", fake_fault="death", fake_fault_block=3,
        fake_fault_step=3)
    death_rows_path = run_dir / "death" / "labels.jsonl"
    death_rows = ([json.loads(line) for line in
                   death_rows_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()] if death_rows_path.is_file() else [])
    death_shard = json.loads((run_dir / "death" / "shard.json").read_text("utf-8")) \
        if (run_dir / "death" / "shard.json").is_file() else {}
    check("death on a complete block: exit 0, 4 rows, death_frame recorded",
          death_rc == 0 and len(death_rows) == 4
          and isinstance(death_shard.get("death_frame"), int),
          f"rc={death_rc} rows={len(death_rows)} "
          f"death_frame={death_shard.get('death_frame')}")
    check("death: the truncated tape still verifies",
          verify_tape(death_rows_path, expect_rows=len(death_rows))["ok"])
    nohash_rc, nohash_code = _try_collect(
        out_dir=str(run_dir / "death-nohash"), work_dir=str(work), blocks=8,
        tag="selftest-death-nohash", fake_fault="death_nohash",
        fake_fault_block=2, fake_fault_step=1)
    nohash_rows_path = run_dir / "death-nohash" / "labels.jsonl"
    nohash_lines = ([line for line in
                     nohash_rows_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()] if nohash_rows_path.is_file() else [])
    check("death with no hash: exit 0, the incomplete block is dropped (2 rows)",
          nohash_rc == 0 and len(nohash_lines) == 2,
          f"rc={nohash_rc} rows={len(nohash_lines)}")

    # -- D. extended RESET (stage/difficulty that the frozen reset cannot express)
    ext_dir = run_dir / "extended"
    ext_rc, ext_code = _try_collect(out_dir=str(ext_dir), work_dir=str(work),
                                    blocks=2, stage=4, difficulty=2,
                                    tag="selftest-extended")
    ext_rows = ([json.loads(line) for line in
                 (ext_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()] if (ext_dir / "labels.jsonl").is_file() else [])
    ext_collect = json.loads((ext_dir / "collect.json").read_text("utf-8")) \
        if (ext_dir / "collect.json").is_file() else {}
    check("extended RESET (stage 4 / difficulty 2) goes through ask()",
          ext_rc == 0 and ext_rows
          and all((row["stage"], row["difficulty"]) == (4, 2) for row in ext_rows)
          and (ext_collect.get("reset") or {}).get("mode") == "extended",
          f"rc={ext_rc} code={ext_code} reset={ext_collect.get('reset')}")

    # -- E. write fences -----------------------------------------------------
    for label, path in (
        (".tmp_v400_d1/x", REPO / ".tmp_v400_d1" / "x"),
        (".tmp_v410_s1c/x", REPO / ".tmp_v410_s1c" / "x"),
        (".tmp_v829_pair/x", REPO / ".tmp_v829_pair" / "x"),
        ("Document/x", REPO / "Document" / "x"),
    ):
        fence_rc, fence_code = _try_collect(out_dir=str(path), work_dir=str(work),
                                            blocks=1, tag="selftest-fence")
        check(f"fence: --out-dir {label} -> OUT_DIR_FORBIDDEN (exit 1)",
              fence_rc == 1 and fence_code == "OUT_DIR_FORBIDDEN",
              f"rc={fence_rc} code={fence_code}")
    outside = REPO / ".tmp_v829_coll_selftest" / "x"
    outside_rc, outside_code = _try_collect(out_dir=str(outside), work_dir=str(work),
                                            blocks=1, tag="selftest-outside")
    check("fence: --fake-parent outside .tmp_v829_info -> OUT_DIR_FORBIDDEN",
          outside_rc == 1 and outside_code == "OUT_DIR_FORBIDDEN",
          f"rc={outside_rc} code={outside_code}")
    busy = run_dir / "busy"
    busy.mkdir(parents=True, exist_ok=True)
    (busy / "somebody-elses.tape").write_text("not ours\n", encoding="utf-8")
    busy_rc, busy_code = _try_collect(out_dir=str(busy), work_dir=str(work),
                                      blocks=1, tag="selftest-busy")
    check("fence: a non-empty --out-dir -> OUT_DIR_NOT_EMPTY (exit 1)",
          busy_rc == 1 and busy_code == "OUT_DIR_NOT_EMPTY",
          f"rc={busy_rc} code={busy_code}")

    # -- F. finished tapes are never overwritten ------------------------------
    over_rc, over_code = _try_collect(out_dir=str(tape_dir), work_dir=str(work),
                                      blocks=12, tag="selftest-tape12",
                                      overwrite_empty=True)
    check("--overwrite-empty never overwrites a finished (status=ok) tape",
          over_rc == 1 and over_code == "OUT_DIR_NOT_EMPTY",
          f"rc={over_rc} code={over_code}")
    failed_dir = run_dir / "fault-advanced_zero"
    leftover = (failed_dir / "collect.json")
    retry_rc, retry_code = _try_collect(out_dir=str(failed_dir), work_dir=str(work),
                                        blocks=3, tag="selftest-retry",
                                        overwrite_empty=True)
    check("--overwrite-empty recovers an earlier FAILED run's leftover",
          retry_rc == 0 and retry_code is None and leftover.is_file(),
          f"rc={retry_rc} code={retry_code}")

    # -- G. tags -------------------------------------------------------------
    tag_rc, tag_code = _try_collect(out_dir=str(run_dir / "tag-bad"),
                                    work_dir=str(work), blocks=1, tag="a")
    check("tag: --tag 'a' -> TAG_REQUIRED (exit 1)",
          tag_rc == 1 and tag_code == "TAG_REQUIRED", f"rc={tag_rc} code={tag_code}")
    _try_collect(out_dir=str(run_dir / "tag-one"), work_dir=str(work), blocks=1,
                 tag="selftest-dup-tag")
    dup_rc, dup_code = _try_collect(out_dir=str(run_dir / "tag-two"),
                                    work_dir=str(work), blocks=1,
                                    tag="selftest-dup-tag")
    check("tag: the same --tag on a second cell -> TAG_NOT_UNIQUE (exit 1)",
          dup_rc == 1 and dup_code == "TAG_NOT_UNIQUE", f"rc={dup_rc} code={dup_code}")
    same_rc, same_code = _try_collect(out_dir=str(run_dir / "tag-one"),
                                      work_dir=str(work), blocks=1,
                                      tag="selftest-dup-tag")
    check("tag: the same --tag on the same cell is a retry, not a collision",
          same_code != "TAG_NOT_UNIQUE", f"rc={same_rc} code={same_code}")

    # -- H. audio gate -------------------------------------------------------
    muted_cfg = run_dir / "muted.th08.cfg"
    try:
        m_cfg().main(["--out", str(muted_cfg)])
        muted = audio_evidence(muted_cfg)
        check("audio: a generated th08.cfg is four-zero muted",
              muted["all_muted"] is True
              and all(value == 0 for value in muted["fields"].values()),
              json.dumps(muted["fields"]))
    except Exception as exc:                                       # noqa: BLE001
        check(f"audio: a generated th08.cfg is four-zero muted ({exc})", False)
    audible_cfg = run_dir / "audible.th08.cfg"
    raw = bytearray(muted_cfg.read_bytes())
    raw[0x20] = 1                      # playSounds
    raw[0x1F] = 1                      # musicMode
    audible_cfg.write_bytes(bytes(raw))
    audible = audio_evidence(audible_cfg)
    check("audio: a tampered cfg is NOT muted", audible["all_muted"] is False,
          json.dumps(audible["fields"]))
    try:
        assert_audio_muted(audible, when="selftest")
        check("audio: AUDIO_NOT_MUTED is raised for it", False, "no exception")
    except CollectError as exc:
        check("audio: AUDIO_NOT_MUTED is raised for it (exit 1)",
              exc.code == "AUDIO_NOT_MUTED" and exc.exit_code == 1,
              f"{exc.code}/{exc.exit_code}")

    # -- I. the error-code table itself --------------------------------------
    wanted_table = {
        "OUT_DIR_FORBIDDEN": 1, "OUT_DIR_NOT_EMPTY": 1, "ENGINE_MISSING": 2,
        "ENGINE_SHA_MISMATCH": 1, "HELLO_FAILED": 1, "RESET_FAILED": 1,
        "STEP_NOT_ADVANCED": 1, "HASH_MISSING": 1, "AUDIO_NOT_MUTED": 1,
        "TAG_REQUIRED": 1, "TAG_NOT_UNIQUE": 1, "OK_UNEXPECTED": 2,
    }
    check("error table: all 12 codes with the specified exit statuses",
          dict(CODE_EXIT) == wanted_table,
          f"{sorted(set(CODE_EXIT) ^ set(wanted_table))}")

    ok_n = sum(1 for entry in checks if entry["ok"])
    total = len(checks)
    for entry in checks:
        mark = "ok  " if entry["ok"] else "FAIL"
        detail = f"  [{entry['detail']}]" if entry["detail"] and not entry["ok"] else ""
        print(f"  {mark} {entry['name']}{detail}", flush=True)
    print(f"  artifacts {run_dir}", flush=True)
    if ok_n == total:
        print(f"SELFTEST: ok {ok_n}/{total}", flush=True)
        return EXIT_OK
    print(f"SELFTEST: FAILED {ok_n}/{total}", flush=True)
    return EXIT_REFUSED


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v829_collect_source.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"{PRODUCER}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    collect = sub.add_parser(
        "collect", help="collect a native 4-frame source tape (starts the engine)",
        description=("Collect one 4-frame-per-decision tape from a fresh RESET: "
                     "HELLO -> RESET -> blocks x 4 x STEP, writing labels.jsonl / "
                     "shard.json / collect.json into a new directory. --fake-parent "
                     "runs the same code path offline (no engine, .tmp_v829_info only)."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    collect.add_argument("--out-dir", required=True,
                         help="new directory for labels.jsonl / shard.json / "
                              "collect.json (existing non-empty dirs are refused)")
    collect.add_argument("--seed", type=int, required=True,
                         help="RESET seed (also the agent's generator seed)")
    collect.add_argument("--stage", type=int, default=0,
                         help="engine stage index, 0-based (default 0)")
    collect.add_argument("--difficulty", type=int, default=1,
                         help="engine difficulty 0..3 (default 1 = what the frozen "
                              "ForkParent.reset sends: RESET 0 1 0 <seed> 0)")
    collect.add_argument("--blocks", type=int, required=True,
                         help="number of 4-frame decision blocks to collect")
    collect.add_argument("--policy", type=policy_spec_arg, default=policy_spec_arg("stay"),
                         metavar="SPEC",
                         help="policy:<ckpt> | random | stay | const:N (default stay)")
    collect.add_argument("--engine", default=ENGINE_DEFAULT,
                         help=f"engine binary (default {ENGINE_DEFAULT})")
    collect.add_argument("--engine-sha256", default=ENGINE_SHA256,
                         help=f"expected engine sha256 (default {ENGINE_SHA256})")
    collect.add_argument("--data-source", default=DATA_SOURCE_DEFAULT,
                         help=f"engine data source directory (default "
                              f"{DATA_SOURCE_DEFAULT})")
    collect.add_argument("--work-dir", default=str(WORK_DEFAULT),
                         help="private data dir / log / tag ledger root "
                              "(default .tmp_v829_info/s15/work)")
    collect.add_argument("--tag", default="",
                         help="cell tag (default: derived from the cell + out-dir)")
    collect.add_argument("--stop-on-death", action=argparse.BooleanOptionalAction,
                         default=True,
                         help="stop at the first exit/game_over (a single tape must "
                              "stay one continuous hash chain, so a death always "
                              "ends the tape; this flag records the intent)")
    collect.add_argument("--overwrite-empty", action="store_true",
                         help="allow writing into a directory that holds only this "
                              "tool's three artifacts from an earlier FAILED run")
    collect.add_argument("--fake-parent", action="store_true",
                         help="offline: use the deterministic fake parent (no engine); "
                              "only allowed under .tmp_v829_info/")
    collect.add_argument("--fake-fault", default="none",
                         choices=("none", "advanced_zero", "hash_none", "chain",
                                  "death", "death_nohash", "hello_fail"),
                         help="inject a fault into the fake parent (selftest only)")
    collect.add_argument("--fake-fault-block", type=int, default=1,
                         help="block index of the injected fault (default 1)")
    collect.add_argument("--fake-fault-step", type=int, default=1,
                         help="frame offset inside that block, 0..3 (default 1)")
    collect.set_defaults(handler=mode_collect)

    verify = sub.add_parser(
        "verify", help="verify a 4-frame tape offline (no engine)",
        description=("Read a tape with the same rules as v829_pair.read_tape (and a "
                     "few more) and print every check. Exit 0 when everything holds, "
                     "exit 1 on the first non-compliant rule, exit 2 if unreadable."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    verify.add_argument("--tape", required=True, help="path to a labels.jsonl")
    verify.add_argument("--expect-rows", type=int, default=None,
                        help="assert the tape has exactly this many rows")
    verify.add_argument("--expect-origin", default=None,
                        help="assert row 0's hash_pre equals this frame-0 hash")
    verify.add_argument("--out", default="", help="optional path for the JSON report")
    verify.set_defaults(handler=mode_verify)

    selftest = sub.add_parser(
        "selftest", help="offline self-test (no engine): end-to-end + faults + fences",
        description=("Run the collect loop with the fake parent, verify the 48-frame "
                     "tape with this tool AND with v829_pair.read_tape, inject the "
                     "three faults that must be refused, and check the path fences, "
                     "the tag ledger and the audio gate."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    selftest.add_argument("--dir", default=str(REPO / ".tmp_v829_info" / "s15"),
                          help="artifact root (default .tmp_v829_info/s15)")
    selftest.set_defaults(handler=mode_selftest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_OK
    try:
        return int(handler(args))
    except CollectError as exc:
        print(f"{LOG_PREFIX} REFUSED [{exc.code}] {exc.detail}", flush=True)
        return exc.exit_code
    except KeyboardInterrupt:
        print(f"{LOG_PREFIX} REFUSED [OK_UNEXPECTED] interrupted", flush=True)
        return EXIT_UNUSABLE
    except SystemExit:
        raise
    except Exception as exc:                                      # noqa: BLE001
        print(f"{LOG_PREFIX} REFUSED [OK_UNEXPECTED] {type(exc).__name__}: {exc}",
              flush=True)
        traceback.print_exc()
        return EXIT_UNUSABLE


if __name__ == "__main__":
    raise SystemExit(main())
