"""WP5: record real RGBA cache, replay identical trajectory in independent env.

Run inside WSL under xvfb-run; never creates a path that already exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import select
import struct
import subprocess
import sys
import time
from pathlib import Path

BINARY = Path('/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern')
SOURCE = Path('/root/v400-data/game')
ROOT = Path('/mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp5_pixel_rebuild_20261108')
FRAMES = 100
PIXEL_BYTES = 640 * 480 * 4
HEADER = struct.Struct('<IIQQIIII')
MASKS = [0] * FRAMES  # The actual recorded one-mask-per-frame trajectory.


class Blocked(Exception):
    pass


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def create_data(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise Blocked(f'data-dir already exists: {path}')
    path.mkdir()
    for name in ('th08.cfg', 'score.dat'):
        (path / name).write_bytes((SOURCE / name).read_bytes())
    for name in ('th08.dat', 'thbgm.dat'):
        (path / name).symlink_to(SOURCE / name)


class Engine:
    def __init__(self, data: Path):
        self.data = data
        self.argv = [str(BINARY), '--data-dir', str(data), '--rl-control', '0,1', '--rl-role', 'env']
        env = {**os.environ, 'SDL_AUDIODRIVER': 'dummy', 'LIBGL_ALWAYS_SOFTWARE': '1', 'LP_NUM_THREADS': '1'}
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, bufsize=0)
        self.buffer = b''
        self.log = []

    def ask(self, command: str) -> dict[str, str]:
        record = {'command': command, 'reply': None}
        self.log.append(record)
        self.proc.stdin.write((command + '\n').encode('ascii'))
        self.proc.stdin.flush()
        deadline = time.monotonic() + 30
        while b'\n' not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Blocked(f'timeout on {command}, exit={self.proc.poll()}')
            ready, _, _ = select.select([self.proc.stdout], [], [], min(remaining, .2))
            if ready:
                chunk = os.read(self.proc.stdout.fileno(), 8192)
                if not chunk:
                    raise Blocked(f'pipe EOF on {command}, exit={self.proc.poll()}')
                self.buffer += chunk
            elif self.proc.poll() is not None:
                raise Blocked(f'engine exit on {command}, exit={self.proc.returncode}')
        line, self.buffer = self.buffer.split(b'\n', 1)
        reply = line.decode('utf-8', 'replace').strip()
        record['reply'] = reply
        if not reply.startswith('OK '):
            raise Blocked(f'{command}: {reply}')
        return dict(part.split('=', 1) for part in reply.split()[1:] if '=' in part)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.ask('QUIT')
            except (Blocked, OSError):
                pass
        try:
            code = self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait(timeout=3)
        stderr = self.proc.stderr.read().decode('utf-8', 'replace')
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            stream.close()
        return {'argv': self.argv, 'exit_code': code, 'stderr': stderr, 'stdout_protocol': self.log}


def capture(engine: Engine, shm_name: str, cache=None, expected=None):
    if engine.ask('HELLO').get('role') != 'env':
        raise Blocked('HELLO does not confirm actual env role')
    reset = engine.ask('RESET 0 1 0 1 0')
    if reset.get('ready') != '1':
        raise Blocked('RESET did not reach StageStart')
    if engine.ask('READBACK 1').get('readback') != '1':
        raise Blocked('pixel readback unavailable')
    shm = engine.ask('SHM /' + shm_name)
    if shm.get('active') != '1' or int(shm.get('bytes', -1)) != 192 + PIXEL_BYTES:
        raise Blocked(f'pixel SHM unavailable: {shm}')
    shm_path = Path('/dev/shm') / shm_name
    hashes = []
    state_hashes = []
    different = []
    for index, mask in enumerate(MASKS):
        step = engine.ask(f'STEP {mask} 1')
        if int(step.get('advanced', -1)) != 1 or int(step.get('frame', -1)) != index + 1:
            raise Blocked(f'not exactly one real frame at {index + 1}: {step}')
        pixels = engine.ask('PIXELS')
        if pixels.get('pixels') != '640x480' or int(pixels.get('nonzero', 0)) <= 0:
            raise Blocked(f'pixel capability missing at {index + 1}: {pixels}')
        raw = shm_path.read_bytes()
        if len(raw) != 192 + PIXEL_BYTES:
            raise Blocked(f'pixel SHM size wrong: {len(raw)}')
        magic, version, frame, _, width, height, _, flags = HEADER.unpack_from(raw)
        if (magic, version, frame, width, height) != (0x30303456, 1, index + 1, 640, 480) or not flags & 2:
            raise Blocked(f'pixel SHM header invalid at {index + 1}')
        rgba = raw[192:]  # Exact unmodified engine RGBA, not a PNG or a checksum reply.
        digest = sha(rgba)
        hashes.append(digest)
        state_hashes.append(step.get('hash'))
        if cache is not None:
            cache.write(rgba)
        if expected is not None:
            baseline = expected.read(PIXEL_BYTES)
            if len(baseline) != PIXEL_BYTES:
                raise Blocked(f'original RGBA cache truncated at {index + 1}')
            if baseline != rgba:
                different.append({'frame': index + 1, 'original_rgba_sha256': sha(baseline),
                                  'replayed_rgba_sha256': digest,
                                  'different_byte_count': sum(a != b for a, b in zip(baseline, rgba))})
    if expected is not None and expected.read(1):
        raise Blocked('original RGBA cache contains extra bytes')
    return {'frames': len(hashes), 'rgba_sha256': hashes, 'state_hashes': state_hashes,
            'different_frames': different}


def run():
    report = {'schema': 'v400-wp5-real-pixel-rebuild/1', 'status': 'blocked',
              'binary': str(BINARY), 'binary_sha256': None, 'harness_commit': None,
              'source_tree_dirty': None, 'trajectory': {'start': [0, 1, 0, 1, 0], 'masks': MASKS},
              'processes': [], 'captures': []}
    if ROOT.exists() or ROOT.is_symlink():
        raise Blocked(f'evidence output already exists: {ROOT}')
    ROOT.mkdir()
    try:
        if not BINARY.is_file() or not SOURCE.is_dir():
            raise Blocked('real binary or data template unavailable')
        report['binary_sha256'] = sha(BINARY.read_bytes())
        repo = '/root/Touhou_RL/third_party/th08-rl'
        report['harness_commit'] = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
        report['source_tree_dirty'] = bool(subprocess.check_output(['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'], text=True).strip())
        first, second = ROOT / 'data-original', ROOT / 'data-replay'
        create_data(first)
        create_data(second)
        report['data_dirs'] = [str(first), str(second)]
        report['data_template_sha256'] = {name: sha((SOURCE / name).read_bytes()) for name in ('th08.cfg', 'score.dat')}
        report['trajectory_sha256'] = sha(json.dumps(report['trajectory'], sort_keys=True, separators=(',', ':')).encode('utf-8'))
        cache_path = ROOT / 'original-rgba640x480x4.bin'
        report['original_rgba_cache'] = str(cache_path)
        for index, directory in enumerate((first, second)):
            engine = Engine(directory)
            try:
                with open(cache_path, 'wb') if index == 0 else open(cache_path, 'rb') as cache:
                    capture_row = capture(engine, f'v400wp5pixel{os.getpid()}{index}',
                                          cache=cache if index == 0 else None,
                                          expected=cache if index == 1 else None)
                report['captures'].append(capture_row)
            finally:
                report['processes'].append(engine.close())
                (ROOT / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if any(row['exit_code'] != 0 for row in report['processes']):
            raise Blocked('real engine did not cleanly exit')
        a, b = report['captures']
        if a['state_hashes'] != b['state_hashes']:
            raise Blocked('same trajectory has divergent game-state hashes')
        report['original_cache_sha256'] = sha(cache_path.read_bytes())
        report['different_frame_count'] = len(b['different_frames'])
        report['status'] = 'pass' if not b['different_frames'] else 'blocked'
        if b['different_frames']:
            report['reason'] = 'original and regenerated unmodified RGBA bytes differ'
    except (Blocked, OSError, ValueError, subprocess.CalledProcessError) as error:
        report['status'] = 'blocked'
        report['reason'] = str(error)
    finally:
        (ROOT / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'frames': [c['frames'] for c in report['captures']],
                      'different_frame_count': report.get('different_frame_count'),
                      'reason': report.get('reason'), 'report': str(ROOT / 'report.json')}))
    return 0 if report['status'] == 'pass' else 3


if __name__ == '__main__':
    try:
        sys.exit(run())
    except (Blocked, OSError) as error:
        print('BLOCKED:', error, file=sys.stderr)
        sys.exit(3)
