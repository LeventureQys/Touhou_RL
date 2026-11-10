"""Fail-closed WP5 real-engine acceptance; run inside WSL using actual binary."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time


class Blocked(Exception):
    pass


class Engine:
    def __init__(self, binary, data, role, timeout):
        self.argv = [str(binary), '--data-dir', str(data), '--rl-control', '0,1', '--rl-role', role]
        env = {**os.environ, 'SDL_AUDIODRIVER': 'dummy', 'LIBGL_ALWAYS_SOFTWARE': '1', 'LP_NUM_THREADS': '1'}
        if role == 'shadow':
            env['DISPLAY'] = ''
        self.process = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, env=env, bufsize=0)
        self.timeout = timeout
        self.buffer = b''
        self.log = []

    def ask(self, command):
        self.log.append({'command': command, 'reply': None})
        try:
            self.process.stdin.write((command + '\n').encode('ascii'))
            self.process.stdin.flush()
        except OSError as error:
            raise Blocked(f'pipe error: {error}; exit={self.process.poll()}') from error
        deadline = time.monotonic() + self.timeout
        while b'\n' not in self.buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                raise Blocked(f'timeout on {command}')
            ready, _, _ = select.select([self.process.stdout], [], [], min(left, .25))
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 8192)
                if not chunk:
                    raise Blocked(f'pipe closed on {command}; exit={self.process.poll()}')
                self.buffer += chunk
            elif self.process.poll() is not None:
                raise Blocked(f'engine exited on {command}; exit={self.process.returncode}')
        line, self.buffer = self.buffer.split(b'\n', 1)
        reply = line.decode('utf-8', 'replace').strip()
        self.log[-1]['reply'] = reply
        if not reply.startswith('OK '):
            raise Blocked(f'{command}: {reply}')
        return dict(item.split('=', 1) for item in reply.split()[1:] if '=' in item)

    def close(self):
        if self.process.poll() is None:
            try:
                self.ask('QUIT')
            except Blocked:
                pass
        try:
            return self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return self.process.wait(timeout=3)
        finally:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream:
                    stream.close()


def shadow_10000(shadow, main):
    roles = [shadow.ask('HELLO').get('role'), main.ask('HELLO').get('role')]
    if roles != ['shadow', 'env'] or shadow.ask('DEBUG').get('threads') != '1':
        raise Blocked(f'real one-thread shadow/environment not demonstrated: {roles}')
    for engine in (shadow, main):
        if engine.ask('RESET 0 1 0 1 0').get('ready') != '1':
            raise Blocked('StageStart not ready')
    hashes = []
    for offset in range(0, 10000, 60):
        n = min(60, 10000 - offset)
        for engine in (shadow, main):
            step = engine.ask(f'STEP 0 {n}')
            if int(step.get('advanced', -1)) != n:
                raise Blocked(f'incomplete STEP at {offset}: {step}')
        lhs, rhs = shadow.ask('HASH'), main.ask('HASH')
        if lhs.get('hash') != rhs.get('hash') or lhs.get('frames') != rhs.get('frames'):
            raise Blocked(f'hash divergence at {offset+n}: {lhs} vs {rhs}')
        hashes.append({'frame': offset+n, 'hash': lhs['hash']})
    return {'frames': 10000, 'checkpoints': len(hashes), 'hashes': hashes}


def branch_timing(shadow):
    if shadow.ask('HELLO').get('role') != 'shadow' or shadow.ask('DEBUG').get('threads') != '1':
        raise Blocked('real one-thread shadow not demonstrated')
    if shadow.ask('RESET 0 1 0 1 0').get('ready') != '1':
        raise Blocked('StageStart not ready')
    rows = []
    for i in range(20):
        before = shadow.ask('FRAME')
        start = time.perf_counter_ns()
        result = shadow.ask('BRANCH 0 300')
        wall_us = (time.perf_counter_ns() - start) / 1000
        after = shadow.ask('FRAME')
        if result.get('branch') != '1' or int(result.get('advanced', -1)) != 300 or before.get('frame') != after.get('frame'):
            raise Blocked(f'branch incomplete or changed parent: {result}')
        if not all(key in result for key in ('pid', 'fork_us', 'child_us')):
            raise Blocked(f'engine timing absent: {result}')
        rows.append({'sample': i, 'pid': result['pid'], 'parent_frame': before['frame'],
                     'engine_fork_us': int(result['fork_us']), 'child_us': int(result['child_us']),
                     'child_us_per_frame': int(result['child_us']) / 300, 'request_wall_us': wall_us})
        if i < 19 and int(shadow.ask('STEP 0 1').get('advanced', -1)) != 1:
            raise Blocked('parent STEP failed')
    return {'samples': len(rows), 'branch_frames': 300, 'rows': rows,
            'mean_engine_fork_us': sum(r['engine_fork_us'] for r in rows) / len(rows),
            'mean_child_us_per_frame': sum(r['child_us_per_frame'] for r in rows) / len(rows),
            'timing_caveat': 'engine fork_us includes child execution and wait; not isolated fork syscall'}


def pixel_gate(cache, trajectory, commit):
    if not cache or not trajectory or not cache.is_file() or not trajectory.is_file():
        raise Blocked('real original pixel cache and input trajectory unavailable; byte regeneration forbidden')
    metadata = json.loads(trajectory.read_text(encoding='utf-8'))
    if metadata.get('harness_commit') != commit or metadata.get('source_verified') is not True:
        raise Blocked('original trajectory provenance or matching harness commit absent')
    raise Blocked('real original cache format regeneration not implemented; synthetic pixels forbidden')


def run(args):
    binary, data = Path(args.binary), Path(args.data_dir)
    report = {'schema': 'v400-wp5-real-acceptance/1', 'binary': str(binary), 'data_dir': str(data),
              'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None,
              'harness_commit': args.harness_commit, 'gates': {}, 'processes': []}
    for gate in ('shadow-10000', 'branch-timing'):
        engines = []
        try:
            if not binary.is_file() or not data.is_dir():
                raise Blocked('real WSL engine binary or data directory unavailable')
            engines.append(Engine(binary, data, 'shadow', args.timeout))
            if gate == 'shadow-10000':
                engines.append(Engine(binary, data, 'env', args.timeout))
                result = shadow_10000(*engines)
            else:
                result = branch_timing(engines[0])
            report['gates'][gate] = {'status': 'pass', **result}
        except (Blocked, OSError, ValueError) as error:
            report['gates'][gate] = {'status': 'blocked', 'reason': str(error)}
        finally:
            for engine in engines:
                exit_code = engine.close()
                report['processes'].append({'gate': gate, 'command': engine.argv, 'exit_code': exit_code,
                                            'protocol': engine.log})
                if exit_code != 0 and report['gates'][gate]['status'] == 'pass':
                    report['gates'][gate] = {'status': 'blocked', 'reason': f'engine exit {exit_code}'}
    try:
        pixel_gate(args.pixel_cache, args.trajectory, args.harness_commit)
    except (Blocked, OSError, ValueError) as error:
        report['gates']['pixel-byte'] = {'status': 'blocked', 'reason': str(error),
                                         'pixel_cache': str(args.pixel_cache) if args.pixel_cache else None,
                                         'trajectory': str(args.trajectory) if args.trajectory else None}
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'data-dir', 'harness-commit', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--pixel-cache', type=Path)
    p.add_argument('--trajectory', type=Path)
    p.add_argument('--timeout', type=float, default=60)
    args = p.parse_args(argv)
    report = run(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'gates': {key: row['status'] for key, row in report['gates'].items()}, 'out': str(path)}, ensure_ascii=False))
    return 0 if all(row['status'] == 'pass' for row in report['gates'].values()) else 3


if __name__ == '__main__':
    raise SystemExit(main())
