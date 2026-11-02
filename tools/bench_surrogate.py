"""Micro-benchmark: vectorized TH08 player+bullet-dynamics surrogate.

Measures how many *player decisions per second* a minimal, fully vectorized
surrogate of Touhou TH08 player+bullet dynamics can sustain, on CPU (numpy,
batch=1) and on the local GPU (torch, batched, eager + CUDA-graph).

The modelled state is deliberately minimal and matches the game's documented
physics constants:
  * field 384 x 448 px
  * player speed 4.0 / 2.8284 / 2.0 / 1.4142 px/frame (fast/slow, cardinal/diagonal)
  * player hitbox radius 0.825 px
  * bullets move linearly at constant velocity; circle-vs-circle collision
Bullet *spawns* are replayed from a precomputed schedule (as if recorded from the
real game), so no enemy/AI logic is simulated.

Usage: python tools/bench_surrogate.py
"""

import json
import statistics
import time

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

FIELD_W, FIELD_H = 384.0, 448.0
HITBOX_R = 0.825
SPEEDS = np.array(
    [4.0, 2.8284, 2.0, 1.4142], dtype=np.float32
)  # [fast-card, fast-diag, slow-card, slow-diag]

# 9 actions: 8 directions + "stay"; each with fast/slow variants -> use 8 dirs x 2 modes
DIRS = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.7071, 0.7071],
        [0.7071, -0.7071],
        [-0.7071, 0.7071],
        [-0.7071, -0.7071],
    ],
    dtype=np.float32,
)

CAP = 1024         # bullet pool capacity (real TH08 pool is 1537 slots)
SPAWN_PER_FRAME = 3
SCHED_LEN = 4096   # spawn-schedule length (cycled)


def make_schedule(rng, n_sched=SCHED_LEN, spawn_per_frame=SPAWN_PER_FRAME):
    """Precomputed spawn schedule: (S, T, 5) = x, y, vx, vy, radius."""
    ang = rng.uniform(0, 2 * np.pi, size=(n_sched, spawn_per_frame)).astype(np.float32)
    spd = rng.uniform(1.0, 4.5, size=(n_sched, spawn_per_frame)).astype(np.float32)
    x = rng.uniform(0, FIELD_W, size=(n_sched, spawn_per_frame)).astype(np.float32)
    y = rng.uniform(-40, 0, size=(n_sched, spawn_per_frame)).astype(np.float32)
    r = rng.uniform(2.0, 8.0, size=(n_sched, spawn_per_frame)).astype(np.float32)
    sched = np.stack(
        [x, y, np.cos(ang) * spd, np.sin(ang) * spd, r], axis=-1
    ).astype(np.float32)
    return sched


# --------------------------------------------------------------------------- #
# numpy, batch = 1 (what a "live" single-env surrogate would cost per step)
# --------------------------------------------------------------------------- #
def bench_numpy(batch=1, steps=20000, cap=CAP, sched=None):
    rng = np.random.default_rng(0)
    sched = sched if sched is not None else make_schedule(rng)
    T, S, _ = sched.shape

    pos = np.tile(np.array([[192.0, 400.0]], np.float32), (batch, 1))
    bpos = np.zeros((batch, cap, 2), np.float32)
    bvel = np.zeros((batch, cap, 2), np.float32)
    brad = np.zeros((batch, cap), np.float32)
    balive = np.zeros((batch, cap), np.float32)
    done = np.zeros(batch, np.float32)
    rr = np.arange(batch)

    t0 = time.perf_counter()
    for step in range(steps):
        act = (step * 7) % 18
        d = DIRS[act // 2]
        sp = SPEEDS[(act % 2) * 2 + (1 if (act // 2) in (5, 6, 7, 8) else 0)]
        vel = d * sp
        pos = pos + vel
        np.clip(pos[:, 0], 8.0, FIELD_W - 8.0, out=pos[:, 0])
        np.clip(pos[:, 1], 8.0, FIELD_H - 8.0, out=pos[:, 1])

        sp_frame = sched[step % T]                 # (S, 5)
        for j in range(S):
            slot = (step * S + j) % cap
            bpos[:, slot, :] = sp_frame[j, 0:2]
            bvel[:, slot, :] = sp_frame[j, 2:4]
            brad[:, slot] = sp_frame[j, 4]
            balive[:, slot] = 1.0

        bpos += bvel
        oob = (
            (bpos[:, :, 0] < -32)
            | (bpos[:, :, 0] > FIELD_W + 32)
            | (bpos[:, :, 1] < -32)
            | (bpos[:, :, 1] > FIELD_H + 32)
        )
        balive *= (~oob).astype(np.float32)

        dx = bpos[:, :, 0] - pos[:, 0:1]
        dy = bpos[:, :, 1] - pos[:, 1:2]
        rad = brad + HITBOX_R
        hit = (dx * dx + dy * dy) < (rad * rad)
        killed = (hit & (balive > 0)).any(axis=1)
        done = np.maximum(done, killed.astype(np.float32))
    dt = time.perf_counter() - t0
    return batch * steps / dt, dt, done.mean()


# --------------------------------------------------------------------------- #
# torch (GPU or CPU), batched, eager
# --------------------------------------------------------------------------- #
def _make_torch(dev, batch, cap, sched_np):
    sched = torch.from_numpy(sched_np).to(dev)
    dirs = torch.from_numpy(DIRS).to(dev)
    speeds = torch.from_numpy(SPEEDS).to(dev)
    pos = torch.empty((batch, 2), device=dev).uniform_(120, 260)
    pos[:, 1] = 400.0
    bpos = torch.zeros((batch, cap, 2), device=dev)
    bvel = torch.zeros((batch, cap, 2), device=dev)
    brad = torch.zeros((batch, cap), device=dev)
    balive = torch.zeros((batch, cap), device=dev)
    done = torch.zeros(batch, device=dev)
    env_off = torch.arange(batch, device=dev) * 97 % sched.shape[1]
    return dict(
        sched=sched, dirs=dirs, speeds=speeds, pos=pos, bpos=bpos, bvel=bvel,
        brad=brad, balive=balive, done=done, env_off=env_off,
    )


def step_torch(st, step, batch, cap):
    sched, dirs, speeds = st["sched"], st["dirs"], st["speeds"]
    T = sched.shape[0]
    act = (step * 7) % 18
    d = dirs[act // 2]
    sp = speeds[(act % 2) * 2 + (1 if (act // 2) in (5, 6, 7, 8) else 0)]
    pos = st["pos"] + d * sp
    pos[:, 0].clamp_(8.0, FIELD_W - 8.0)
    pos[:, 1].clamp_(8.0, FIELD_H - 8.0)
    st["pos"] = pos

    idx = (st["env_off"] + step) % T
    sf = sched.index_select(0, idx)          # (B, S, 5)
    S = sched.shape[1]
    for j in range(S):
        slot = (step * S + j) % cap
        st["bpos"][:, slot, :] = sf[:, j, 0:2]
        st["bvel"][:, slot, :] = sf[:, j, 2:4]
        st["brad"][:, slot] = sf[:, j, 4]
        st["balive"][:, slot] = 1.0

    st["bpos"] += st["bvel"]
    bp = st["bpos"]
    oob = (
        (bp[..., 0] < -32) | (bp[..., 0] > FIELD_W + 32)
        | (bp[..., 1] < -32) | (bp[..., 1] > FIELD_H + 32)
    )
    st["balive"] *= (~oob).to(st["balive"].dtype)

    dx = bp[..., 0] - st["pos"][:, 0:1]
    dy = bp[..., 1] - st["pos"][:, 1:2]
    rad = st["brad"] + HITBOX_R
    hit = (dx * dx + dy * dy) < rad * rad
    killed = (hit & (st["balive"] > 0)).any(dim=1)
    st["done"] = torch.maximum(st["done"], killed.to(st["done"].dtype))
    return killed


def bench_torch(dev, batch, steps, cap=CAP, sched=None, use_graph=True,
                policy=None):
    sched = sched if sched is not None else make_schedule(np.random.default_rng(0))
    st = _make_torch(dev, batch, cap, sched)
    for s in range(50):  # warmup
        step_torch(st, s, batch, cap)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    if use_graph and dev.type == "cuda":
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for k in range(20):
                    step_torch(st, k, batch, cap)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for k in range(steps):
                step_torch(st, k, batch, cap)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return batch * steps / dt, dt

    t0 = time.perf_counter()
    for k in range(steps):
        step_torch(st, k, batch, cap)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch * steps / dt, dt


# --------------------------------------------------------------------------- #
# policy nets
# --------------------------------------------------------------------------- #
def make_mlp(n_in, dev):
    m = torch.nn.Sequential(
        torch.nn.Linear(n_in, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 16),
    ).to(dev)
    return m.eval()


def make_cnn(dev):
    m = torch.nn.Sequential(
        torch.nn.Conv2d(4, 32, 8, 4), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 4, 2), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, 2), torch.nn.ReLU(),
        torch.nn.Flatten(), torch.nn.Linear(64 * 4 * 4, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 16),
    ).to(dev)
    return m.eval()


def bench_net(net, x, iters=200):
    for _ in range(20):
        net(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        net(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return x.shape[0] * iters / dt


def main():
    out = {}
    rng = np.random.default_rng(0)
    sched = make_schedule(rng)

    # --- CPU, numpy, single env -------------------------------------------
    for b in (1,):
        nps, dt, frac = bench_numpy(batch=b, steps=20000, sched=sched)
        out[f"numpy_cpu_b{b}"] = round(nps, 1)
        print(f"numpy CPU, batch={b}: {nps:,.0f} steps/s ({dt:.2f}s for 20k) "
              f"death_frac={frac:.3f}")

    if torch is None:
        print(json.dumps(out, indent=2))
        return

    out["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        print("GPU:", out["gpu"])

    # --- CPU torch, batch 1 & 64 ------------------------------------------
    for b in (1, 64):
        s, _ = bench_torch(torch.device("cpu"), b, steps=2000, sched=sched,
                           use_graph=False)
        out[f"torch_cpu_b{b}"] = round(s, 1)
        print(f"torch CPU, batch={b}: {s:,.0f} steps/s")

    if not torch.cuda.is_available():
        print(json.dumps(out, indent=2))
        return

    dev = torch.device("cuda")
    # --- GPU eager --------------------------------------------------------
    for b in (256, 1024, 4096, 16384):
        s, _ = bench_torch(dev, b, steps=300, sched=sched, use_graph=False)
        out[f"torch_gpu_eager_b{b}"] = round(s, 0)
        print(f"GPU eager, batch={b}: {s:,.0f} steps/s")

    # --- GPU CUDA-graph ---------------------------------------------------
    for b in (256, 1024, 4096, 16384, 65536):
        steps = max(16, min(2000, 2_000_000 // b))
        s, dt = bench_torch(dev, b, steps=steps, sched=sched, use_graph=True)
        out[f"torch_gpu_graph_b{b}"] = round(s, 0)
        print(f"GPU cuda-graph, batch={b}: {s:,.0f} steps/s ({dt*1e3:.1f} ms "
              f"for {steps} steps)")

    # --- policy nets ------------------------------------------------------
    x = torch.randn(4096, 344, device=dev)
    mlp = make_mlp(344, dev)
    f = bench_net(mlp, x)
    out["mlp_344_256_256_16_fwd_per_s_b4096"] = round(f, 0)
    print(f"MLP forward (344->256->256->16), batch 4096: {f:,.0f} fwd/s")

    for b in (512, 2048):
        xc = torch.randn(b, 4, 84, 84, device=dev)
        cnn = make_cnn(dev)
        fc = bench_net(cnn, xc)
        out[f"cnn_84x84x4_fwd_per_s_b{b}"] = round(fc, 0)
        print(f"small CNN (4x84x84 -> 16), batch {b}: {fc:,.0f} fwd/s")

    # --- combined: physics + MLP inside one graph (end-to-end decisions/s)
    batch = 8192
    st = _make_torch(dev, batch, CAP, sched)
    mlp = make_mlp(344, dev)
    obs = torch.randn(batch, 344, device=dev)

    def full_step(step):
        step_torch(st, step, batch, CAP)
        return mlp(obs)

    for _ in range(5):
        full_step(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for k in range(20):
            full_step(k)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for k in range(200):
            full_step(k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out["end_to_end_physics_plus_mlp_b8192"] = round(batch * 200 / dt, 0)
    print(f"END-TO-END physics+MLP, batch {batch}: {batch*200/dt:,.0f} decisions/s")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

