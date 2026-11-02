"""Learner throughput on the local RTX 4070 SUPER: samples/s for fwd+bwd+Adam."""
import time
import torch

dev = torch.device("cuda")
print(torch.cuda.get_device_name(0))


def mlp(n_in=344):
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 16),
    ).to(dev)


def cnn():
    return torch.nn.Sequential(
        torch.nn.Conv2d(4, 32, 8, 4), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 4, 2), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, 2), torch.nn.ReLU(),
        torch.nn.Flatten(), torch.nn.Linear(64 * 4 * 4, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 16),
    ).to(dev)


def bench(net, x, iters=100, lr=3e-4):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    y = torch.randn(x.shape[0], 16, device=dev)
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return x.shape[0] * iters / dt


for b in (2048, 8192, 32768):
    x = torch.randn(b, 344, device=dev)
    s = bench(mlp(), x)
    print(f"MLP 344->256->256->16  train samples/s, batch {b}: {s:,.0f}")

for b in (256, 1024):
    x = torch.randn(b, 4, 84, 84, device=dev)
    s = bench(cnn(), x, iters=50)
    print(f"CNN 4x84x84->16        train samples/s, batch {b}: {s:,.0f}")
