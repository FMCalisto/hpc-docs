"""Train a small CNN on MNIST, on CPU or GPU. Used by examples 3 and 4.

    python mnist.py --device cpu  --epochs 3
    python mnist.py --device cuda --epochs 3 --amp
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop(torch.flatten(x, 1))
        return self.fc2(self.drop(F.relu(self.fc1(x))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 2)) // 2)
    p.add_argument("--amp", action="store_true", help="mixed precision (GPU only)")
    p.add_argument("--data", type=Path, default=Path(f"/scratch/{os.environ['USER']}/datasets"))
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()

    dev = torch.device(a.device)
    print(f"device={dev} torch_threads={torch.get_num_threads()} workers={a.workers}")
    if dev.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Downloaded once per node into /scratch, then reused by later jobs on that node.
    train = datasets.MNIST(a.data, train=True, download=True, transform=tf)
    test = datasets.MNIST(a.data, train=False, download=True, transform=tf)
    kw = dict(num_workers=a.workers, pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    train_dl = DataLoader(train, batch_size=a.batch_size, shuffle=True, **kw)
    test_dl = DataLoader(test, batch_size=1000, **kw)

    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    use_amp = a.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        train_s = time.time() - t0

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x.to(dev)).argmax(1) == y.to(dev)).sum().item()
        print(f"epoch {epoch}: loss {loss.item():.4f}  test acc {100 * correct / len(test):.2f}%"
              f"  epoch time {train_s:.1f}s")

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"mnist-{dev.type}-{os.environ.get('SLURM_JOB_ID', 'local')}.pt"
    torch.save(model.state_dict(), path)
    print("Saved", path)


if __name__ == "__main__":
    main()
