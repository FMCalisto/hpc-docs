"""ResNet-18 on CIFAR-10 with PyTorch DistributedDataParallel: one process per GPU, any number of nodes.

Started by torchrun (see job.sbatch), which sets RANK, LOCAL_RANK and WORLD_SIZE.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


def cifar_resnet18():
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # 32×32 images: no aggressive downsampling
    m.maxpool = nn.Identity()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256, help="per GPU")
    p.add_argument("--data", type=Path, required=True, help="dataset folder on this node's /scratch")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    main_proc = rank == 0
    print(f"rank {rank}/{world} on {os.uname().nodename}, GPU {torch.cuda.get_device_name(dev)}", flush=True)

    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    # The data was downloaded to each node's /scratch by job.sbatch before training.
    train = torchvision.datasets.CIFAR10(a.data, train=True, download=False, transform=train_tf)
    test = torchvision.datasets.CIFAR10(a.data, train=False, download=False, transform=test_tf)
    sampler = DistributedSampler(train, shuffle=True)  # each rank sees a different 1/world of the data
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1
    train_dl = DataLoader(train, batch_size=a.batch_size, sampler=sampler, num_workers=workers,
                          pin_memory=True, persistent_workers=True, drop_last=True)
    test_dl = DataLoader(test, batch_size=1000, num_workers=2)

    model = DDP(cifar_resnet18().to(dev, memory_format=torch.channels_last), device_ids=[local_rank])
    # Scale the learning rate with the global batch size.
    lr = 0.1 * a.batch_size * world / 256
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=a.epochs, steps_per_epoch=len(train_dl))

    epoch_times = []
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        for x, y in train_dl:
            x = x.to(dev, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # DDP averages the gradients across all GPUs here (NCCL all-reduce)
            opt.step()
            sched.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        epoch_times.append(dt)
        if main_proc:
            imgs = len(train_dl) * a.batch_size * world
            print(f"epoch {epoch + 1}/{a.epochs}: loss {loss.item():.3f}  {dt:.1f}s  {imgs / dt:,.0f} images/s",
                  flush=True)

    if main_proc:
        model.eval()
        correct = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in test_dl:
                correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        steady = epoch_times[1:] or epoch_times  # skip the first epoch (warm-up)
        result = {
            "gpus": world,
            "nodes": int(os.environ.get("SLURM_NNODES", 1)),
            "global_batch": a.batch_size * world,
            "test_accuracy": correct / len(test),
            "seconds_per_epoch": sum(steady) / len(steady),
            "images_per_second": len(train_dl) * a.batch_size * world * len(steady) / sum(steady),
        }
        print(json.dumps(result), flush=True)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "result.json").write_text(json.dumps(result, indent=2))
        torch.save(model.module.state_dict(), a.out / "resnet18-cifar10.pt")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
