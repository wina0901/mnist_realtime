from __future__ import annotations

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 
from model import SimpleCNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optim.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / max(len(loader.dataset), 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train MNIST baseline weights for app.py demo")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2) 
    p.add_argument("--data_dir", type=str, default="./data", help="MNIST download/cache directory") 
    p.add_argument("--out", type=str, default="./weights/baseline.pt", help="Output .pt path")

    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # MNIST dataset
    tfm = transforms.Compose([transforms.ToTensor()])  # (1,28,28), 0~1
    train_ds = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
 
    model = SimpleCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optim, device)
        acc = evaluate(model, test_loader, device)
        print(f"[EPOCH {epoch:02d}/{args.epochs}] loss={loss:.4f}  test_acc={acc*100:.2f}%")

    torch.save(model.state_dict(), args.out)
    print(f"[DONE] saved weights -> {args.out}")


if __name__ == "__main__":
    main()