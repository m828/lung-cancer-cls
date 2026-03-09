from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from lung_cancer_cls.dataset import LungCT2DDataset, Sample, discover_iqothnccd_samples
from lung_cancer_cls.model import SimpleCTClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(samples: Sequence[Sample], train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        by_label[s.label].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for _, idxs in by_label.items():
        rng.shuffle(idxs)
        split = max(1, int(len(idxs) * train_ratio))
        train_idx.extend(idxs[:split])
        val_idx.extend(idxs[split:])
    return train_idx, val_idx


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    samples = discover_iqothnccd_samples(args.data_root)
    train_idx, val_idx = stratified_split(samples, args.train_ratio, args.seed)

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_ds = Subset(LungCT2DDataset(samples, transform=train_tf), train_idx)
    val_ds = Subset(LungCT2DDataset(samples, transform=val_tf), val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SimpleCTClassifier(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = running_loss / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_acc, "history": history}, f, ensure_ascii=False, indent=2)
    print(f"Done. best_val_acc={best_acc:.4f}, artifacts saved to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CT-only training on IQ-OTH/NCCD")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="outputs/ct_baseline")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    train_main(build_parser().parse_args())
