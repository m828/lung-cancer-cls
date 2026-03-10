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
from lung_cancer_cls.model import SimpleCTClassifier, ResNet18CTClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(samples: Sequence[Sample], train_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    """Split samples into train/val/test with exactly the same method as project366.ipynb.
    First split 80-20 into train-temp, then split temp 50-50 into val-test (final: 80-10-10).
    """
    from sklearn.model_selection import train_test_split
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, s in enumerate(samples):
        by_label[s.label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in by_label.items():
        # First split: train vs temp (80-20)
        train_imgs, temp_imgs = train_test_split(idxs, test_size=(1-train_ratio), random_state=seed, shuffle=True)
        # Second split: temp into val and test (50-50)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=seed, shuffle=True)

        train_idx.extend(train_imgs)
        val_idx.extend(val_imgs)
        test_idx.extend(test_imgs)
    return train_idx, val_idx, test_idx


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
    train_idx, val_idx, test_idx = stratified_split(samples, args.train_ratio, args.seed)

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    val_test_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_ds = Subset(LungCT2DDataset(samples, transform=train_tf), train_idx)
    val_ds = Subset(LungCT2DDataset(samples, transform=val_test_tf), val_idx)
    test_ds = Subset(LungCT2DDataset(samples, transform=val_test_tf), test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Dataset split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    if args.model == "resnet18":
        model = ResNet18CTClassifier(num_classes=3, pretrained=args.pretrained).to(device)
    else:
        model = SimpleCTClassifier(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    # Load best model and evaluate on test set
    print(f"\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "history": history},
        f, ensure_ascii=False, indent=2)
    print(f"Done. best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f} artifacts saved to: {out_dir}")


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
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio (default: 0.8), remaining 20% split equally into val and test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model", type=str, choices=["simple", "resnet18"], default="simple", help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    return parser


if __name__ == "__main__":
    train_main(build_parser().parse_args())
