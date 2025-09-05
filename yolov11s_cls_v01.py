
import sys
sys.path = [p for p in sys.path if not p.startswith('/opt/ros/')]
# ------------------------------------------------------------------

import os, json, shutil, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from ultralytics import YOLO


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# Logging
# =========================
def get_logger(log_file: Path):
    def log(msg: str, echo: bool = True):
        if echo:
            print(msg)
        with open(log_file, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
    return log


# =========================
# Data
# =========================
class GaussianNoiseTransform:
    """Add zero-mean Gaussian noise (σ in [0,1] for [0,1] tensors), then clamp."""
    def __init__(self, sigma: float = 0.0):
        self.sigma = float(sigma)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma <= 0:
            return x
        noise = torch.randn_like(x) * self.sigma
        return torch.clamp(x + noise, 0.0, 1.0)


def make_loader(root: Path, split: str, imgsz: int, batch: int, workers: int,
                noise_sigma: float = 0.0):
    tfm = transforms.Compose([
        transforms.Resize((imgsz, imgsz), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),                 # -> [0,1]
        GaussianNoiseTransform(noise_sigma),   # noise di [0,1], lalu clamp
        
    ])
    ds = datasets.ImageFolder(str(root / split), transform=tfm)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return loader, ds.classes



# =========================
# Inference helpers
# =========================
@torch.no_grad()
def infer_logits(weights_path: Path, loader: DataLoader, device: torch.device):
    """
    Fast, quiet classifier inference using the raw torch model.
    Returns: (y_true: np.ndarray, y_pred: np.ndarray)
    """
    mdl = YOLO(str(weights_path))
    net = mdl.model.to(device).eval()

    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)

        # Some Ultralytics builds return (logits, ...) instead of a Tensor
        out = net(imgs)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.ndim == 1:  # safety for batch=1
            logits = logits.unsqueeze(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.cpu().numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    return acc * 100.0


# =========================
# Plots
# =========================
def plot_composite(df: pd.DataFrame,
                   cm_clean: np.ndarray,
                   class_names: list[str],
                   per_class_clean: list[float],
                   test_acc_clean: float,
                   out_path: Path,
                   title: str,
                   per_class_ref: tuple[np.ndarray, float] | None = None,
                   cm_img_path: Path | None = None):
    """TL: loss, TR: val acc, BL: Ultralytics CM (jika ada), BR: per-class bars."""
    fig = plt.figure(figsize=(18, 10))

    # TL — Loss
    ax1 = plt.subplot(2, 2, 1)
    if 'train/loss' in df.columns:
        ax1.plot(df['train/loss'], label='Train Loss', linewidth=2)
    if 'val/loss' in df.columns:
        ax1.plot(df['val/loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3); ax1.legend()

    # TR — Val Acc
    ax2 = plt.subplot(2, 2, 2)
    if 'metrics/accuracy_top1' in df.columns:
        ax2.plot(df['metrics/accuracy_top1'] * 100.0, label='Val Top-1', linewidth=2)
    if 'metrics/accuracy_top5' in df.columns:
        ax2.plot(df['metrics/accuracy_top5'] * 100.0, label='Val Top-5', linewidth=2)
    ax2.set_title('Validation Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3); ax2.legend()

    # BL — Confusion Matrix 
    ax3 = plt.subplot(2, 2, 3)
    if cm_img_path is not None and cm_img_path.exists():
        img = plt.imread(cm_img_path)
        ax3.imshow(img)
        ax3.axis('off')
        ax3.set_title(f'Confusion Matrix (Ultralytics Test, Acc={test_acc_clean:.2f}%)')
    else:
        im = ax3.imshow(cm_clean, cmap='Blues')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title(f'Confusion Matrix (Test Clean, Acc={test_acc_clean:.2f}%)')
        ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
        ax3.set_xticks(range(len(class_names))); ax3.set_yticks(range(len(class_names)))
        ax3.set_xticklabels(class_names, rotation=45, ha='right'); ax3.set_yticklabels(class_names)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax3.text(j, i, int(cm_clean[i, j]), ha='center', va='center', fontsize=9)

    # BR — Per-class bars
    ax4 = plt.subplot(2, 2, 4)
    x = np.arange(len(class_names))
    w = 0.38
    b1 = ax4.bar(x - w/2, per_class_clean, width=w, label='Clean')
    if per_class_ref is not None:
        per_class_noisy, ref_sigma = per_class_ref
        b2 = ax4.bar(x + w/2, per_class_noisy, width=w, label=f'Noisy σ={ref_sigma}')
    ax4.set_title('Per-Class Test Accuracy')
    ax4.set_ylabel('Accuracy (%)'); ax4.set_ylim(0, 105)
    ax4.set_xticks(x); ax4.set_xticklabels(class_names, rotation=45, ha='right')
    for b, acc in zip(b1, per_class_clean):
        ax4.text(b.get_x()+b.get_width()/2, min(100, acc)+1, f"{acc:.1f}%",
                 ha='center', va='bottom', fontsize=9)
    if per_class_ref is not None:
        for b, acc in zip(b2, per_class_noisy):
            ax4.text(b.get_x()+b.get_width()/2, min(100, acc)+1, f"{acc:.1f}%",
                     ha='center', va='bottom', fontsize=9)
        ax4.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_noise_curve(sigmas: list[float], accs: list[float], out_path: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(sigmas, accs, marker='o', linewidth=2)
    plt.xticks(sigmas, [f"{s:.2f}" for s in sigmas])  # tampilkan semua σ
    plt.xlabel('Gaussian σ')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Robustness Curve: Accuracy vs Gaussian Noise')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# =========================
# Single experiment routine
# =========================
def run_single_experiment(exp_name: str,
                          base_data_dir: Path,
                          out_root: Path,
                          noise_list: list[float],
                          imgsz: int = 320,
                          batch: int = 32,
                          epochs: int = 100,
                          device_index: int = 0,
                          workers: int = 8,
                          seed: int = 42,
                          config_updates: dict | None = None) -> dict:
    """Train, log, evaluate (clean + noise sweep), and return summary."""
    timestamp = now_tag()
    EXP_DIR = out_root / f"yolo_{exp_name}_{timestamp}"
    (EXP_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "models").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "results").mkdir(parents=True, exist_ok=True)

    log = get_logger(EXP_DIR / "logs" / "training_log.txt")
    DEVICE = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    log("="*70)
    log(f"Experiment: {exp_name}")
    log(f"Dataset:   {base_data_dir}")
    log(f"Artifacts: {EXP_DIR}")
    log("="*70)

    # Base YOLO train config
    train_cfg = dict(
        data=str(base_data_dir),
        epochs=epochs, imgsz=imgsz, batch=batch,
        lr0=0.0005, lrf=0.00001, weight_decay=0.001,
        degrees=2, translate=0.02, scale=0.05, shear=0,
        flipud=0.0, fliplr=0.2, mosaic=0.0, mixup=0.0,
        hsv_h=0.0, hsv_s=0.1, hsv_v=0.6,
        dropout=0.3,
        optimizer="SGD", momentum=0.937, cos_lr=False,
        warmup_epochs=5, warmup_bias_lr=0.0001, warmup_momentum=0.5,
        val=True, save=True, save_period=10,
        device=device_index, workers=workers,
        amp=True, cache=True, plots=True, seed=seed,
        deterministic=True, single_cls=False, verbose=True,
        project=str(EXP_DIR), name="train"
    )
    if config_updates:
        train_cfg.update(config_updates)

    log("\nConfiguration updates:")
    for k, v in (config_updates or {}).items():
        log(f"  {k}: {v}")

    # Train
    model = YOLO("yolo11s-cls.pt", task="classify")
    _ = model.train(**train_cfg)

    # Read results.csv
    results_csv = EXP_DIR / "train" / "results.csv"
    if not results_csv.exists():
        log("ERROR: results.csv not found.")
        return {"experiment": exp_name, "error": "results.csv missing"}

    df = pd.read_csv(results_csv)
    assert 'metrics/accuracy_top1' in df.columns and 'val/loss' in df.columns, "results.csv missing required columns."
    best_idx = int(df['metrics/accuracy_top1'].idxmax())
    best_row = df.iloc[best_idx]
    final_row = df.iloc[-1]

    best_top1 = float(best_row['metrics/accuracy_top1'] * 100.0)
    best_top5 = float(best_row['metrics/accuracy_top5'] * 100.0) if 'metrics/accuracy_top5' in df.columns else float('nan')
    final_top1 = float(final_row['metrics/accuracy_top1'] * 100.0)
    final_top5 = float(final_row['metrics/accuracy_top5'] * 100.0) if 'metrics/accuracy_top5' in df.columns else float('nan')

    # CNN-style epoch logs
    log("\nEpoch logs (reconstructed from results.csv):")
    training_history = []
    n_epochs = len(df)
    for i, row in df.iterrows():
        tr_loss = row['train/loss'] if 'train/loss' in df.columns else float('nan')
        va_loss = row['val/loss'] if 'val/loss' in df.columns else float('nan')
        va_top1 = row['metrics/accuracy_top1'] if 'metrics/accuracy_top1' in df.columns else float('nan')
        lr = row['lr/pg0'] if 'lr/pg0' in df.columns else (row['train/lr'] if 'train/lr' in df.columns else float('nan'))

        log(f"Epoch [{i+1:3d}/{n_epochs:3d}] "
            f"Train Loss: {tr_loss:.4f}, Val Loss: {va_loss:.4f}, "
            f"Val Acc: {va_top1*100.0 if np.isfinite(va_top1) else float('nan'):.2f}% | LR: {lr:.6g}")

        training_history.append({
            "epoch": int(i + 1),
            "train_loss": float(tr_loss) if np.isfinite(tr_loss) else None,
            "val_loss": float(va_loss) if np.isfinite(va_loss) else None,
            "val_acc": float(va_top1 * 100.0) if np.isfinite(va_top1) else None,
            "lr": float(lr) if np.isfinite(lr) else None
        })

    # Copy best.pt; copy VAL CM; TEST eval via Ultralytics and grab CM image + top-1
    best_pt_src = EXP_DIR / "train" / "weights" / "best.pt"
    MODELS_DIR = EXP_DIR / "models"
    PLOTS_DIR = EXP_DIR / "plots"
    RESULTS_DIR = EXP_DIR / "results"

    if best_pt_src.exists():
        shutil.copy2(best_pt_src, MODELS_DIR / "best.pt")
        log(f"Copied best.pt → {MODELS_DIR/'best.pt'}")

        # copy VAL CM (if produced during training)
        cm_val_src = EXP_DIR / "train" / "confusion_matrix.png"
        if cm_val_src.exists():
            shutil.copy2(cm_val_src, PLOTS_DIR / "confusion_matrix_val.png")

        # Ultralytics TEST eval (this writes test_eval/confusion_matrix.png)
        metrics = YOLO(str(best_pt_src)).val(
            data=str(base_data_dir),
            imgsz=imgsz,
            batch=batch,
            split='test',
            plots=True,
            project=str(EXP_DIR),
            name='test_eval'
        )

        cm_test_src = EXP_DIR / "test_eval" / "confusion_matrix.png"
        if cm_test_src.exists():
            shutil.copy2(cm_test_src, PLOTS_DIR / "confusion_matrix_test.png")
        else:
            log("WARNING: Ultralytics test confusion_matrix.png not found.")

        # Ultralytics Top-1 for composite title
        ultra_test_top1 = float(metrics.results_dict.get('metrics/accuracy_top1', 0.0) * 100.0)

    else:
        log("ERROR: best.pt not found; cannot evaluate on test.")
        return {"experiment": exp_name, "error": "best.pt missing"}



    # Our own TEST eval (clean + noise sweep 0.00..0.25)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    loader_clean, class_names = make_loader(base_data_dir, "test", imgsz, batch, workers, noise_sigma=0.0)
    y_true_clean, y_pred_clean = infer_logits(MODELS_DIR / "best.pt", loader_clean, device)

    cm_clean = confusion_matrix(y_true_clean, y_pred_clean, labels=list(range(len(class_names))))
    per_class_clean = per_class_accuracy(cm_clean)
    overall_clean = float((y_true_clean == y_pred_clean).mean() * 100.0)

    # Noise sweep (default includes 0.25)
    noise_results = {}
    overall_vs_sigma = []
    ref_sigma = 0.10 if 0.10 in noise_list else (noise_list[len(noise_list)//2] if noise_list else None)
    per_class_ref = None

    log("\nGaussian-noise robustness sweep:")
    for sigma in noise_list:
        loader_noisy, _ = make_loader(base_data_dir, "test", imgsz, batch, workers, noise_sigma=sigma)
        y_t, y_p = infer_logits(MODELS_DIR / "best.pt", loader_noisy, device)
        cm = confusion_matrix(y_t, y_p, labels=list(range(len(class_names))))
        per_class = per_class_accuracy(cm)
        overall = float((y_t == y_p).mean() * 100.0)
        noise_results[str(sigma)] = {
            "overall_accuracy": overall,
            "per_class_accuracy": {n: float(a) for n, a in zip(class_names, per_class)}
        }
        overall_vs_sigma.append(overall)
        log(f"  σ={sigma:.2f} → Overall: {overall:.2f}%")
        if ref_sigma is not None and abs(sigma - ref_sigma) < 1e-9:
            per_class_ref = (per_class, sigma)

    # Composite plot + noise curve
    composite_png = PLOTS_DIR / "composite_results.png"
    plot_composite(
        df=df,
        cm_clean=cm_clean,
        class_names=class_names,
        per_class_clean=per_class_clean.tolist(),
        test_acc_clean=ultra_test_top1 if 'ultra_test_top1' in locals() else overall_clean,
        out_path=composite_png,
        title=f"YOLOv11s-cls — {exp_name} — {timestamp}",
        per_class_ref=per_class_ref,
        cm_img_path=cm_test_src if 'cm_test_src' in locals() and cm_test_src.exists() else None
    )


    noise_curve_png = PLOTS_DIR / "noise_accuracy_curve.png"
    plot_noise_curve(noise_list, overall_vs_sigma, noise_curve_png)

    # JSON save
    cls_report = classification_report(y_true_clean, y_pred_clean, target_names=class_names, output_dict=True)
    complete = {
        "model": "YOLOv11s-cls",
        "experiment": exp_name,
        "timestamp": timestamp,
        "dataset": {"root": str(base_data_dir), "num_classes": len(class_names), "class_names": class_names},
        "training_curves": {
            "best_epoch_index": int(best_idx),
            "best_val_top1": best_top1,
            "best_val_top5": best_top5,
            "final_val_top1": final_top1,
            "final_val_top5": final_top5,
        },
        "training_history": training_history,
        "test_results_clean": {
            "overall_accuracy": overall_clean,
            "per_class_accuracy": {n: float(a) for n, a in zip(class_names, per_class_clean)},
            "confusion_matrix": cm_clean.tolist(),
            "classification_report": cls_report
        },
        "noise_results": noise_results,
        "artifacts": {
            "best_weights": str(MODELS_DIR / "best.pt"),
            "composite_plot": str(composite_png),
            "noise_curve_plot": str(noise_curve_png),
            "cm_val_png": str(PLOTS_DIR / "confusion_matrix_val.png"),
            "cm_test_png": str(PLOTS_DIR / "confusion_matrix_test.png")
        }
    }
    with open(RESULTS_DIR / "complete_results.json", "w") as f:
        json.dump(complete, f, indent=2)

    # Console footer (CNN-style)
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✔ Model dir:   {MODELS_DIR}")
    print(f"✔ Plots dir:   {PLOTS_DIR}")
    print(f"✔ Logs dir:    {EXP_DIR/'logs'}")
    print(f"✔ Results dir: {RESULTS_DIR}\n")
    print("Key Results:")
    print(f"- Best Val Accuracy (Top-1): {best_top1:.2f}%")
    print(f"- Test Accuracy (clean):     {overall_clean:.2f}%")
    if 'ultra_test_top1' in locals():
        print(f"- Ultralytics Test Top-1:     {ultra_test_top1:.2f}%")

    print("Per-Class Test Accuracy (clean):")
    for name, acc in zip(class_names, per_class_clean):
        status = "✓" if acc >= 85 else "✗"
        print(f"  {status} {name}: {acc:.2f}%")
    print("\nGaussian-noise sweep:")
    for s, acc in zip(noise_list, overall_vs_sigma):
        print(f"  σ={s:.2f}: {acc:.2f}%")
    print(f"\nSaved plot:   {composite_png}")
    print(f"Noise curve:  {noise_curve_png}")
    print(f"Saved JSON:   {RESULTS_DIR / 'complete_results.json'}")
    print(f"Saved log:    {EXP_DIR / 'logs' / 'training_log.txt'}")

    # Return brief summary for global table
    return {
        "experiment": exp_name,
        "best_val_top1": best_top1,
        "test_acc_clean": overall_clean
    }


# =========================
# Experiment grid (Y0–Y5)
# =========================
EXPERIMENTS = {
    'Y0_baseline': {},
    'Y1_large_image': {'imgsz': 384},
    'Y2_adamw': {'optimizer': 'AdamW', 'lr0': 0.001, 'lrf': 0.0001},
    'Y3_large_batch': {'batch': 64, 'lr0': 0.001, 'lrf': 0.00002},
    'Y4_small_batch': {'batch': 16, 'lr0': 0.00025, 'lrf': 0.000005},
    'Y5_cosine': {'cos_lr': True}
}


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser("YOLOv11s-cls — CNN-style logs + Gaussian-noise robustness (Y0–Y5)")
    parser.add_argument("--data", type=str, default="/data/dn564/sampledv11_cropped")
    parser.add_argument("--out", type=str, default="/data/dn564/experiments")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=str,
                        default="0,0.05,0.08,0.10,0.12,0.15,0.20,0.25",
                        help="Comma-separated σ list (on [0,1]) for Gaussian-noise test (0.00 .. 0.25 inclusive)")
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all"] + list(EXPERIMENTS.keys()),
                        help="Which experiment to run")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: set epochs=10 for all experiments")
    args = parser.parse_args()

    seed_everything(args.seed)

    noise_list = [float(s.strip()) for s in args.noise.split(",") if s.strip() != ""]
    base_data = Path(args.data)
    out_root = Path(args.out)

    # Optionally shorten epochs
    exps = {k: dict(v) for k, v in EXPERIMENTS.items()}
    if args.quick:
        for v in exps.values():
            v['epochs'] = 10

    summaries = {}

    if args.exp == "all":
        for name, cfg in exps.items():
            print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            # allow per-exp override while keeping CLI imgsz/batch/epochs if provided
            imgsz = cfg.get('imgsz', args.imgsz)
            batch = cfg.get('batch', args.batch)
            epochs = cfg.get('epochs', args.epochs)
            # remove these keys from updates so we don't pass duplicates
            cfg_updates = dict(cfg)
            for k in ['imgsz', 'batch', 'epochs']:
                cfg_updates.pop(k, None)
            summary = run_single_experiment(
                exp_name=name,
                base_data_dir=base_data,
                out_root=out_root,
                noise_list=noise_list,
                imgsz=imgsz, batch=batch, epochs=epochs,
                device_index=args.device, workers=args.workers, seed=args.seed,
                config_updates=cfg_updates
            )
            summaries[name] = summary
            print(f"Completed {name}: ValTop1={summary['best_val_top1']:.2f}% | TestClean={summary['test_acc_clean']:.2f}%")
    else:
        name = args.exp
        cfg = exps[name]
        imgsz = cfg.get('imgsz', args.imgsz)
        batch = cfg.get('batch', args.batch)
        epochs = cfg.get('epochs', args.epochs)
        cfg_updates = dict(cfg)
        for k in ['imgsz', 'batch', 'epochs']:
            cfg_updates.pop(k, None)
        summary = run_single_experiment(
            exp_name=name,
            base_data_dir=base_data,
            out_root=out_root,
            noise_list=noise_list,
            imgsz=imgsz, batch=batch, epochs=epochs,
            device_index=args.device, workers=args.workers, seed=args.seed,
            config_updates=cfg_updates
        )
        summaries[name] = summary
        print(f"Completed {name}: ValTop1={summary['best_val_top1']:.2f}% | TestClean={summary['test_acc_clean']:.2f}%")

    # Overall summary file
    overall_path = out_root / f"yolo_experiments_summary_{now_tag()}.json"
    with open(overall_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Experiment':<18} {'ValTop1':>8} {'TestClean':>11}")
    print("-"*40)
    for name, s in summaries.items():
        print(f"{name:<18} {s['best_val_top1']:>8.2f}% {s['test_acc_clean']:>11.2f}%")
    print(f"\nSummary saved to: {overall_path}")


if __name__ == "__main__":
    main()
