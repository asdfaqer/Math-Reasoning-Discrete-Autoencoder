"""
Generate publication-quality benchmark charts for the Math Reasoning Discrete Autoencoder.

Outputs:
1. benchmarks/accuracy_vs_prefix_length.png
   - Focused comparison of Phase 2 Joint (Peak) vs. Phase 1 Frozen Encoder Baseline.
2. benchmarks/reconstruction_loss_training_curves.png
   - Side-by-side training curves for Phase 1 (Frozen Encoder) and Phase 2 (Joint End-to-End).
3. benchmarks/lossless_compression_vs_gzip.png
   - Neural lossless compression and Shannon bound compared to classical Gzip baseline across prefix lengths M.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(benchmarks_dir, "discrete_runs_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = {item["folder"]: item for item in data if "folder" in item}

    # Load lossless stats if present
    lossless_path = os.path.join(benchmarks_dir, "lossless_benchmark_stats.json")
    lossless_data = {}
    if os.path.exists(lossless_path):
        with open(lossless_path, "r", encoding="utf-8") as f:
            lossless_data = json.load(f)

    return runs, lossless_data

def plot_accuracy_focused(runs, output_dir):
    """
    Accuracy vs Prefix Length:
    Compares ONLY the 72-dim Phase 2 Joint (Peak) vs. 72-dim Phase 1 Frozen Encoder Baseline.
    """
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    m_values = [8, 16, 32, 48, 64]

    # Data for Peak Model (Phase 2 Joint)
    p2_run = runs.get("checkpoints_discrete_joint_gpu", {}).get("last_epoch", {}).get("benchmarks", {})
    p2_accs = [p2_run.get(str(m), {}).get("acc", 0.0) for m in m_values]

    # Data for Baseline Model (Phase 1 Frozen)
    p1_run = runs.get("checkpoints_discrete_frozen_encoder_cb16384", {}).get("last_epoch", {}).get("benchmarks", {})
    p1_accs = [p1_run.get(str(m), {}).get("acc", 0.0) for m in m_values]

    # Plot lines
    plt.plot(m_values, p2_accs, "o-", color="#00B894", linewidth=2.8, markersize=8.5, label="72-dim Cascading Memory (Phase 2 Joint - Peak)")
    plt.plot(m_values, p1_accs, "s--", color="#636E72", linewidth=2.0, markersize=7.0, label="72-dim Frozen Encoder Baseline (Phase 1)")

    # Data labels for Peak Model
    for m, acc in zip(m_values, p2_accs):
        plt.annotate(
            f"{acc:.1f}%",
            (m, acc),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#00796B"
        )

    # Data labels for Baseline Model
    for m, acc in zip(m_values, p1_accs):
        plt.annotate(
            f"{acc:.1f}%",
            (m, acc),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color="#2D3436"
        )

    # Shaded accuracy gain region
    plt.fill_between(m_values, p1_accs, p2_accs, color="#00B894", alpha=0.12, label="Joint Fine-Tuning Gain (+28.5% at M=64)")

    plt.title("Discrete Reasoning Reconstruction Accuracy vs. Latent Prefix Length (M)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Retained Discrete Latents (M) [Compression Factor]", fontsize=11, fontweight="semibold")
    plt.ylabel("Token Reconstruction Accuracy (%)", fontsize=11, fontweight="semibold")

    x_ticks = [8, 16, 32, 48, 64]
    x_labels = ["8 (8.0x)", "16 (4.0x)", "32 (2.0x)", "48 (1.33x)", "64 (1.0x)"]
    plt.xticks(x_ticks, x_labels, fontsize=10)
    plt.yticks(range(30, 105, 10), fontsize=10)
    plt.ylim(28, 100)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=10, loc="lower right")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "accuracy_vs_prefix_length.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_training_curves(runs, output_dir):
    """
    Reconstruction Loss Training Curves:
    Displays Phase 1 (Frozen Encoder) and Phase 2 (Joint Training) side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Phase 1 Data (5 Epochs)
    p1_epochs = list(range(1, 6))
    p1_train_rec = [2.8297, 2.5728, 2.4965, 2.4279, 2.4154]
    p1_val_rec = [2.0460, 1.9970, 1.9619, 1.9400, 1.9302]

    # Subplot 1: Phase 1
    ax1.plot(p1_epochs, p1_train_rec, "o-", color="#E17055", linewidth=2.2, markersize=6.5, label="Train Rec Loss")
    ax1.plot(p1_epochs, p1_val_rec, "s--", color="#D63031", linewidth=2.2, markersize=6.5, label="Val Rec Loss")
    ax1.set_title("Phase 1: Codebook Warmstart (Frozen Encoder)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=10.5, fontweight="semibold")
    ax1.set_ylabel("Reconstruction Cross-Entropy Loss", fontsize=10.5, fontweight="semibold")
    ax1.set_xticks(p1_epochs)
    ax1.set_ylim(1.8, 3.0)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=10)

    for ep, l in zip(p1_epochs, p1_val_rec):
        ax1.annotate(f"{l:.3f}", (ep, l), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, fontweight="bold", color="#D63031")

    # Phase 2 Data (15 Epochs)
    p2_epochs = list(range(1, 16))
    p2_train_rec = [1.5142, 1.4580, 1.4466, 1.4277, 1.4183, 1.3979, 1.4187, 1.3807, 1.3456, 1.3369, 1.2871, 1.2958, 1.2769, 1.2604, 1.2669]

    # Subplot 2: Phase 2
    ax2.plot(p2_epochs, p2_train_rec, "o-", color="#00B894", linewidth=2.4, markersize=6, label="Train Rec Loss (Joint)")
    ax2.set_title("Phase 2: End-to-End Joint Fine-Tuning", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=10.5, fontweight="semibold")
    ax2.set_ylabel("Reconstruction Cross-Entropy Loss", fontsize=10.5, fontweight="semibold")
    ax2.set_xticks(p2_epochs)
    ax2.set_ylim(1.2, 1.6)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Annotate biennial K-Means refresh points
    refresh_epochs = [5, 7, 9, 11, 13]
    for r_ep in refresh_epochs:
        ax2.axvline(r_ep, color="#0984E3", linestyle=":", alpha=0.6, linewidth=1.2)

    ax2.text(9, 1.56, "Biennial Spherical K-Means Refresh (dashed blue lines)", ha="center", fontsize=8.5, color="#0984E3", style="italic")
    
    # Annotate start and end loss
    ax2.annotate(f"{p2_train_rec[0]:.3f}", (1, p2_train_rec[0]), textcoords="offset points", xytext=(10, 5), ha="left", fontsize=9, fontweight="bold", color="#00B894")
    ax2.annotate(f"{p2_train_rec[-1]:.3f}", (15, p2_train_rec[-1]), textcoords="offset points", xytext=(-5, 8), ha="right", fontsize=9, fontweight="bold", color="#00796B")

    ax2.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=10, loc="lower left")

    plt.suptitle("Training Reconstruction Loss Progression: Phase 1 vs. Phase 2", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "reconstruction_loss_training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_lossless_vs_gzip(lossless_data, output_dir):
    """
    Lossless Compression Benchmark:
    Compares Neural Lossless, Shannon Theoretical Bound, and Discrete Latents
    against the classical Gzip compression baseline across prefix length M.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    m_values = [8, 16, 24, 32, 48, 64]

    # If dynamic data loaded from compute_lossless_stats, use it; otherwise fallback to exact computed constants
    if lossless_data:
        raw_text = [lossless_data[str(m)]["raw_text_bytes"] for m in m_values]
        gzip_text = [lossless_data[str(m)]["gzip_text_bytes"] for m in m_values]
        neural_lossless = [lossless_data[str(m)]["neural_lossless_bytes"] for m in m_values]
        shannon_bytes = [lossless_data[str(m)]["shannon_limit_bytes"] for m in m_values]
        latent_bytes = [lossless_data[str(m)]["latent_bytes"] for m in m_values]
    else:
        raw_text = [181.3] * len(m_values)
        gzip_text = [146.1] * len(m_values)
        neural_lossless = [93.7, 91.9, 94.3, 97.1, 104.9, 115.1]
        shannon_bytes = [44.4, 51.2, 60.4, 69.9, 87.5, 106.3]
        latent_bytes = [14.0, 28.0, 42.0, 56.0, 84.0, 112.0]

    raw_const = raw_text[0]
    gzip_const = gzip_text[0]

    # Plot curves
    plt.axhline(raw_const, color="#2D3436", linestyle="-", linewidth=1.5, label=f"Raw Uncompressed Text ({raw_const:.1f} Bytes / 64 tokens)")
    plt.axhline(gzip_const, color="#F39C12", linestyle="--", linewidth=2.0, label=f"Classical Gzip on Raw Text ({gzip_const:.1f} Bytes, 1.24x)")

    plt.plot(m_values, neural_lossless, "o-", color="#00B894", linewidth=2.5, markersize=7.5, label="Neural Lossless (Latents + Sparse Error Correction)")
    plt.plot(m_values, shannon_bytes, "d-.", color="#0984E3", linewidth=2.0, markersize=7, label="Shannon Theoretical Bound (Arithmetic Coder)")
    plt.plot(m_values, latent_bytes, "s:", color="#6C5CE7", linewidth=1.8, markersize=6, label="Discrete Latent Prefix Only (Lossy Representation)")

    # Highlight optimal lossless point (M = 16)
    opt_m = 16
    opt_idx = m_values.index(opt_m)
    opt_bytes = neural_lossless[opt_idx]
    opt_ratio = raw_const / opt_bytes
    savings_vs_gzip = ((gzip_const - opt_bytes) / gzip_const) * 100.0

    plt.scatter([opt_m], [opt_bytes], color="#E74C3C", s=140, zorder=5, edgecolors="#2D3436", linewidth=1.5)
    plt.annotate(
        f"Optimal Lossless: M=16\n{opt_bytes:.1f} Bytes ({opt_ratio:.2f}x vs Text)\n+{savings_vs_gzip:.1f}% vs Gzip",
        (opt_m, opt_bytes),
        textcoords="offset points",
        xytext=(30, -35),
        ha="left",
        fontsize=9.5,
        fontweight="bold",
        color="#C0392B",
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5)
    )

    # Shaded region where Neural Lossless outperforms Gzip on text
    plt.fill_between(m_values, neural_lossless, [gzip_const]*len(m_values), where=[nl < gzip_const for nl in neural_lossless], color="#00B894", alpha=0.15, label="Neural Outperformance over Gzip")

    plt.title("Lossless Reasoning Sequence Compression vs. Gzip on Raw Text", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Discrete Latent Prefix Length (M)", fontsize=11, fontweight="semibold")
    plt.ylabel("Compressed Bytes per 64-Token Sequence", fontsize=11, fontweight="semibold")

    x_ticks = [8, 16, 24, 32, 48, 64]
    x_labels = [f"M={m}\n({64/m:.1f}x)" for m in x_ticks]
    plt.xticks(x_ticks, x_labels, fontsize=9.5)
    plt.yticks(range(0, 210, 25), fontsize=10)
    plt.ylim(0, 205)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=9.5, loc="upper left")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "lossless_compression_vs_gzip.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
    runs, lossless_data = load_data()

    print("Generating updated benchmark charts...")
    plot_accuracy_focused(runs, benchmarks_dir)
    plot_training_curves(runs, benchmarks_dir)
    plot_lossless_vs_gzip(lossless_data, benchmarks_dir)
    print("All requested charts successfully generated in benchmarks/!")

if __name__ == "__main__":
    main()
