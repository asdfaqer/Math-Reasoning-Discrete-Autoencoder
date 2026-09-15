import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import time
import json
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cascading_memory_model import SmallCascadingMemoryAutoencoder
from train_cascading_memory import load_dataset, evaluate, multi_prefix_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Discrete VQ Codebook with Frozen Pretrained Continuous Encoder")
    # Paths & Checkpoints
    parser.add_argument("--source_ckpt", type=str,
                        default="models/checkpoints_cascading_uniform_appended_m_causal/cascading_memory_epoch_23.pt",
                        help="Path to continuous pretrained checkpoint")
    parser.add_argument("--save_dir", type=str,
                        default="models/checkpoints_discrete_frozen_encoder_cb16384",
                        help="Save directory for discrete fine-tuned checkpoints")
    parser.add_argument("--data_cache", type=str, default="tokenized_fast_tensor_64.pt",
                        help="Tokenized dataset cache path")

    # Architecture / Prefix
    parser.add_argument("--prefix_distribution", type=str, default="uniform", choices=["fixed", "uniform"],
                        help="Prefix distribution during fine-tuning (default: uniform)")
    parser.add_argument("--min_prefix_len", type=int, default=8, help="Min prefix length (default: 8)")
    parser.add_argument("--max_prefix_len", type=int, default=64, help="Max prefix length (default: 64)")
    parser.add_argument("--fixed_prefix_len", type=int, default=32, help="Fixed prefix length if fixed (default: 32)")

    # Clustering / Codebook Warmstart
    parser.add_argument("--kmeans_batches", type=int, default=150,
                        help="Number of train batches to extract latents for K-Means (default: 150 -> ~150k latents)")
    parser.add_argument("--kmeans_iters", type=int, default=10,
                        help="Number of spherical K-Means iterations (default: 10)")
    parser.add_argument("--freeze_codebook", action="store_true", default=False,
                        help="Strictly freeze codebook centroids as well (only train decoder)")
    parser.add_argument("--skip_kmeans", action="store_true", default=False,
                        help="Skip K-Means clustering and use existing checkpoint codebook weights")

    # Optimization
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
    parser.add_argument("--vq_loss_weight", type=float, default=0.1, help="VQ commitment loss weight (default: 0.1)")
    parser.add_argument("--start_word_dropout", type=float, default=0.4, help="Start word dropout (default: 0.4)")
    parser.add_argument("--end_word_dropout", type=float, default=0.1, help="End word dropout (default: 0.1)")
    parser.add_argument("--max_val_steps", type=int, default=100, help="Max validation steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log step interval (default: 100)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def fit_spherical_kmeans(model, train_loader, device, K=16384, num_batches=150, iters=10):
    """
    Extracts continuous latent vectors from the frozen encoder and fits K spherical centroids.
    """
    print(f"\n[CLUSTERING] Extracting continuous latents from {num_batches} batches...")
    model.eval()
    latents = []
    with torch.no_grad():
        for step, batch in enumerate(train_loader):
            if step >= num_batches:
                break
            input_ids = batch[0].to(device)
            # Sample variable prefix length for diverse latent distribution
            p_len = torch.randint(8, 65, (1,)).item()
            _, _, _, h = model.encoder(input_ids, prefix_len=p_len, return_continuous=True)
            has_appended = getattr(model, "append_m_query", False) and getattr(model, "use_compression_embeddings", False)
            exp_len = p_len + 1 if has_appended else p_len
            latents.append(h[:, :exp_len, :].reshape(-1, model.d_model))

    X = torch.cat(latents, dim=0) # [N, d_model]
    N = X.size(0)
    print(f"[CLUSTERING] Collected {N:,} continuous latent vectors. L2-normalizing for Spherical K-Means...")
    X_norm = F.normalize(X, p=2, dim=-1)

    # Initialize centroids randomly from data samples
    K = min(K, N)
    rand_idx = torch.randperm(N, device=device)[:K]
    centroids = X_norm[rand_idx].clone() # [K, d_model]

    print(f"[CLUSTERING] Running {iters} iterations of fast spherical mini-batch K-Means (K={K:,})...")
    batch_sz = 16384
    for it in range(1, iters + 1):
        idx_chunk = torch.randint(0, N, (batch_sz,), device=device)
        chunk = X_norm[idx_chunk]
        sim = torch.matmul(chunk, centroids.t()) # [batch_sz, K]
        assign = sim.argmax(dim=-1)
        unique_k = torch.unique(assign)
        for k_idx in unique_k:
            pts = chunk[assign == k_idx]
            centroids[k_idx] = F.normalize(pts.mean(dim=0), p=2, dim=-1)
        if it % 2 == 0 or it == iters:
            print(f"  Iteration [{it:2d}/{iters}] completed.")

    # Measure quantization cosine alignment on a hold-out sample
    sample_eval = X_norm[:10000]
    sim_eval = torch.matmul(sample_eval, centroids.t()).max(dim=-1).values
    avg_cos = sim_eval.mean().item()
    avg_deg = torch.acos(torch.tensor(avg_cos)).item() * 180 / 3.14159
    print(f"[CLUSTERING] Centroids fitted! Mean alignment: cos={avg_cos:.4f} (angular distortion: {avg_deg:.2f} deg)\n")

    return centroids


def main():
    args = parse_args()
    os.makedirs("debug_outputs", exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 85)
    print("      STRATEGY A: DISCRETE VQ CODEBOOK FINE-TUNING (FROZEN ENCODER)")
    print("=" * 85)
    print(f"Source Checkpoint: {args.source_ckpt}")
    print(f"Save Directory:    {args.save_dir}")
    print(f"Device:            {device}")

    # 1. Load dataset
    train_loader, val_loader, train_size, val_size = load_dataset(args.data_cache, batch_size=args.batch_size)

    # 2. Instantiate Model and Load Checkpoint
    ckpt = torch.load(args.source_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    model = SmallCascadingMemoryAutoencoder(
        vocab_size=ckpt_args.get("vocab_size", 50265),
        d_model=ckpt_args.get("d_model", 72),
        nhead=ckpt_args.get("nhead", 4),
        text_encoder_layers=ckpt_args.get("text_encoder_layers", 3),
        num_layers=ckpt_args.get("num_layers", 3),
        decoder_num_layers=ckpt_args.get("decoder_num_layers", 3),
        codebook_size=ckpt_args.get("codebook_size", 16384),
        max_length=ckpt_args.get("max_length", 64),
        fixed_prefix_len=args.fixed_prefix_len,
        prefix_distribution=args.prefix_distribution,
        min_prefix_len=args.min_prefix_len,
        max_prefix_len=args.max_prefix_len,
        skip_bottleneck=False, # DISCRETE MODE
        norm_first=ckpt_args.get("norm_first", True),
        causal_cross_attn=ckpt_args.get("causal_cross_attn", True),
        causal_encoder_queries=ckpt_args.get("causal_encoder_queries", True),
        normalize_prefix_pos=ckpt_args.get("normalize_prefix_pos", False),
        prefix_pos_type=ckpt_args.get("prefix_pos_type", "interpolated"),
        use_compression_embeddings=ckpt_args.get("use_compression_embeddings", True),
        append_m_query=ckpt_args.get("append_m_query", True),
        adaptive_prefix_queries=ckpt_args.get("adaptive_encoder_queries", False),
        vq_loss_weight=args.vq_loss_weight,
        diversity_loss_weight=0.5,
        k_samples=1
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[MODEL] Loaded weights from {args.source_ckpt} (Epoch {ckpt.get('epoch', 'N/A')})")

    # 3. Optional Spherical K-Means Centroid Initialization
    if not args.skip_kmeans:
        centroids = fit_spherical_kmeans(
            model,
            train_loader,
            device,
            K=model.codebook_size,
            num_batches=args.kmeans_batches,
            iters=args.kmeans_iters
        )
        with torch.no_grad():
            model.encoder.vq.embedding.weight.copy_(centroids)
        print(f"[CODEBOOK] Successfully initialized {model.codebook_size} codebook entries with K-Means centroids!")

    # 4. Freeze Encoder to Prevent Latent Collapse
    print("\n[FREEZING] Configuring frozen and trainable parameter groups...")
    frozen_params = 0
    trainable_params = 0

    for name, param in model.encoder.named_parameters():
        if "vq.embedding" in name and not args.freeze_codebook:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()

    # Shared text embedding stays frozen with encoder
    for name, param in model.decoder.named_parameters():
        if "text_embedding" in name and model.share_embeddings:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()

    # Accuracy probe stays trainable
    for name, param in model.probe.named_parameters():
        param.requires_grad = True
        trainable_params += param.numel()

    print(f"  Frozen Parameters:    {frozen_params:,} (Text Encoder, Pos Tables, Cascading Stack)")
    print(f"  Trainable Parameters: {trainable_params:,} (Decoder, Codebook Vectors, Probe)")

    # 5. Baseline Evaluation Before Fine-Tuning
    print("\n[EVAL] Running baseline evaluation on continuous vs. initial discrete codebook...")
    model.eval()
    benchmarks_list = [64, 48, 32, 16, 8]
    base_eval = multi_prefix_benchmark(model, val_loader, device, benchmarks=benchmarks_list, max_eval_steps=20)
    disc_accs = [f"{m}: {base_eval[m]['acc']:.2f}%" for m in benchmarks_list]
    cont_accs = [f"{m}: {base_eval[m].get('cont_acc', 0.0):.2f}%" for m in benchmarks_list]
    print(f"  Initial Discrete Codebook:   {' | '.join(disc_accs)}")
    print(f"  Reference Continuous Latents: {' | '.join(cont_accs)}")

    # 6. Optimizer (Trainable parameters only)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # Save initial config
    model_config = {
        "model_class": model.__class__.__name__,
        "args": vars(args),
        "source_ckpt": args.source_ckpt,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params
    }
    with open(os.path.join(args.save_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # 7. Fine-Tuning Loop
    total_epochs = args.epochs
    steps_per_epoch = len(train_loader)
    print(f"\n[TRAIN] Beginning {total_epochs} epochs of discrete codebook training (skip_bottleneck=False)...")

    for epoch in range(1, total_epochs + 1):
        epoch_start_time = time.time()
        progress = (epoch - 1) / max(1, total_epochs - 1)
        word_drop = args.start_word_dropout - progress * (args.start_word_dropout - args.end_word_dropout)

        print(f"\n--- Epoch [{epoch}/{total_epochs}] Discrete Mode (word_dropout={word_drop:.2f}) ---")
        model.train()
        # Keep encoder in eval mode so LayerNorm / Dropout stats remain fixed
        model.encoder.eval()

        total_loss_accum = 0.0
        rec_loss_accum = 0.0
        vq_loss_accum = 0.0
        acc_accum = 0.0
        unique_codes = set()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None

            if args.prefix_distribution == "uniform":
                step_prefix = torch.randint(args.min_prefix_len, args.max_prefix_len + 1, (1,)).item()
            else:
                step_prefix = args.fixed_prefix_len

            # Discrete forward: skip_bottleneck=False
            out = model(
                input_ids,
                attention_mask=attention_mask,
                prefix_len=step_prefix,
                word_dropout=word_drop,
                skip_bottleneck=False,
                tau=0.0
            )

            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()

            total_loss_accum += loss.item()
            rec_loss_accum += out["rec_loss"].item()
            vq_loss_accum += out["vq_loss"].item()
            acc_accum += out["mean_true_acc"]

            with torch.no_grad():
                codes = out["code_indices"][:, :step_prefix]
                unique_codes.update(codes.flatten().cpu().tolist())

            if step % args.log_interval == 0 or step == steps_per_epoch:
                avg_rec = rec_loss_accum / step
                avg_vq = vq_loss_accum / step
                avg_acc = (acc_accum / step) * 100.0
                m_str = f"M: {step_prefix}" if args.prefix_distribution == "uniform" else f"M: {args.fixed_prefix_len}"
                print(
                    f"Epoch [{epoch}/{total_epochs}] Step [{step}/{steps_per_epoch}] {m_str} | "
                    f"Loss: {loss.item():.4f} (Rec: {avg_rec:.4f}, VQ: {avg_vq:.4f}) | "
                    f"Disc Acc: {avg_acc:.2f}% | Active Codes: {len(unique_codes)} | Grad: {grad_norm:.3f}",
                    flush=True
                )

        epoch_dur = time.time() - epoch_start_time

        # Validation Benchmarking
        val_metrics = evaluate(model, val_loader, device, max_val_steps=args.max_val_steps, prefix_len=args.fixed_prefix_len)
        benchmarks = multi_prefix_benchmark(model, val_loader, device, benchmarks=benchmarks_list, max_eval_steps=30)

        disc_bench_strs = [f"{bm} tok: {benchmarks[bm]['acc']:.2f}%" for bm in benchmarks_list]
        cont_bench_strs = [f"{bm} tok: {benchmarks[bm].get('cont_acc', 0.0):.2f}%" for bm in benchmarks_list]
        print(f"\n--> Epoch [{epoch}/{total_epochs}] Finished ({epoch_dur:.1f}s)")
        print(f"    Discrete Codebook Accuracies -> {' | '.join(disc_bench_strs)}")
        print(f"    Reference Continuous Accs   -> {' | '.join(cont_bench_strs)}")
        print(f"    Active Codebook Utilization  -> {len(unique_codes)} / {model.codebook_size} ({len(unique_codes)/model.codebook_size*100:.1f}%)")

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"discrete_codebook_epoch_{epoch}.pt")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_rec_loss": val_metrics["rec_loss"],
            "val_acc": val_metrics["true_acc"],
            "benchmarks": benchmarks,
            "active_codes": len(unique_codes),
            "word_dropout": word_drop,
            "args": vars(args)
        }
        torch.save(save_dict, ckpt_path)

        # Append to training_run.json
        log_path = os.path.join(args.save_dir, "training_run.json")
        log_data = {}
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_data = json.load(f)
            except Exception:
                log_data = {}
        if not log_data or epoch == 1:
            log_data = {
                "trainer": "train_discrete_codebook_frozen_encoder.py",
                "source_ckpt": args.source_ckpt,
                "command_args": vars(args),
                "epochs": []
            }
        log_data["epochs"].append({
            "epoch": epoch,
            "word_dropout": word_drop,
            "train_loss": total_loss_accum / steps_per_epoch,
            "train_rec_loss": rec_loss_accum / steps_per_epoch,
            "train_vq_loss": vq_loss_accum / steps_per_epoch,
            "train_acc": (acc_accum / steps_per_epoch) * 100.0,
            "val_acc_discrete": val_metrics["true_acc"],
            "val_rec_loss": val_metrics["rec_loss"],
            "benchmarks": benchmarks,
            "active_codes": len(unique_codes),
            "duration_sec": epoch_dur
        })
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)


if __name__ == "__main__":
    main()
