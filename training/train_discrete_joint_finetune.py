import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import time
import math
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
    parser = argparse.ArgumentParser(description="End-to-End Joint Fine-tuning of Encoder and Decoder with Discrete Codebook")
    # Paths & Checkpoints
    parser.add_argument("--source_ckpt", type=str,
                        default="models/checkpoints_discrete_joint_finetune/joint_discrete_epoch_1.pt",
                        help="Path to joint discrete checkpoint")
    parser.add_argument("--fallback_ckpt", type=str,
                        default="models/checkpoints_discrete_frozen_encoder_cb16384/discrete_codebook_epoch_5.pt",
                        help="Fallback path if epoch 1 is not found")
    parser.add_argument("--save_dir", type=str,
                        default="models/checkpoints_discrete_joint_cpu",
                        help="Save directory for CPU joint fine-tuned checkpoints")
    parser.add_argument("--data_cache", type=str, default="tokenized_fast_tensor_64.pt",
                        help="Tokenized dataset cache path")

    # K-Means Codebook Refresh
    parser.add_argument("--re_kmeans", action="store_true", default=True,
                        help="Re-fit spherical K-Means centroids on current encoder latents (default: True)")
    parser.add_argument("--no_re_kmeans", action="store_false", dest="re_kmeans",
                        help="Disable re-fitting K-Means centroids")
    parser.add_argument("--kmeans_batches", type=int, default=100,
                        help="Number of train batches to extract latents for K-Means (default: 100)")
    parser.add_argument("--kmeans_iters", type=int, default=10,
                        help="Number of spherical K-Means iterations (default: 10)")
    parser.add_argument("--kmeans_interval", type=int, default=2,
                        help="Run spherical K-Means codebook refresh every N epochs (0 to disable, default: 2)")
    parser.add_argument("--start_epoch", type=int, default=None,
                        help="Starting epoch number (default: auto-detect from checkpoint + 1)")

    # Architecture / Prefix
    parser.add_argument("--prefix_distribution", type=str, default="uniform", choices=["fixed", "uniform"],
                        help="Prefix distribution during fine-tuning (default: uniform)")
    parser.add_argument("--min_prefix_len", type=int, default=8, help="Min prefix length (default: 8)")
    parser.add_argument("--max_prefix_len", type=int, default=64, help="Max prefix length (default: 64)")
    parser.add_argument("--fixed_prefix_len", type=int, default=32, help="Fixed prefix length if fixed (default: 32)")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--encoder_lr", type=float, default=0.0001, help="Encoder learning rate (default: 0.0001)")
    parser.add_argument("--decoder_lr", type=float, default=0.00015, help="Decoder/Codebook learning rate (default: 0.00015)")
    parser.add_argument("--vq_loss_weight", type=float, default=0.1, help="VQ commitment loss weight (default: 0.1)")
    parser.add_argument("--start_word_dropout", type=float, default=0.28, help="Start word dropout (default: 0.28)")
    parser.add_argument("--end_word_dropout", type=float, default=0.05, help="End word dropout (default: 0.05)")
    parser.add_argument("--max_val_steps", type=int, default=50, help="Max validation steps")
    parser.add_argument("--log_interval", type=int, default=50, help="Log step interval (default: 50)")
    parser.add_argument("--num_threads", type=int, default=6, help="CPU PyTorch thread count (default: 6)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (default: cpu)")
    return parser.parse_args()


def fit_spherical_kmeans(model, train_loader, device, K=16384, num_batches=100, iters=10):
    """
    Extracts continuous latent vectors from the encoder and fits K spherical centroids
    using fast vectorized index_add_ with dead centroid replacement.
    """
    print(f"\n[CLUSTERING] Extracting continuous latents from {num_batches} batches on {device}...")
    model.eval()
    latents = []
    with torch.no_grad():
        for step, batch in enumerate(train_loader):
            if step >= num_batches:
                break
            input_ids = batch[0].to(device)
            p_len = torch.randint(8, 65, (1,)).item()
            _, _, _, h = model.encoder(input_ids, prefix_len=p_len, return_continuous=True)
            has_appended = getattr(model, "append_m_query", False) and getattr(model, "use_compression_embeddings", False)
            exp_len = p_len + 1 if has_appended else p_len
            latents.append(h[:, :exp_len, :].reshape(-1, model.d_model))

    X = torch.cat(latents, dim=0) # [N, d_model]
    N = X.size(0)
    print(f"[CLUSTERING] Collected {N:,} continuous latent vectors. L2-normalizing for Spherical K-Means...")
    X_norm = F.normalize(X, p=2, dim=-1)

    K = min(K, N)
    rand_idx = torch.randperm(N, device=device)[:K]
    centroids = X_norm[rand_idx].clone()

    print(f"[CLUSTERING] Running {iters} iterations of vectorized spherical mini-batch K-Means (K={K:,})...")
    batch_sz = min(16384, N)
    for it in range(1, iters + 1):
        idx_chunk = torch.randint(0, N, (batch_sz,), device=device)
        chunk = X_norm[idx_chunk]
        sim = torch.matmul(chunk, centroids.t())
        best_cluster = torch.argmax(sim, dim=-1) # [batch_sz]
        
        # Fast vectorized cluster accumulation
        counts = torch.bincount(best_cluster, minlength=K).float().unsqueeze(1) # [K, 1]
        sum_vecs = torch.zeros(K, model.d_model, device=device)
        sum_vecs.index_add_(0, best_cluster, chunk)
        
        active_mask = (counts.squeeze(1) > 0)
        centroids[active_mask] = F.normalize(sum_vecs[active_mask], p=2, dim=-1)
        
        # Active replacement for unused clusters
        empty_count = (~active_mask).sum().item()
        if empty_count > 0:
            rand_replace = torch.randint(0, N, (empty_count,), device=device)
            centroids[~active_mask] = X_norm[rand_replace]
            
        if it % 2 == 0 or it == iters:
            print(f"  Iteration [{it:2d}/{iters}] completed. Active clusters: {active_mask.sum().item():,}/{K:,}")

    sample_eval = X_norm[:10000]
    sim_eval = torch.matmul(sample_eval, centroids.t()).max(dim=-1).values
    avg_cos = sim_eval.mean().item()
    avg_deg = math.acos(min(1.0, max(-1.0, avg_cos))) * 180 / 3.14159
    print(f"[CLUSTERING] Centroids fitted! Mean alignment: cos={avg_cos:.4f} (angular distortion: {avg_deg:.2f} deg)\n")
    return centroids


def run_benchmark(model, val_loader, device, benchmarks=[64, 48, 32, 16, 8], max_eval_steps=20):
    """
    Evaluates both Discrete and Continuous modes across target prefix budgets using multi_prefix_benchmark.
    """
    model.eval()
    orig_skip = model.skip_bottleneck
    model.skip_bottleneck = True
    bench_results = multi_prefix_benchmark(model, val_loader, device, benchmarks=benchmarks, max_eval_steps=max_eval_steps)
    model.skip_bottleneck = orig_skip
    return bench_results


def main():
    args = parse_args()
    os.makedirs("debug_outputs", exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # Configure CPU multi-threading
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)
        print(f"[CPU] Configured PyTorch intra-op threads to: {torch.get_num_threads()}")

    # Resolve source checkpoint
    target_ckpt = args.source_ckpt
    if not os.path.exists(target_ckpt):
        if os.path.exists(args.fallback_ckpt):
            print(f"[INIT] {target_ckpt} not found, falling back to {args.fallback_ckpt}")
            target_ckpt = args.fallback_ckpt
        else:
            raise FileNotFoundError(f"Neither {target_ckpt} nor {args.fallback_ckpt} found!")

    print("=" * 85)
    print("      JOINT DISCRETE FINE-TUNING (CPU MODE + RE-KMEANS CODEBOOK REFRESH)")
    print("=" * 85)
    print(f"Source Checkpoint: {target_ckpt}")
    print(f"Save Directory:    {args.save_dir}")
    print(f"Device:            {device}")
    print(f"Encoder LR:        {args.encoder_lr}")
    print(f"Decoder LR:        {args.decoder_lr}")

    # 1. Load dataset
    train_loader, val_loader, train_size, val_size = load_dataset(args.data_cache, batch_size=args.batch_size)

    # 2. Load Checkpoint
    ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
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
        vq_loss_weight=args.vq_loss_weight,
        normalize_prefix_pos=ckpt_args.get("normalize_prefix_pos", False),
        append_m_query=ckpt_args.get("append_m_query", True),
        use_compression_embeddings=ckpt_args.get("use_compression_embeddings", True)
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[MODEL] Successfully loaded model state from {target_ckpt} (Epoch {ckpt.get('epoch', '?')})")

    # 3. Optional Re-KMeans Codebook Refresh
    if args.re_kmeans:
        new_centroids = fit_spherical_kmeans(
            model,
            train_loader,
            device,
            K=model.codebook_size,
            num_batches=args.kmeans_batches,
            iters=args.kmeans_iters
        )
        model.encoder.vq.embedding.weight.data.copy_(new_centroids)
        print(f"[CODEBOOK] Successfully refreshed all {model.codebook_size:,} codebook entries with re-fitted K-Means centroids!\n")

    # 4. Unfreeze ALL parameters
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        param.requires_grad = True
        if "encoder" in name or "pos_encoder" in name or "memory_layers" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    enc_count = sum(p.numel() for p in encoder_params)
    dec_count = sum(p.numel() for p in decoder_params)
    print(f"[PARAMETERS] Configured ALL parameters as trainable:")
    print(f"  Encoder Parameters: {enc_count:,} (LR={args.encoder_lr})")
    print(f"  Decoder/VQ Params:  {dec_count:,} (LR={args.decoder_lr})")
    print(f"  Total Trainable:    {enc_count + dec_count:,}")

    # 5. Baseline Evaluation Before Joint Training
    print("\n[EVAL] Running baseline evaluation on both modes...")
    benchmarks_list = [64, 48, 32, 16, 8]
    base_bench = run_benchmark(model, val_loader, device, benchmarks=benchmarks_list, max_eval_steps=20)
    disc_accs = [f"{m}: {base_bench[m]['acc']:.2f}%" for m in benchmarks_list]
    cont_accs = [f"{m}: {base_bench[m].get('cont_acc', 0.0):.2f}%" for m in benchmarks_list]
    print(f"  Baseline Discrete Codebook:   {' | '.join(disc_accs)}")
    print(f"  Baseline Continuous Latents:  {' | '.join(cont_accs)}")

    # 6. Optimizer with Parameter Groups
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_lr},
        {"params": decoder_params, "lr": args.decoder_lr}
    ])

    # Save initial config
    model_config = {
        "model_class": model.__class__.__name__,
        "args": vars(args),
        "source_ckpt": target_ckpt,
        "re_kmeans": args.re_kmeans,
        "encoder_params": enc_count,
        "decoder_params": dec_count
    }
    with open(os.path.join(args.save_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # 7. Fine-Tuning Loop
    total_epochs = args.epochs
    steps_per_epoch = len(train_loader)
    
    start_epoch = args.start_epoch if args.start_epoch is not None else (ckpt.get("epoch", 0) + 1 if "epoch" in ckpt else 1)
    print(f"\n[TRAIN] Beginning JOINT discrete training on {device} from epoch {start_epoch} to {total_epochs} (skip_bottleneck=False)...")
    if args.kmeans_interval > 0:
        print(f"[TRAIN] Periodic codebook refresh: will run Spherical K-Means every {args.kmeans_interval} epochs.")

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()
        progress = (epoch - 1) / max(1, total_epochs - 1)
        word_drop = args.start_word_dropout - progress * (args.start_word_dropout - args.end_word_dropout)

        print(f"\n--- Epoch [{epoch}/{total_epochs}] Joint Mode (word_dropout={word_drop:.2f}) ---")
        model.train()
        model.skip_bottleneck = False

        total_loss_accum = 0.0
        rec_loss_accum = 0.0
        vq_loss_accum = 0.0
        acc_accum = 0.0
        unique_codes = set()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None

            # Uniform sample prefix length M in [8, 64]
            if args.prefix_distribution == "uniform":
                prefix_len = torch.randint(args.min_prefix_len, args.max_prefix_len + 1, (1,)).item()
            else:
                prefix_len = args.fixed_prefix_len

            optimizer.zero_grad()

            out = model(
                input_ids,
                attention_mask=attention_mask,
                prefix_len=prefix_len,
                word_dropout=word_drop,
                skip_bottleneck=False,
                tau=0.0
            )

            loss = out["loss"]
            loss.backward()

            # Gradient clipping across all parameters
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item()
            optimizer.step()

            # Metrics tracking
            total_loss_accum += loss.item()
            rec_loss_accum += out["rec_loss"].item()
            vq_loss_accum += out["vq_loss"].item()
            acc_accum += out["mean_true_acc"]

            with torch.no_grad():
                codes = out["code_indices"][:, :prefix_len]
                unique_codes.update(codes.flatten().cpu().tolist())

            if step % args.log_interval == 0:
                avg_loss = total_loss_accum / step
                avg_rec = rec_loss_accum / step
                avg_vq = vq_loss_accum / step
                avg_acc = (acc_accum / step) * 100.0

                print(
                    f"Epoch [{epoch}/{total_epochs}] Step [{step}/{steps_per_epoch}] "
                    f"M: {prefix_len:2d} | "
                    f"Loss: {loss.item():.4f} (Rec: {avg_rec:.4f}, VQ: {avg_vq:.4f}) | "
                    f"Disc Acc: {avg_acc:.2f}% | "
                    f"Active Codes: {len(unique_codes)} | "
                    f"Grad: {grad_norm:.3f}"
                )

        epoch_dur = time.time() - epoch_start_time

        # Validation Benchmarking (Both Discrete and Continuous)
        bench = run_benchmark(model, val_loader, device, benchmarks=benchmarks_list, max_eval_steps=25)

        disc_bench_strs = [f"{bm} tok: {bench[bm]['acc']:.2f}%" for bm in benchmarks_list]
        cont_bench_strs = [f"{bm} tok: {bench[bm].get('cont_acc', 0.0):.2f}%" for bm in benchmarks_list]
        print(f"\n--> Epoch [{epoch}/{total_epochs}] Finished ({epoch_dur:.1f}s)")
        print(f"    Discrete Codebook Accuracies -> {' | '.join(disc_bench_strs)}")
        print(f"    Continuous Latents Accs      -> {' | '.join(cont_bench_strs)}")
        print(f"    Active Codebook Utilization  -> {len(unique_codes)} / {model.codebook_size} ({len(unique_codes)/model.codebook_size*100:.1f}%)")

        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"joint_discrete_epoch_{epoch}.pt")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "benchmarks": bench,
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
                "trainer": "train_discrete_joint_finetune.py",
                "source_ckpt": target_ckpt,
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
            "benchmarks": bench,
            "active_codes": len(unique_codes),
            "duration_sec": epoch_dur
        })
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        # Periodic K-Means refresh between every other epoch
        if args.kmeans_interval > 0 and (epoch % args.kmeans_interval == 0) and (epoch < total_epochs):
            print(f"\n[K-MEANS REFRESH] Triggering periodic Spherical K-Means refresh (Epoch {epoch} finished, interval: every {args.kmeans_interval} epochs)...")
            new_centroids = fit_spherical_kmeans(
                model,
                train_loader,
                device,
                K=model.codebook_size,
                num_batches=args.kmeans_batches,
                iters=args.kmeans_iters
            )
            model.encoder.vq.embedding.weight.data.copy_(new_centroids)
            print(f"[CODEBOOK] Successfully refreshed all {model.codebook_size:,} codebook entries with re-fitted centroids!")
            print("[EVAL] Running post-KMeans baseline evaluation:")
            post_bench = run_benchmark(model, val_loader, device, benchmarks=benchmarks_list, max_eval_steps=20)
            post_disc_strs = [f"{bm} tok: {post_bench[bm]['acc']:.2f}%" for bm in benchmarks_list]
            print(f"    Post-KMeans Discrete Codebook -> {' | '.join(post_disc_strs)}\n")


if __name__ == "__main__":
    main()
