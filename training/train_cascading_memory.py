import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
import time
import json
import datetime
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cascading_memory_model import (
    SmallCascadingMemoryAutoencoder,
    CascadingMemoryAutoencoder,
    SmallDecoderOnlyAblationModel,
    DecoderOnlyAblationModel
)


def get_sigmoid_schedule(epoch, start_val=1.0, end_val=0.15, midpoint=4.0, temperature=1.0):
    if start_val == end_val:
        return start_val
    if epoch <= 1:
        return start_val
    decay_fraction = 1.0 / (1.0 + math.exp(-(epoch - midpoint) / max(temperature, 1e-4)))
    return start_val - decay_fraction * (start_val - end_val)


def get_linear_schedule(epoch, total_epochs, start_val=1.0, end_val=0.15):
    if total_epochs <= 1 or start_val == end_val:
        return start_val
    progress = min(max((epoch - 1) / (total_epochs - 1), 0.0), 1.0)
    return start_val - progress * (start_val - end_val)


def get_word_dropout(epoch, total_epochs, args):
    if not getattr(args, "masking", True):
        return 0.0
    sched = getattr(args, "schedule_type", "sigmoid")
    if sched == "constant":
        return getattr(args, "word_dropout", args.end_word_dropout)
    elif sched == "linear":
        return get_linear_schedule(
            epoch,
            total_epochs,
            start_val=args.start_word_dropout,
            end_val=args.end_word_dropout
        )
    else:
        # Default: sigmoid decay based on effective_epoch
        effective_epoch = epoch * (args.masking_ref_epochs / max(1, total_epochs))
        return get_sigmoid_schedule(
            effective_epoch,
            start_val=args.start_word_dropout,
            end_val=args.end_word_dropout,
            midpoint=args.masking_midpoint,
            temperature=args.masking_temperature
        )


def get_tau_schedule(epoch, total_epochs, args):
    if not getattr(args, "use_tau_skip", False):
        return 0.0
    if getattr(args, "learnable_tau", False):
        return None
    sched = getattr(args, "tau_schedule_type", "linear")
    if sched == "constant":
        return getattr(args, "init_tau", 0.5)
    elif sched == "linear":
        return get_linear_schedule(
            epoch,
            total_epochs,
            start_val=args.start_tau,
            end_val=args.end_tau
        )
    else:
        effective_epoch = epoch * (args.tau_ref_epochs / max(1, total_epochs))
        return get_sigmoid_schedule(
            effective_epoch,
            start_val=args.start_tau,
            end_val=args.end_tau,
            midpoint=args.tau_midpoint,
            temperature=args.tau_temperature
        )


def save_run_log(args, model_name, epoch_entries, device_name):
    """Appends per-epoch results to a permanent training_run.json in save_dir."""
    log_path = os.path.join(args.save_dir, "training_run.json")
    log = {}
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = {}
    if not log or (epoch_entries.get("epoch") == 1 and not args.resume_from):
        log = {
            "trainer": "train_cascading_memory.py",
            "model": model_name,
            "command_args": vars(args),
            "torch_version": torch.__version__,
            "device": device_name,
            "started_at": datetime.datetime.now().isoformat(),
            "epochs": [],
        }
    log["epochs"].append(epoch_entries)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def load_dataset(cache_path, batch_size=8):
    if not os.path.exists(cache_path):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt = os.path.join(root_dir, "data", os.path.basename(cache_path))
        if os.path.exists(alt):
            cache_path = alt
        elif os.path.exists(os.path.join("data", os.path.basename(cache_path))):
            cache_path = os.path.join("data", os.path.basename(cache_path))
        else:
            raise FileNotFoundError(f"Cache file {cache_path} not found! Run tokenization preprocessing first.")

    print(f"[CACHE] Loading cached tokenized data from {cache_path}...", flush=True)
    cached = torch.load(cache_path)
    if isinstance(cached, torch.Tensor):
        dataset = TensorDataset(cached.long(), (cached != 1).long())
    elif isinstance(cached, dict) and "input_ids_list" in cached:
        dense = torch.tensor([s[:64] + [1]*(64-len(s[:64])) for s in cached["input_ids_list"][:20000]], dtype=torch.long)
        dataset = TensorDataset(dense, (dense != 1).long())
    else:
        raise ValueError(f"Unexpected cache format in {cache_path}")

    total_samples = len(dataset)
    val_size = max(200, int(total_samples * 0.1))
    train_size = total_samples - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_size, val_size


def evaluate(model, val_loader, device, max_val_steps=100, prefix_len=32):
    model.eval()
    total_rec_loss = 0.0
    total_acc = 0.0
    total_cont_acc = 0.0
    steps = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_val_steps > 0 and i >= max_val_steps:
                break
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None
            # Evaluate using Discrete Codebook (skip_bottleneck=False, tau=0.0)
            out_disc = model(input_ids, attention_mask=attention_mask, prefix_len=prefix_len, train_probe=False, word_dropout=0.0, skip_bottleneck=False, tau=0.0)
            total_rec_loss += out_disc["rec_loss"].item()
            total_acc += out_disc["mean_true_acc"]

            if getattr(model, "skip_bottleneck", False):
                out_cont = model(input_ids, attention_mask=attention_mask, prefix_len=prefix_len, train_probe=False, word_dropout=0.0, skip_bottleneck=True)
                total_cont_acc += out_cont["mean_true_acc"]

            steps += 1
    res = {
        "rec_loss": total_rec_loss / max(1, steps),
        "true_acc": (total_acc / max(1, steps)) * 100.0
    }
    if getattr(model, "skip_bottleneck", False):
        res["cont_acc"] = (total_cont_acc / max(1, steps)) * 100.0
    return res


def multi_prefix_benchmark(model, val_loader, device, benchmarks=[64, 48, 32, 16], max_eval_steps=30):
    model.eval()
    results = {}
    with torch.no_grad():
        for m in benchmarks:
            tot_acc = 0.0
            tot_loss = 0.0
            tot_cont_acc = 0.0
            steps = 0
            for i, batch in enumerate(val_loader):
                if max_eval_steps > 0 and i >= max_eval_steps:
                    break
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None
                # Always benchmark on Discrete Codebook (skip_bottleneck=False, tau=0.0)
                out = model(input_ids, attention_mask=attention_mask, prefix_len=m, train_probe=False, word_dropout=0.0, skip_bottleneck=False, tau=0.0)
                tot_acc += out["mean_true_acc"]
                tot_loss += out["rec_loss"].item()
                if getattr(model, "skip_bottleneck", False):
                    out_c = model(input_ids, attention_mask=attention_mask, prefix_len=m, train_probe=False, word_dropout=0.0, skip_bottleneck=True)
                    tot_cont_acc += out_c["mean_true_acc"]
                steps += 1
            results[m] = {
                "acc": (tot_acc / max(1, steps)) * 100.0,
                "rec_loss": tot_loss / max(1, steps)
            }
            if getattr(model, "skip_bottleneck", False):
                results[m]["cont_acc"] = (tot_cont_acc / max(1, steps)) * 100.0
    return results


def train_cascading_memory(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == "cuda" else "CPU"
    print(f"Using device: {device} ({device_name})", flush=True)
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # 1. Load Data
    train_loader, val_loader, n_train, n_val = load_dataset(args.data_cache, batch_size=args.batch_size)
    steps_per_epoch = len(train_loader) if args.max_steps_per_epoch == -1 else min(args.max_steps_per_epoch, len(train_loader))
    print(f"[DATASET] Train: {n_train:,} ({len(train_loader)} batches), Val: {n_val:,} | Steps/Epoch: {steps_per_epoch}", flush=True)

    # 2. Instantiate Model
    init_tau_val = args.start_tau if args.start_tau is not None else args.init_tau
    if getattr(args, "ablate_encoder", False):
        if args.d_model <= 72:
            model = SmallDecoderOnlyAblationModel(
                vocab_size=tokenizer.vocab_size,
                d_model=args.d_model,
                nhead=args.nhead,
                decoder_num_layers=args.decoder_num_layers,
                max_length=args.max_length,
                word_dropout=args.end_word_dropout,
                norm_first=args.norm_first,
                tie_weights=args.tie_weights
            ).to(device)
        else:
            model = DecoderOnlyAblationModel(
                vocab_size=tokenizer.vocab_size,
                d_model=args.d_model,
                nhead=args.nhead,
                decoder_num_layers=args.decoder_num_layers,
                max_length=args.max_length,
                word_dropout=args.end_word_dropout,
                norm_first=args.norm_first,
                tie_weights=args.tie_weights
            ).to(device)
    elif args.d_model <= 72:
        model = SmallCascadingMemoryAutoencoder(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            decoder_num_layers=args.decoder_num_layers,
            text_encoder_layers=args.text_encoder_layers,
            latent_self_attn_layers=args.latent_self_attn_layers,
            share_embeddings=args.share_embeddings,
            tie_weights=args.tie_weights,
            codebook_size=args.codebook_size,
            max_length=args.max_length,
            k_samples=args.k_samples,
            noise_std=args.noise_std,
            exploration_mode=args.exploration_mode,
            selection_metric=args.selection_metric,
            probe_hidden_dim=args.probe_hidden_dim,
            probe_loss_weight=args.probe_loss_weight,
            use_cosine=args.use_cosine,
            reset_dead_codes=args.reset_dead_codes,
            use_tau_skip=args.use_tau_skip,
            init_tau=init_tau_val,
            tau_l1_coeff=args.tau_l1_coeff,
            temperature=args.temperature,
            learnable_tau=args.learnable_tau,
            skip_bottleneck=args.skip_bottleneck,
            vq_loss_weight=args.vq_loss_weight,
            norm_first=args.norm_first,
            causal_cross_attn=args.causal_cross_attn,
            causal_encoder_queries=args.causal_encoder_queries,
            normalize_prefix_pos=args.normalize_prefix_pos,
            prefix_pos_type=args.prefix_pos_type,
            latent_grad_scale_mode=args.latent_grad_scale_mode,
            latent_grad_scale_max=args.latent_grad_scale_max,
            adaptive_prefix_queries=args.adaptive_encoder_queries,
            use_compression_embeddings=args.use_compression_embeddings,
            append_m_query=args.append_m_query,
            diversity_loss_weight=args.diversity_loss_weight,
            diversity_threshold=args.diversity_threshold,
            diversity_loss_type=args.diversity_loss_type,
            fixed_prefix_len=args.fixed_prefix_len,
            prefix_distribution=args.prefix_distribution,
            min_prefix_len=args.min_prefix_len,
            max_prefix_len=args.max_prefix_len,
            word_dropout=args.end_word_dropout,
            mlp_prefix_pos=args.mlp_prefix_pos,
            mlp_pos_hidden_dim=args.mlp_pos_hidden_dim,
            indexer_mode=getattr(args, "indexer_mode", "prefix"),
            balance_loss_weight=getattr(args, "balance_loss_weight", 0.1),
            indexer_noise_std=getattr(args, "indexer_noise_std", 0.05)
        ).to(device)
    else:
        model = CascadingMemoryAutoencoder(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            decoder_num_layers=args.decoder_num_layers,
            text_encoder_layers=args.text_encoder_layers,
            latent_self_attn_layers=args.latent_self_attn_layers,
            share_embeddings=args.share_embeddings,
            tie_weights=args.tie_weights,
            codebook_size=args.codebook_size,
            max_length=args.max_length,
            k_samples=args.k_samples,
            noise_std=args.noise_std,
            exploration_mode=args.exploration_mode,
            selection_metric=args.selection_metric,
            probe_hidden_dim=args.probe_hidden_dim,
            probe_loss_weight=args.probe_loss_weight,
            use_cosine=args.use_cosine,
            reset_dead_codes=args.reset_dead_codes,
            use_tau_skip=args.use_tau_skip,
            init_tau=init_tau_val,
            tau_l1_coeff=args.tau_l1_coeff,
            temperature=args.temperature,
            learnable_tau=args.learnable_tau,
            skip_bottleneck=args.skip_bottleneck,
            vq_loss_weight=args.vq_loss_weight,
            norm_first=args.norm_first,
            causal_cross_attn=args.causal_cross_attn,
            causal_encoder_queries=args.causal_encoder_queries,
            normalize_prefix_pos=args.normalize_prefix_pos,
            prefix_pos_type=args.prefix_pos_type,
            latent_grad_scale_mode=args.latent_grad_scale_mode,
            latent_grad_scale_max=args.latent_grad_scale_max,
            adaptive_prefix_queries=args.adaptive_encoder_queries,
            use_compression_embeddings=args.use_compression_embeddings,
            append_m_query=args.append_m_query,
            diversity_loss_weight=args.diversity_loss_weight,
            diversity_threshold=args.diversity_threshold,
            diversity_loss_type=args.diversity_loss_type,
            fixed_prefix_len=args.fixed_prefix_len,
            prefix_distribution=args.prefix_distribution,
            min_prefix_len=args.min_prefix_len,
            max_prefix_len=args.max_prefix_len,
            word_dropout=args.end_word_dropout,
            mlp_prefix_pos=args.mlp_prefix_pos,
            mlp_pos_hidden_dim=args.mlp_pos_hidden_dim,
            indexer_mode=getattr(args, "indexer_mode", "prefix"),
            balance_loss_weight=getattr(args, "balance_loss_weight", 0.1),
            indexer_noise_std=getattr(args, "indexer_noise_std", 0.05)
        ).to(device)

    total_params = sum(p.numel() for p in set(model.parameters()))
    tau_info = f", TauSkip: {args.use_tau_skip} (tau={args.start_tau}->{args.end_tau})" if args.use_tau_skip else ""
    print(f"[MODEL] Architecture: {model.__class__.__name__} | Total Params: {total_params:,} (Tied/Shared: {args.tie_weights}, Pre-LN: {args.norm_first})", flush=True)
    prefix_info = f"Uniform M: [{args.min_prefix_len}, {args.max_prefix_len}]" if args.prefix_distribution == "uniform" else f"Fixed M: {args.fixed_prefix_len}"
    adapt_info = f" | CompressionEmbeddings: True (append_m_query: {args.append_m_query})" if args.use_compression_embeddings else (" | AdaptiveQueries: True" if args.adaptive_encoder_queries else "")
    grad_scale_info = f" | LatentGradScale: {args.latent_grad_scale_mode} (max={args.latent_grad_scale_max})" if args.latent_grad_scale_mode != "none" else ""
    indexer_info = f" | Indexer: {args.indexer_mode} (noise={args.indexer_noise_std}, bal_weight={args.balance_loss_weight})" if getattr(args, "indexer_mode", "prefix") != "prefix" else ""
    print(f"[SETTINGS] Stack: {args.text_encoder_layers} Text Enc -> {args.num_layers} Query Casc -> {args.latent_self_attn_layers} Latent Attn -> {args.decoder_num_layers} Dec | {prefix_info}{adapt_info}{grad_scale_info}{indexer_info} | Word Dropout: {args.start_word_dropout} -> {args.end_word_dropout} (sched: {args.schedule_type}){tau_info} | K: {args.k_samples} | Noise: {args.noise_std} | LR: {args.lr} | Reset Codes: {args.reset_dead_codes} | Skip Bottleneck: {args.skip_bottleneck} (VQ Weight: {args.vq_loss_weight})", flush=True)

    # Save model configuration
    model_config = {
        "model_class": model.__class__.__name__,
        "args": vars(args)
    }
    with open(os.path.join(args.save_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 1
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[RESUME] Loading checkpoint from {args.resume_from}...", flush=True)
        ckpt = torch.load(args.resume_from, map_location=device)
        ckpt_state = ckpt["model_state_dict"]

        # Support expanding or altering codebook size (e.g. 4096 -> 8192)
        if "encoder.vq.embedding.weight" in ckpt_state:
            old_w = ckpt_state["encoder.vq.embedding.weight"]
            new_w = model.encoder.vq.embedding.weight.data
            if old_w.shape[0] != new_w.shape[0]:
                print(f"[CODEBOOK EXPANSION] Expanding codebook from {old_w.shape[0]} to {new_w.shape[0]} codes...", flush=True)
                num_copy = min(old_w.shape[0], new_w.shape[0])
                new_w[:num_copy] = old_w[:num_copy].to(device)
                if new_w.shape[0] > old_w.shape[0]:
                    remaining = new_w.shape[0] - num_copy
                    tile_repeats = (remaining // num_copy) + 1
                    tiled = old_w.repeat(tile_repeats, 1)[:remaining].to(device)
                    jitter = torch.randn_like(tiled) * 0.02
                    new_w[num_copy:] = F.normalize(tiled + jitter, p=2, dim=-1)
                del ckpt_state["encoder.vq.embedding.weight"]
                if "probe.codebook_embedding.weight" in ckpt_state:
                    del ckpt_state["probe.codebook_embedding.weight"]
                if "encoder.vq.cluster_usage" in ckpt_state:
                    del ckpt_state["encoder.vq.cluster_usage"]

        model.load_state_dict(ckpt_state, strict=False)
        if not getattr(args, "reset_epoch", False):
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            print("[WARM-START] Resetting epoch counter to 1 for fine-tuning.", flush=True)

    print(f"Starting Cascading Memory Training from epoch {start_epoch} to {args.epochs} on {device}...\n", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()

        # Compute dynamic annealed word dropout
        current_word_dropout = get_word_dropout(epoch, args.epochs, args)
        effective_epoch = epoch * (args.masking_ref_epochs / max(1, args.epochs))

        # Compute dynamic annealed tau
        if args.use_tau_skip and not args.learnable_tau:
            current_tau = get_tau_schedule(epoch, args.epochs, args)
            model.set_tau(current_tau)
        elif args.use_tau_skip and args.learnable_tau:
            current_tau = model.encoder.vq.tau.item() if hasattr(model.encoder.vq, "tau") else 0.0
        else:
            current_tau = 0.0

        tau_str = f", tau={current_tau:.4f}" if args.use_tau_skip else ""
        print(f"--- Epoch [{epoch}/{args.epochs}] effective_epoch={effective_epoch:.1f}, word_dropout={current_word_dropout:.2f}{tau_str} ---", flush=True)

        total_loss_accum = 0.0
        rec_loss_accum = 0.0
        vq_loss_accum = 0.0
        probe_loss_accum = 0.0
        div_loss_accum = 0.0
        bal_loss_accum = 0.0
        acc_accum = 0.0
        unique_codes = set()
        optimizer.zero_grad()
        grad_norm = 0.0

        for step, batch in enumerate(train_loader, start=1):
            if args.max_steps_per_epoch != -1 and step > args.max_steps_per_epoch:
                break

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None

            if args.prefix_distribution == "uniform":
                step_prefix_len = torch.randint(args.min_prefix_len, args.max_prefix_len + 1, (1,)).item()
            else:
                step_prefix_len = args.fixed_prefix_len

            out = model(
                input_ids,
                attention_mask=attention_mask,
                prefix_len=step_prefix_len,
                word_dropout=current_word_dropout,
                tau=current_tau if args.use_tau_skip else None,
                k_samples=args.k_samples,
                noise_std=args.noise_std,
                exploration_mode=args.exploration_mode,
                selection_metric=args.selection_metric
            )

            loss = out["loss"]
            scaled_loss = loss / max(1, args.grad_accum_steps)
            scaled_loss.backward()

            if step % max(1, args.grad_accum_steps) == 0 or step == steps_per_epoch:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss_accum += loss.item()
            rec_loss_accum += out["rec_loss"].item()
            vq_loss_accum += out["vq_loss"].item()
            probe_loss_accum += out["probe_loss"].item()
            div_loss_accum += out["diversity_loss"].item() if "diversity_loss" in out else 0.0
            bal_loss_accum += out["balance_loss"].item() if "balance_loss" in out else 0.0
            acc_accum += out["mean_true_acc"]

            with torch.no_grad():
                codes = out["code_indices"][:, :step_prefix_len]
                unique_codes.update(codes.flatten().cpu().tolist())

            if step % args.log_interval == 0 or step == steps_per_epoch:
                avg_rec = rec_loss_accum / step
                avg_vq = vq_loss_accum / step
                avg_probe = probe_loss_accum / step
                avg_div = div_loss_accum / step
                avg_bal = bal_loss_accum / step
                avg_acc = (acc_accum / step) * 100.0
                win_str = f"| Wins: {[round(w, 2) for w in out.get('win_counts', [])]}" if args.k_samples > 1 else ""
                cos_str = f", Div: {avg_div:.4f} [cos={out.get('mean_intra_cos', 0.0):.2f}]" if args.diversity_loss_weight > 0 else ""
                bal_str = f", Bal: {avg_bal:.4f}" if getattr(args, "indexer_mode", "prefix") == "attention_topk" else ""
                m_str = f"M: {step_prefix_len}" if args.prefix_distribution == "uniform" else f"M: {args.fixed_prefix_len}"
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step}/{steps_per_epoch}] "
                    f"{m_str} (drop={current_word_dropout:.2f}{tau_str}) | Loss: {loss.item():.4f} (Rec: {avg_rec:.4f}, VQ: {avg_vq:.4f}{cos_str}{bal_str}) | "
                    f"Acc: {avg_acc:.2f}% (Pred: {out['mean_pred_acc']*100:.2f}%) | Grad: {grad_norm:.3f} {win_str}",
                    flush=True
                )

        epoch_duration = time.time() - epoch_start_time
        val_metrics = evaluate(model, val_loader, device, max_val_steps=args.max_val_steps, prefix_len=args.fixed_prefix_len)
        benchmark_targets = sorted(list(set([64, 48, 32, 16, 8, args.fixed_prefix_len])), reverse=True)
        benchmarks = multi_prefix_benchmark(model, val_loader, device, benchmarks=benchmark_targets, max_eval_steps=30)

        if getattr(args, "ablate_encoder", False):
            print(
                f"\n--> Epoch [{epoch}/{args.epochs}] Finished ({epoch_duration:.1f}s) | "
                f"Mode: Pure Next-Token Prediction LM (No Encoder, No Prefix) | "
                f"Train Loss: {rec_loss_accum / steps_per_epoch:.4f} | "
                f"Val Loss: {val_metrics['rec_loss']:.4f} | "
                f"Next-Token Prediction Val Acc: {val_metrics['true_acc']:.2f}%\n",
                flush=True
            )
        else:
            cont_str = f" | Continuous Val Acc: {val_metrics['cont_acc']:.2f}%" if "cont_acc" in val_metrics else ""
            active_prefix_str = f"Uniform [{args.min_prefix_len}..{args.max_prefix_len}]" if args.prefix_distribution == "uniform" else f"{args.fixed_prefix_len}"
            print(
                f"\n--> Epoch [{epoch}/{args.epochs}] Finished ({epoch_duration:.1f}s) | "
                f"Active Prefix: {active_prefix_str} | Word Dropout: {current_word_dropout:.2f}{tau_str} | "
                f"Train Rec: {rec_loss_accum / steps_per_epoch:.4f} | "
                f"Val Rec (Discrete): {val_metrics['rec_loss']:.4f} | "
                f"Val Acc (Discrete Codebook @ M={args.fixed_prefix_len}): {val_metrics['true_acc']:.2f}%{cont_str} | "
                f"Active Codes: {len(unique_codes)}/{args.codebook_size}",
                flush=True
            )
            if getattr(model, "skip_bottleneck", False):
                cont_bench_strs = [f"{bm} tok: {benchmarks[bm].get('cont_acc', 0.0):.2f}%" for bm in benchmark_targets]
                disc_bench_strs = [f"{bm} tok: {benchmarks[bm]['acc']:.2f}%" for bm in benchmark_targets]
                print(f"    Benchmark Continuous Accuracies -> {' | '.join(cont_bench_strs)}", flush=True)
                print(f"    Benchmark Discrete Accuracies   -> {' | '.join(disc_bench_strs)}\n", flush=True)
            else:
                print(
                    f"    Benchmark Discrete Accuracies -> "
                    f"64 tok (1x): {benchmarks[64]['acc']:.2f}% | "
                    f"48 tok (1.33x): {benchmarks[48]['acc']:.2f}% | "
                    f"32 tok (2x): {benchmarks[32]['acc']:.2f}% | "
                    f"16 tok (4x): {benchmarks[16]['acc']:.2f}%\n",
                    flush=True
                )

        ckpt_path = os.path.join(args.save_dir, f"cascading_memory_epoch_{epoch}.pt")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_rec_loss": val_metrics["rec_loss"],
            "val_acc": val_metrics["true_acc"],
            "benchmarks": benchmarks,
            "word_dropout": current_word_dropout,
            "tau": current_tau,
            "args": vars(args)
        }
        if "cont_acc" in val_metrics:
            save_dict["val_acc_continuous"] = val_metrics["cont_acc"]
        torch.save(save_dict, ckpt_path)

        # Log epoch summary to permanent JSON
        epoch_entry = {
            "epoch": epoch,
            "effective_epoch": effective_epoch,
            "word_dropout": current_word_dropout,
            "tau": current_tau,
            "train_total_loss": total_loss_accum / steps_per_epoch,
            "train_rec_loss": rec_loss_accum / steps_per_epoch,
            "train_vq_loss": vq_loss_accum / steps_per_epoch,
            "train_probe_loss": probe_loss_accum / steps_per_epoch,
            "train_diversity_loss": div_loss_accum / steps_per_epoch,
            "train_acc": (acc_accum / steps_per_epoch) * 100.0,
            "val_rec_loss": val_metrics["rec_loss"],
            "val_acc": val_metrics["true_acc"],
            "active_codes": len(unique_codes),
            "benchmarks": benchmarks,
            "duration_sec": epoch_duration
        }
        if "cont_acc" in val_metrics:
            epoch_entry["val_acc_continuous"] = val_metrics["cont_acc"]
        save_run_log(args, model.__class__.__name__, epoch_entry, device_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cascading Cross-Attention Memory Autoencoder")
    # Architecture
    parser.add_argument("--d_model", type=int, default=72, help="Hidden dimension (default: 72)")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--text_encoder_layers", type=int, default=3, help="Bidirectional text TransformerEncoder layers (default: 3)")
    parser.add_argument("--num_layers", type=int, default=3, help="Cascading query cross-attention layers (default: 3)")
    parser.add_argument("--decoder_num_layers", type=int, default=3, help="Autoregressive decoder layers (default: 3)")
    parser.add_argument("--codebook_size", type=int, default=4096, help="VQ codebook size (default: 4096)")
    parser.add_argument("--max_length", type=int, default=64, help="Max sequence length (default: 64)")
    parser.add_argument("--fixed_prefix_len", type=int, default=32, help="Fixed prefix length M (default: 32)")
    parser.add_argument("--prefix_distribution", type=str, default="fixed", choices=["fixed", "uniform"],
                        help="Prefix length distribution during training: fixed or uniform (default: fixed)")
    parser.add_argument("--min_prefix_len", type=int, default=8, help="Minimum prefix length when uniform (default: 8)")
    parser.add_argument("--max_prefix_len", type=int, default=64, help="Maximum prefix length when uniform (default: 64)")

    # Word Dropout & Annealing Schedule
    parser.add_argument("--word_dropout", type=float, default=0.15, help="Constant word dropout rate if schedule_type=constant (default: 0.15)")
    parser.add_argument("--start_word_dropout", type=float, default=1.0, help="Initial word dropout rate at epoch 1 (default: 1.0)")
    parser.add_argument("--end_word_dropout", type=float, default=0.15, help="Final target word dropout rate (default: 0.15)")
    parser.add_argument("--masking_midpoint", type=float, default=4.0, help="Sigmoid decay midpoint epoch (default: 4.0)")
    parser.add_argument("--masking_temperature", type=float, default=1.0, help="Sigmoid decay sharpness/temperature (default: 1.0)")
    parser.add_argument("--masking_ref_epochs", type=float, default=10.0, help="Reference epoch scale for decay schedule (default: 10.0)")
    parser.add_argument("--schedule_type", type=str, default="sigmoid", choices=["sigmoid", "linear", "constant"], help="Word dropout schedule type (default: sigmoid)")
    parser.add_argument("--masking", dest="masking", action="store_true", default=True, help="Enable word dropout masking and annealing (default: True)")
    parser.add_argument("--no_masking", dest="masking", action="store_false", help="Disable word dropout masking entirely")

    # Tau Annealing / Soft-Hard VQ
    parser.add_argument("--use_tau_skip", action="store_true", default=False,
                        help="Enable continuous-to-discrete SoftHard VQ tau blending")
    parser.add_argument("--no_tau_skip", dest="use_tau_skip", action="store_false")
    parser.add_argument("--start_tau", type=float, default=0.8, help="Initial tau at epoch 1 (default: 0.8)")
    parser.add_argument("--end_tau", type=float, default=0.0, help="Final tau at last epoch (default: 0.0)")
    parser.add_argument("--init_tau", type=float, default=0.8, help="Init tau if constant (default: 0.8)")
    parser.add_argument("--tau_schedule_type", type=str, default="linear", choices=["linear", "sigmoid", "constant"],
                        help="Tau annealing schedule (default: linear)")
    parser.add_argument("--tau_midpoint", type=float, default=4.0, help="Sigmoid tau midpoint epoch (default: 4.0)")
    parser.add_argument("--tau_temperature", type=float, default=1.0, help="Sigmoid tau decay temperature (default: 1.0)")
    parser.add_argument("--tau_ref_epochs", type=float, default=10.0, help="Reference epoch scale for tau decay (default: 10.0)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for soft codebook distribution (default: 1.0)")
    parser.add_argument("--tau_l1_coeff", type=float, default=0.01, help="L1 regularizer weight if learnable_tau is True (default: 0.01)")
    parser.add_argument("--learnable_tau", action="store_true", default=False, help="Make tau a learnable parameter")

    # Exploration & Closest-Sample Training
    parser.add_argument("--k_samples", type=int, default=1, help="Number of exploration candidate samples (default: 1)")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Exploration noise standard deviation (default: 0.1)")
    parser.add_argument("--exploration_mode", type=str, default="latent", choices=["latent", "gumbel", "both"], help="Exploration mode (default: latent)")
    parser.add_argument("--selection_metric", type=str, default="loss", choices=["loss", "acc"], help="Closest sample metric (default: loss)")

    # Probe & VQ
    parser.add_argument("--probe_hidden_dim", type=int, default=128, help="Probe hidden dimension")
    parser.add_argument("--probe_loss_weight", type=float, default=1.0, help="Probe loss weight")
    parser.add_argument("--use_cosine", action="store_true", default=True, help="Cosine normalized VQ")
    parser.add_argument("--reset_dead_codes", action="store_true", default=False,
                        help="Enable dead-code revival / vector resetting in the VQ; pass --no_reset_dead_codes to disable")
    parser.add_argument("--no_reset_dead_codes", dest="reset_dead_codes", action="store_false")
    parser.add_argument("--skip_bottleneck", action="store_true", default=False,
                        help="Skip discrete latent token bottleneck and feed full continuous latent vectors directly to decoder")
    parser.add_argument("--no_skip_bottleneck", dest="skip_bottleneck", action="store_false")
    parser.add_argument("--vq_loss_weight", type=float, default=1.0,
                        help="Weight for VQ codebook loss (default: 1.0; set to 0.0 to completely detach/disable VQ loss)")
    parser.add_argument("--latent_self_attn_layers", type=int, default=1,
                        help="Bidirectional self-attention layers on retained prefix latents (default: 1)")
    parser.add_argument("--share_embeddings", action="store_true", default=True,
                        help="Share text embedding matrix between encoder and decoder (default: True)")
    parser.add_argument("--no_share_embeddings", dest="share_embeddings", action="store_false")
    parser.add_argument("--tie_weights", action="store_true", default=True,
                        help="Tie decoder fc_out projection weight with text embedding (default: True)")
    parser.add_argument("--no_tie_weights", dest="tie_weights", action="store_false")
    parser.add_argument("--norm_first", action="store_true", default=True,
                        help="Enable Pre-LN (norm_first=True) for direct identity gradient flow (default: True)")
    parser.add_argument("--no_norm_first", dest="norm_first", action="store_false",
                        help="Use Post-LN (norm_first=False)")
    parser.add_argument("--causal_cross_attn", action="store_true", default=True,
                        help="Enable causal memory mask on encoder cross-attention (Query i only attends to Text 0..i) (default: True)")
    parser.add_argument("--no_causal_cross_attn", dest="causal_cross_attn", action="store_false",
                        help="Allow queries to attend to full text without causal cross-attention masking")
    parser.add_argument("--causal_encoder_queries", action="store_true", default=True,
                        help="Enable causal mask on encoder latent queries (default: True)")
    parser.add_argument("--no_causal_encoder_queries", dest="causal_encoder_queries", action="store_false",
                        help="Disable causal mask on encoder queries (Exp 1: non-causal encoder's decoder)")
    parser.add_argument("--normalize_prefix_pos", action="store_true", default=True,
                        help="Normalize positional encoding for the M prefix tokens based on M (default: True)")
    parser.add_argument("--no_normalize_prefix_pos", dest="normalize_prefix_pos", action="store_false",
                        help="Disable positional encoding normalization for prefix tokens")
    parser.add_argument("--prefix_pos_type", type=str, default="interpolated", choices=["interpolated", "sinusoidal", "unnormalized"],
                        help="Prefix positional encoding type (default: interpolated)")
    parser.add_argument("--latent_grad_scale_mode", type=str, default="none", choices=["none", "inv_freq", "m_over_min_m", "combined"],
                        help="Token-wise gradient normalization scheme across variable M prefix (default: none)")
    parser.add_argument("--latent_grad_scale_max", type=float, default=20.0,
                        help="Maximum gradient scale factor for inverse frequency weighting (default: 20.0)")
    parser.add_argument("--adaptive_encoder_queries", action="store_true", default=True,
                        help="Generate M compression-adaptive queries in encoder with normalized positional encoding (default: True)")
    parser.add_argument("--no_adaptive_encoder_queries", dest="adaptive_encoder_queries", action="store_false",
                        help="Disable adaptive encoder queries (compute all 64 and slice)")
    parser.add_argument("--use_compression_embeddings", action="store_true", default=False,
                        help="Option A: Use learned compression-level embedding c_M (default: False)")
    parser.add_argument("--no_use_compression_embeddings", dest="use_compression_embeddings", action="store_false",
                        help="Disable learned compression-level embeddings")
    parser.add_argument("--append_m_query", action="store_true", default=True,
                        help="Append learned M embedding as a separate query token [M + 1] (default: True)")
    parser.add_argument("--no_append_m_query", dest="append_m_query", action="store_false",
                        help="Disable appending learned M embedding as a separate query token (elementwise add instead)")
    parser.add_argument("--mlp_prefix_pos", "--proj_prefix_pos", dest="mlp_prefix_pos", action="store_true", default=False,
                        help="Enable per-position learned linear projection conditioned on normalized budget M / max_length (default: False)")
    parser.add_argument("--no_mlp_prefix_pos", "--no_proj_prefix_pos", dest="mlp_prefix_pos", action="store_false")
    parser.add_argument("--mlp_pos_hidden_dim", type=int, default=64,
                        help="Unused for single-layer linear projection; kept for backward compatibility (default: 64)")
    parser.add_argument("--ablate_encoder", action="store_true", default=False,
                        help="Ablate encoder and skip cross-attention layers in decoder (Exp 2)")
    parser.add_argument("--diversity_loss_weight", type=float, default=0.5,
                        help="Weight for intra-sequence pairwise cosine diversity loss (default: 0.5)")
    parser.add_argument("--diversity_threshold", type=float, default=0.2,
                        help="Cosine threshold if using hinge diversity loss (default: 0.2)")
    parser.add_argument("--diversity_loss_type", type=str, default="squared", choices=["squared", "hinge"],
                        help="Diversity loss formula: squared off-diagonal cosine or hinge (default: squared)")

    # Attention-Guided Top-M Indexer
    parser.add_argument("--indexer_mode", type=str, default="prefix", choices=["prefix", "attention_topk"],
                        help="Latent retention mode: 'prefix' (contiguous slice 0..M) or 'attention_topk' (attention-guided Top-M selection) (default: prefix)")
    parser.add_argument("--balance_loss_weight", type=float, default=0.1,
                        help="Auxiliary load-balancing loss weight for attention_topk indexer (default: 0.1)")
    parser.add_argument("--indexer_noise_std", type=float, default=0.05,
                        help="Exploration noise std for attention_topk indexer (default: 0.05)")

    # Optimization & Schedules
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (effective batch size = batch_size * grad_accum_steps) (default: 1)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_steps_per_epoch", type=int, default=-1, help="Max steps per epoch (-1 for full)")
    parser.add_argument("--max_val_steps", type=int, default=100, help="Max validation steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between log prints")

    # Paths
    parser.add_argument("--data_cache", type=str, default="tokenized_fast_tensor_64.pt", help="Cached data tensor")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base", help="Tokenizer name")
    parser.add_argument("--save_dir", type=str, default="models/checkpoints_cascading_memory_m32", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume checkpoint path")
    parser.add_argument("--reset_epoch", action="store_true", default=False, help="Reset epoch counter to 1 when warm-starting from a checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()
    train_cascading_memory(args)
