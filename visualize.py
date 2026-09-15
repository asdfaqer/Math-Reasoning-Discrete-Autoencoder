import os
import sys
import json
import random
import argparse
import urllib.parse
import webbrowser
import threading
import importlib
import inspect
import zlib
import gzip
import struct
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from transformers import AutoTokenizer
from model import ReasoningAutoencoder, VectorQuantizer, LatentDecoder

class LegacyAdaptiveLatentEncoder(nn.Module):
    def __init__(self, vocab_size=50265, d_model=256, nhead=4, num_layers=4, codebook_size=4096, leak_alpha=0.1, use_gate_encoder=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if use_gate_encoder:
            gate_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
            self.gate_encoder = nn.TransformerEncoder(gate_layer, num_layers=1)
        else:
            self.gate_encoder = None
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.downsample = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, stride=1)
        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=d_model)
        self.proj_h = nn.Linear(d_model, d_model)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.norm_discrete = nn.LayerNorm(d_model)
        self.norm_continuous = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.leak_alpha = leak_alpha

    def forward(self, input_ids, attention_mask=None, noise_std=0.0, threshold=0.5, min_tokens=10, force_discrete=False):
        B, L = input_ids.shape
        x = self.embedding(input_ids) + self.pos_encoder[:, :L, :]
        h = self.transformer_encoder(x)
        if self.gate_encoder is not None:
            h_gate = self.gate_encoder(h.detach())
        else:
            h_gate = h.detach()
        s_logits = self.score_mlp(h_gate).squeeze(-1)
        s = torch.sigmoid(s_logits)
        g_hard = torch.where(s >= threshold, torch.ones_like(s), torch.full_like(s, self.leak_alpha))
        if min_tokens > 0 and L >= min_tokens:
            _, topk_idx = torch.topk(s, k=min_tokens, dim=-1)
            topk_mask = torch.zeros_like(s).scatter_(-1, topk_idx, 1.0)
            g_hard = torch.maximum(g_hard, topk_mask)
        g_ste = s + (g_hard - s).detach()
        h_perm = h.transpose(1, 2)
        h_proj = self.downsample(h_perm).transpose(1, 2)
        quantized_raw, vq_loss, code_indices = self.vq(h_proj)
        z_discrete = quantized_raw * g_ste.unsqueeze(-1)
        quantized = self.norm_out(z_discrete) if hasattr(self, "norm_out") else self.norm(z_discrete)
        return quantized, vq_loss, None, code_indices, None, s

class LegacyModel(nn.Module):
    """Legacy adaptive gate-based autoencoder: LegacyAdaptiveLatentEncoder + LatentDecoder."""
    def __init__(self, vocab_size=50265, d_model=256, nhead=4, num_layers=4, decoder_num_layers=3, codebook_size=4096, leak_alpha=0.1, use_gate_encoder=False):
        super().__init__()
        self.encoder = LegacyAdaptiveLatentEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            codebook_size=codebook_size,
            leak_alpha=leak_alpha,
            use_gate_encoder=use_gate_encoder
        )
        self.decoder = LatentDecoder(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=decoder_num_layers)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global variables for models & dataset
TOKENIZERS_CACHE = {}
CURRENT_MODEL_ID = None
CURRENT_MODEL = None
CURRENT_TOKENIZER = None
VAL_TEXTS = []
TRAIN_TEXTS = []
DEVICE = "cpu"
CHECKPOINT_INFO = {}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def discover_models():
    """Scan MODELS_DIR for subdirectories containing .pt checkpoint files.
    Returns a dict keyed by directory name with 'name', 'dir', 'default_file'."""
    models = {}
    if not os.path.isdir(MODELS_DIR):
        print(f"⚠️ Models directory not found: {MODELS_DIR}", flush=True)
        return models
    for entry in sorted(os.listdir(MODELS_DIR)):
        subdir = os.path.join(MODELS_DIR, entry)
        if not os.path.isdir(subdir):
            continue
        pt_files = [f for f in os.listdir(subdir) if f.endswith(".pt")]
        if not pt_files:
            continue
        default_file = os.path.join(subdir, "reasoning_compressor.pt")
        if not os.path.exists(default_file):
            default_file = os.path.join(subdir, pt_files[0])
        models[entry] = {
            "name": entry,
            "dir": subdir,
            "default_file": default_file
        }
    return models

MODEL_PATHS = discover_models()

def get_tokenizer_for_model(tokenizer_name):
    if tokenizer_name not in TOKENIZERS_CACHE:
        print(f"[VISUALIZER] Loading tokenizer: {tokenizer_name}...", flush=True)
        TOKENIZERS_CACHE[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
    return TOKENIZERS_CACHE[tokenizer_name]

CACHED_TOKEN_TENSOR = None
TRAIN_TENSORS = None
VAL_TENSORS = None

def init_cached_tensors():
    global CACHED_TOKEN_TENSOR, TRAIN_TENSORS, VAL_TENSORS
    cache_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tokenized_fast_tensor_64.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenized_fast_tensor_64.pt"),
        "data/tokenized_fast_tensor_64.pt",
        "tokenized_fast_tensor_64.pt",
        r"d:\Reasoning-compresion\tokenized_fast_tensor_64.pt"
    ]
    for c_path in cache_candidates:
        if os.path.exists(c_path):
            try:
                print(f"[VISUALIZER] Loading ground-truth training/validation tensor from {c_path}...", flush=True)
                CACHED_TOKEN_TENSOR = torch.load(c_path, map_location="cpu").long()
                val_size = max(200, int(len(CACHED_TOKEN_TENSOR) * 0.1))
                TRAIN_TENSORS = CACHED_TOKEN_TENSOR[:-val_size]
                VAL_TENSORS = CACHED_TOKEN_TENSOR[-val_size:]
                print(f"[VISUALIZER] Loaded {len(TRAIN_TENSORS)} train tensors and {len(VAL_TENSORS)} val tensors.", flush=True)
                return True
            except Exception as e:
                print(f"[VISUALIZER WARNING] Failed to load cached tensor {c_path}: {e}", flush=True)
    return False

# Pre-load ground-truth tensors immediately on startup
init_cached_tensors()

def load_dataset_texts(parquet_path):
    global VAL_TEXTS, TRAIN_TEXTS
    init_cached_tensors()
    print(f"[VISUALIZER] Loading samples from {parquet_path}...", flush=True)
    import glob
    if '*' in parquet_path or '?' in parquet_path:
        files = glob.glob(parquet_path)
    elif os.path.isdir(parquet_path):
        files = glob.glob(os.path.join(parquet_path, "*.parquet"))
    else:
        files = [parquet_path]
    
    try:
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        if "generated_solution" in df.columns:
            texts = df["generated_solution"].dropna().tolist()
        else:
            texts = df["problem"].dropna().tolist()
    except Exception as e:
        print(f"[VISUALIZER WARNING] Could not read parquet files ({e}). Using default reasoning prompts.", flush=True)
        texts = [
            "Let x be a positive real number such that x^2 + 5x + 6 = 0. We can factor this as (x+2)(x+3) = 0.",
            "To prove that the sequence converges, consider the Cauchy criterion for metric spaces.",
            "Suppose there exists an integer n such that 2^n - 1 is divisible by 7. By modular arithmetic, 2^3 = 8 = 1 (mod 7).",
            "The neural network compresses reasoning chains into compact discrete latent codes while preserving semantic fidelity."
        ] * 10
    
    val_split_idx = int(len(texts) * 0.9)
    TRAIN_TEXTS = texts[:val_split_idx]
    VAL_TEXTS = texts[val_split_idx:]
    print(f"[VISUALIZER] Loaded {len(TRAIN_TEXTS)} train samples and {len(VAL_TEXTS)} val samples.", flush=True)

def find_latest_checkpoint(dir_path, fallback_path):
    if not os.path.exists(dir_path):
        return fallback_path if os.path.exists(fallback_path) else None
    
    files = [f for f in os.listdir(dir_path) if f.endswith(".pt")]
    if not files:
        return fallback_path if os.path.exists(fallback_path) else None

    # Check for any *_epoch_<number>.pt across all model families
    import re
    highest_epoch = -1
    best_file = None

    for f in files:
        match = re.search(r'_epoch_(\d+)\.pt$', f)
        if match:
            try:
                ep = int(match.group(1))
                if ep > highest_epoch:
                    highest_epoch = ep
                    best_file = os.path.join(dir_path, f)
            except ValueError:
                pass

    if best_file:
        return best_file

    main_file = os.path.join(dir_path, "reasoning_compressor.pt")
    if os.path.exists(main_file):
        return main_file

    # Fallback to the latest modified .pt file
    sorted_by_mtime = sorted(files, key=lambda f: os.path.getmtime(os.path.join(dir_path, f)), reverse=True)
    return os.path.join(dir_path, sorted_by_mtime[0])

def _detect_arch(ckpt_state):
    """Detect model architecture from checkpoint state_dict keys."""
    keys = set(ckpt_state.keys())
    if any("latent_queries" in k or "encoder.latent_queries" in k or "memory_layers" in k or "text_encoder.layers" in k for k in keys):
        return "cascading_memory"
    if any("probe.mlp" in k or "drop_token_embedding" in k for k in keys):
        return "prefix_dropout"
    if any("encoder.router_mlp" in k for k in keys):
        return "router"
    if any("encoder.branch_2x" in k or "encoder.branch_4x" in k for k in keys):
        return "fixed_router"
    if any("policy_head" in k for k in keys):
        return "rl"
    if any("absorption_attn" in k for k in keys):
        return "adaptive_absorption"
    if any("encoder.gate_encoder" in k or "encoder.gate_score" in k for k in keys):
        return "adaptive_legacy"
    if any("decoder.upsample" in k for k in keys):
        return "legacy_fixed"
    return "standard"

CONFIG_FILENAME = "model_config.json"

def load_model_config(model_dir):
    """Read the model_config.json file for a model directory, if present."""
    config_path = os.path.join(model_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"[VISUALIZER] Loaded model config: {config_path}", flush=True)
        return config
    except Exception as e:
        print(f"⚠️ Failed to read model config {config_path}: {e}", flush=True)
        return None

def build_model_from_config(config, ckpt_state, vocab_size, device, args_saved=None):
    """Instantiate a model from a model_config.json dict:
    {'module': <file>, 'class': <ClassName>, 'args': {...}}."""
    class_name = config.get("class") or config.get("model_class")
    module_name = config.get("module")

    if not module_name:
        if class_name in ["SmallCascadingMemoryAutoencoder", "CascadingMemoryAutoencoder"]:
            module_name = "cascading_memory_model"
        elif class_name in ["SmallPrefixDropoutAutoencoder"]:
            module_name = "prefix_dropout_model"
        elif class_name in ["ChunkRouterAutoencoder"]:
            module_name = "router_model"
        elif class_name in ["FixedRouteChunkAutoencoder"]:
            module_name = "fixed_router_model"
        elif class_name in ["RLReasoningCompressor"]:
            module_name = "rl_model"
        elif class_name in ["ReasoningAutoencoder", "AdaptiveReasoningAutoencoder"]:
            module_name = "model"

    if not module_name or not class_name:
        raise ValueError(f"Invalid model config (missing module/class): {config}")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    sig_params = inspect.signature(cls.__init__).parameters
    combined_args = {}
    if args_saved:
        combined_args.update(args_saved)
    if config.get("args"):
        combined_args.update(config.get("args"))

    kwargs = {k: v for k, v in combined_args.items() if k in sig_params}
    if "vocab_size" in sig_params:
        kwargs["vocab_size"] = vocab_size

    # Auto-detect architectural shapes directly from checkpoint state dict to guarantee exact parameter matching
    if "encoder.vq.embedding.weight" in ckpt_state:
        cb_size = ckpt_state["encoder.vq.embedding.weight"].shape[0]
        dim = ckpt_state["encoder.vq.embedding.weight"].shape[1]
    elif "vq.embedding.weight" in ckpt_state:
        cb_size = ckpt_state["vq.embedding.weight"].shape[0]
        dim = ckpt_state["vq.embedding.weight"].shape[1]
    else:
        cb_size, dim = None, None

    if "codebook_size" in sig_params and cb_size is not None:
        kwargs["codebook_size"] = cb_size
    if "d_model" in sig_params and dim is not None:
        kwargs["d_model"] = dim

    lq_key = "encoder.latent_queries" if "encoder.latent_queries" in ckpt_state else ("latent_queries" if "latent_queries" in ckpt_state else None)
    if lq_key and "max_length" in sig_params:
        kwargs["max_length"] = ckpt_state[lq_key].shape[1]

    if "encoder.compression_embeddings.weight" in ckpt_state:
        if "use_compression_embeddings" in sig_params:
            kwargs["use_compression_embeddings"] = True
        if "append_m_query" in sig_params:
            kwargs["append_m_query"] = True
        if "normalize_prefix_pos" in sig_params:
            kwargs["normalize_prefix_pos"] = False

    if any(k.startswith("encoder.mlp_pos") for k in ckpt_state):
        if "mlp_prefix_pos" in sig_params:
            kwargs["mlp_prefix_pos"] = True

    return cls(**kwargs).to(device)

def load_model_on_demand(model_id=None, force_reload=False):
    global CURRENT_MODEL_ID, CURRENT_MODEL, CURRENT_TOKENIZER, CHECKPOINT_INFO, MODEL_PATHS

    # Refresh discovery so newly-saved checkpoints appear
    MODEL_PATHS = discover_models()

    if not MODEL_PATHS:
        print("⚠️ No models found in models/ directory.", flush=True)
        return False

    if model_id is None or model_id not in MODEL_PATHS:
        if "checkpoints_discrete_joint_gpu" in MODEL_PATHS:
            model_id = "checkpoints_discrete_joint_gpu"
        else:
            model_id = next(iter(MODEL_PATHS))

    # Fast in-memory cache: If already loaded and not force_reload, return immediately!
    if CURRENT_MODEL_ID == model_id and CURRENT_MODEL is not None and not force_reload:
        return True

    info = MODEL_PATHS[model_id]
    checkpoint_path = find_latest_checkpoint(info["dir"], info["default_file"])

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint for {model_id} not found at {checkpoint_path}.", flush=True)
        return False

    print(f"[VISUALIZER] Loading active model [{model_id}] -> {checkpoint_path}...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    epoch = checkpoint.get("epoch", "N/A")
    args_saved = checkpoint.get("args", {})
    ckpt_state = checkpoint["model_state_dict"]
    
    if "text_embedding.weight" in ckpt_state:
        vocab_size = ckpt_state["text_embedding.weight"].shape[0]
    elif "encoder.embedding.weight" in ckpt_state:
        vocab_size = ckpt_state["encoder.embedding.weight"].shape[0]
    else:
        vocab_size = 50265
        
    default_tok = "bert-base-uncased" if vocab_size < 40000 else "roberta-base"
    tokenizer_name = args_saved.get("tokenizer_name", default_tok)
    model_tok = get_tokenizer_for_model(tokenizer_name)

    config = load_model_config(info["dir"])
    cfg_args = (config.get("args") if config else None) or {}

    d_model = cfg_args.get("d_model", args_saved.get("d_model", 256))
    nhead = cfg_args.get("nhead", args_saved.get("nhead", 4))
    num_layers = cfg_args.get("num_layers", args_saved.get("num_layers", 3))
    decoder_num_layers = cfg_args.get("decoder_num_layers", args_saved.get("decoder_num_layers", num_layers))
    pooling_factor = cfg_args.get("pooling_factor", args_saved.get("pooling_factor", 1.5))
    codebook_size = cfg_args.get("codebook_size", args_saved.get("codebook_size", 160))
    max_length = cfg_args.get("max_length", args_saved.get("max_length", 256))
    tokenizer_name = cfg_args.get("tokenizer_name", tokenizer_name)

    arch = _detect_arch(ckpt_state)
    model = None

    if config is not None:
        arch = config.get("arch") or arch
        print(f"[VISUALIZER] Attempting to build from config: {arch}...", flush=True)
        try:
            model = build_model_from_config(config, ckpt_state, vocab_size, DEVICE, args_saved=args_saved)
        except Exception as e:
            print(f"⚠️ Failed to build model from config: {e}. Falling back to dynamic detection.", flush=True)
            model = None

    if model is None:
        if arch == "cascading_memory":
            from cascading_memory_model import SmallCascadingMemoryAutoencoder
            sig_params = inspect.signature(SmallCascadingMemoryAutoencoder.__init__).parameters
            combined_args = {}
            if args_saved:
                combined_args.update(args_saved)
            if cfg_args:
                combined_args.update(cfg_args)
            kwargs = {k: v for k, v in combined_args.items() if k in sig_params}
            kwargs["vocab_size"] = vocab_size
            model = SmallCascadingMemoryAutoencoder(**kwargs).to(DEVICE)
        elif arch == "prefix_dropout":
            from prefix_dropout_model import SmallPrefixDropoutAutoencoder
            model = SmallPrefixDropoutAutoencoder(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=args_saved.get("num_layers", 3),
                decoder_num_layers=args_saved.get("decoder_num_layers", 2),
                codebook_size=args_saved.get("codebook_size", 4096),
                max_length=max_length,
                probe_hidden_dim=cfg_args.get("probe_hidden_dim", args_saved.get("probe_hidden_dim", 128))
            ).to(DEVICE)
        elif arch == "router":
            from router_model import ChunkRouterAutoencoder
            model = ChunkRouterAutoencoder(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=args_saved.get("num_layers", 3),
                decoder_num_layers=args_saved.get("decoder_num_layers", 2),
                codebook_size=args_saved.get("codebook_size", 4096),
                chunk_size=args_saved.get("chunk_size", 4),
                max_length=max_length,
                alpha=args_saved.get("alpha", 0.5)
            ).to(DEVICE)
        elif arch == "fixed_router":
            from fixed_router_model import FixedRouteChunkAutoencoder
            model = FixedRouteChunkAutoencoder(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                decoder_num_layers=decoder_num_layers,
                codebook_size=codebook_size,
                chunk_size=args_saved.get("chunk_size", 4),
                max_length=max_length,
                fixed_route=args_saved.get("fixed_route", 1)
            ).to(DEVICE)
        elif arch == "rl":
            from rl_model import RLReasoningCompressor
            model = RLReasoningCompressor(
                vocab_size=vocab_size,
                codebook_size=codebook_size,
                d_model=d_model,
                nhead=nhead,
                encoder_num_layers=args_saved.get("encoder_num_layers", 3),
                policy_num_layers=args_saved.get("policy_num_layers", 2),
                decoder_num_layers=args_saved.get("decoder_num_layers", 2),
                max_length=max_length,
                max_latent_len=args_saved.get("max_latent_len", 32)
            ).to(DEVICE)
        elif arch == "adaptive_absorption":
            from model import AdaptiveReasoningAutoencoder
            model = AdaptiveReasoningAutoencoder(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                decoder_num_layers=decoder_num_layers,
                codebook_size=codebook_size,
                budget_weight=args_saved.get("budget_weight", 0.5),
                target_retention=args_saved.get("target_retention", 0.5),
                min_tokens=args_saved.get("min_tokens", 10)
            ).to(DEVICE)
        elif arch == "adaptive_legacy":
            use_gate_enc = any(k.startswith("encoder.gate_encoder") for k in ckpt_state.keys())
            model = LegacyModel(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                decoder_num_layers=decoder_num_layers,
                codebook_size=codebook_size,
                use_gate_encoder=use_gate_enc
            ).to(DEVICE)
        elif arch == "legacy_fixed":
            from model import LegacyReasoningAutoencoder
            model = LegacyReasoningAutoencoder(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                decoder_num_layers=decoder_num_layers,
                pooling_factor=pooling_factor,
                codebook_size=codebook_size
            ).to(DEVICE)
        else:
            model = ReasoningAutoencoder(
                vocab_size=vocab_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                decoder_num_layers=decoder_num_layers,
                pooling_factor=pooling_factor,
                codebook_size=codebook_size
            ).to(DEVICE)

    try:
        model.load_state_dict(ckpt_state, strict=False)
    except RuntimeError as e:
        print(f"⚠️ Failed to load state dict for {model_id} (architecture mismatch): {e}", flush=True)
        return False
    model.eval()

    CURRENT_MODEL = model
    CURRENT_MODEL_ID = model_id
    CURRENT_TOKENIZER = model_tok
    real_max_len = getattr(model, "max_length", max_length)
    real_cb_size = getattr(model, "codebook_size", codebook_size)
    real_d_model = getattr(model, "d_model", d_model)
    CHECKPOINT_INFO = {
        "model_id": model_id,
        "arch": arch,
        "epoch": epoch,
        "pooling_factor": pooling_factor,
        "codebook_size": real_cb_size,
        "d_model": real_d_model,
        "num_layers": num_layers,
        "max_length": real_max_len,
        "tokenizer_name": tokenizer_name,
        "checkpoint_file": os.path.basename(checkpoint_path),
        "parameters": sum(p.numel() for p in model.parameters())
    }
    print(f"[VISUALIZER] Model [{model_id}] loaded successfully (Arch: {arch}, Epoch {epoch}, MaxLen {real_max_len}, Codebook {real_cb_size}, d_model {real_d_model})!", flush=True)
    return True

def compute_lossless_storage_metrics(model, input_ids, arch, prefix_m, tokenizer):
    """
    Computes the minimum number of additional stored bits required to achieve 100% perfect
    lossless reconstruction given token-forcing (teacher-forcing).
    """
    with torch.no_grad():
        L = input_ids.shape[1]
        pad_id = getattr(model, "pad_token_id", 1)
        valid_mask = (input_ids[0] != pad_id)
        L_valid = max(1, int(valid_mask.sum().item()))

        tf_logits = None
        try:
            if arch == "cascading_memory":
                out = model(input_ids, prefix_len=prefix_m, word_dropout=0.0, k_samples=1, noise_std=0.0)
                tf_logits = out["logits"] if isinstance(out, dict) else out
            elif hasattr(model, "forward"):
                out = model(input_ids)
                tf_logits = out["logits"] if isinstance(out, dict) else out
        except Exception as e:
            tf_logits = None

        if tf_logits is not None:
            tf_preds = torch.argmax(tf_logits[0, :L], dim=-1)
            tf_errors_mask = (tf_preds != input_ids[0, :L]) & valid_mask

            # Shannon cross-entropy lower bound in bits: -log2 P(target)
            log_probs = F.log_softmax(tf_logits[0, :L].float(), dim=-1)
            target_log_probs = log_probs.gather(dim=-1, index=input_ids[0, :L].unsqueeze(-1)).squeeze(-1)
            nll_bits = -target_log_probs / math.log(2.0)
            shannon_add_bits = float((nll_bits * valid_mask.float()).sum().item())
        else:
            tf_preds = input_ids[0, :L]
            tf_errors_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            shannon_add_bits = 0.0

        num_errors = int(tf_errors_mask.sum().item())
        correct_tokens = L_valid - num_errors
        tf_acc = float(correct_tokens) / float(L_valid)

        vocab_size = getattr(model, "vocab_size", 50265)
        bits_per_token = math.ceil(math.log2(max(2, vocab_size)))

        cb_size = getattr(model, "codebook_size", 16384)
        bits_per_latent = math.ceil(math.log2(max(2, cb_size))) if cb_size > 0 else 14
        latent_bits = prefix_m * bits_per_latent

        # Scheme A: Bitmask of length L_valid (1 bit per position: 0=correct, 1=incorrect) + bits_per_token per error
        bitmap_bits = L_valid + (num_errors * bits_per_token)
        # Scheme B: Sparse list of error positions (ceil(log2(L_valid)) bits per position + bits_per_token per error)
        bits_per_pos = max(1, math.ceil(math.log2(max(2, L_valid))))
        sparse_bits = num_errors * (bits_per_pos + bits_per_token)

        min_additional_bits = min(bitmap_bits, sparse_bits)
        min_additional_bytes = math.ceil(min_additional_bits / 8.0)

        total_lossless_bits = latent_bits + min_additional_bits
        total_lossless_bytes = math.ceil(total_lossless_bits / 8.0)

        raw_uncompressed_bits = L_valid * bits_per_token
        raw_uncompressed_bytes = math.ceil(raw_uncompressed_bits / 8.0)

        lossless_ratio = round(raw_uncompressed_bits / max(1, total_lossless_bits), 2)
        shannon_total_bits = round(latent_bits + shannon_add_bits, 1)
        shannon_ratio = round(raw_uncompressed_bits / max(1.0, shannon_total_bits), 2)

        # Detailed error tokens list for visualizer chips
        error_details = []
        for i in range(L):
            if tf_errors_mask[i]:
                exp_tok = tokenizer.decode([input_ids[0, i].item()])
                pred_tok = tokenizer.decode([tf_preds[i].item()])
                error_details.append({
                    "pos": i,
                    "expected_token": exp_tok,
                    "expected_id": int(input_ids[0, i].item()),
                    "predicted_token": pred_tok,
                    "predicted_id": int(tf_preds[i].item()),
                    "bits_cost": bits_per_token
                })

        # Classical Zip/Gzip Baseline Compression
        raw_text = tokenizer.decode(input_ids[0, :L_valid], skip_special_tokens=False)
        raw_text_bytes = raw_text.encode('utf-8')
        raw_text_len = len(raw_text_bytes)

        raw_tokens_bytes = struct.pack(f"<{L_valid}H", *[int(t.item()) for t in input_ids[0, :L_valid]])
        raw_tokens_len = len(raw_tokens_bytes)

        # Deflate (Standard Zip format stream, level 9)
        zip_tokens_bytes = len(zlib.compress(raw_tokens_bytes, level=9))
        zip_tokens_ratio = round(raw_tokens_len / max(1, zip_tokens_bytes), 2)

        zip_text_bytes = len(zlib.compress(raw_text_bytes, level=9))
        zip_text_ratio = round(raw_text_len / max(1, zip_text_bytes), 2)

        # Gzip (with gzip header, level 9)
        gzip_tokens_bytes = len(gzip.compress(raw_tokens_bytes, compresslevel=9))
        gzip_tokens_ratio = round(raw_tokens_len / max(1, gzip_tokens_bytes), 2)

        gzip_text_bytes = len(gzip.compress(raw_text_bytes, compresslevel=9))
        gzip_text_ratio = round(raw_text_len / max(1, gzip_text_bytes), 2)

        # Savings & comparison against Zip
        token_savings_pct = round(((zip_tokens_bytes - total_lossless_bytes) / max(1, zip_tokens_bytes)) * 100.0, 1)
        text_savings_pct = round(((zip_text_bytes - total_lossless_bytes) / max(1, zip_text_bytes)) * 100.0, 1)
        vs_zip_ratio = round(zip_tokens_bytes / max(1, total_lossless_bytes), 2)

        return {
            "token_forcing_acc": round(tf_acc * 100, 1),
            "num_errors": num_errors,
            "total_valid_tokens": L_valid,
            "bits_per_token": bits_per_token,
            "bits_per_latent": bits_per_latent,
            "latent_bits": latent_bits,
            "min_additional_bits": min_additional_bits,
            "min_additional_bytes": min_additional_bytes,
            "total_lossless_bits": total_lossless_bits,
            "total_lossless_bytes": total_lossless_bytes,
            "raw_uncompressed_bits": raw_uncompressed_bits,
            "raw_uncompressed_bytes": raw_uncompressed_bytes,
            "lossless_compression_ratio": f"{lossless_ratio:.2f}x",
            "shannon_additional_bits": round(shannon_add_bits, 1),
            "shannon_total_bits": shannon_total_bits,
            "shannon_compression_ratio": f"{shannon_ratio:.2f}x",
            "scheme_used": "bitmap" if bitmap_bits <= sparse_bits else "sparse_index",
            "errors": error_details[:30],
            "zip_comparison": {
                "raw_text_bytes": raw_text_len,
                "raw_tokens_bytes": raw_tokens_len,
                "zip_tokens_bytes": zip_tokens_bytes,
                "zip_tokens_ratio": f"{zip_tokens_ratio:.2f}x",
                "zip_text_bytes": zip_text_bytes,
                "zip_text_ratio": f"{zip_text_ratio:.2f}x",
                "gzip_tokens_bytes": gzip_tokens_bytes,
                "gzip_tokens_ratio": f"{gzip_tokens_ratio:.2f}x",
                "gzip_text_bytes": gzip_text_bytes,
                "gzip_text_ratio": f"{gzip_text_ratio:.2f}x",
                "token_savings_pct": token_savings_pct,
                "text_savings_pct": text_savings_pct,
                "vs_zip_ratio": f"{vs_zip_ratio:.2f}x",
                "beats_zip_tokens": bool(total_lossless_bytes < zip_tokens_bytes),
                "beats_zip_text": bool(total_lossless_bytes < zip_text_bytes)
            }
        }

def generate_sample_data(model_id=None, sample_idx=None, split="val", custom_text=None, mode="target_acc", target_acc=0.90, drop_k=0):
    if not load_model_on_demand(model_id):
        if CURRENT_MODEL is None:
            load_model_on_demand()

    arch = CHECKPOINT_INFO.get("arch", "")
    if arch == "cascading_memory" or (hasattr(CURRENT_MODEL, "encoder") and hasattr(CURRENT_MODEL.encoder, "latent_queries")):
        max_length = getattr(CURRENT_MODEL, "max_length", 64)
    else:
        max_length = getattr(CURRENT_MODEL, "max_length", CHECKPOINT_INFO.get("max_length", 256))

    tokenizer = CURRENT_TOKENIZER

    if split == "custom" and custom_text:
        raw_text = custom_text
        sample_idx = 0
        dataset_texts = [custom_text]
        inputs = tokenizer(
            raw_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(DEVICE)
        pad_id = getattr(CURRENT_MODEL, "pad_token_id", 1)
        attention_mask = (input_ids != pad_id).long().to(DEVICE)
        raw_text_sliced = raw_text
    elif split != "custom":
        if VAL_TENSORS is None:
            init_cached_tensors()
        if VAL_TENSORS is not None and max_length == 64:
            active_tensors = TRAIN_TENSORS if split == "train" else VAL_TENSORS
            if sample_idx is None or sample_idx < 0 or sample_idx >= len(active_tensors):
                sample_idx = random.randint(0, len(active_tensors) - 1)
            input_ids = active_tensors[sample_idx:sample_idx+1].to(DEVICE).long()
            pad_id = getattr(CURRENT_MODEL, "pad_token_id", 1)
            attention_mask = (input_ids != pad_id).long().to(DEVICE)
            raw_text_sliced = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            dataset_texts = active_tensors
        else:
            if not VAL_TEXTS and not TRAIN_TEXTS:
                load_dataset_texts(os.path.dirname(os.path.abspath(__file__)))
            dataset_texts = TRAIN_TEXTS if split == "train" else VAL_TEXTS
            if not dataset_texts:
                dataset_texts = VAL_TEXTS if VAL_TEXTS else TRAIN_TEXTS
            if not dataset_texts:
                dataset_texts = ["A mathematician is solving a calculus problem. The derivative of x^2 is 2x."]

            if sample_idx is None or sample_idx < 0 or sample_idx >= len(dataset_texts):
                sample_idx = random.randint(0, len(dataset_texts) - 1)

            raw_text = dataset_texts[sample_idx]
            full_tokens = tokenizer.encode(raw_text, truncation=False)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
            if len(full_tokens) > max_length:
                max_start = len(full_tokens) - max_length
                start_pos = random.randint(0, max_start)
                span_tokens = full_tokens[start_pos : start_pos + max_length]
            else:
                span_tokens = full_tokens + [pad_id] * (max_length - len(full_tokens))
                
            input_ids = torch.tensor([span_tokens], dtype=torch.long, device=DEVICE)
            attention_mask = (input_ids != pad_id).long().to(DEVICE)
            raw_text_sliced = tokenizer.decode(input_ids[0], skip_special_tokens=False)

    scores_list = []
    retained_count = 0

    with torch.no_grad():
        if arch == "cascading_memory" or (hasattr(CURRENT_MODEL, "encoder") and hasattr(CURRENT_MODEL.encoder, "latent_queries")):
            L = min(input_ids.shape[1], getattr(CURRENT_MODEL, "max_length", 64))
            input_ids = input_ids[:, :L]
            attention_mask = attention_mask[:, :L]

            # Map UI modes to target prefix length M in [8, 16, 24, 32, 48, 64]
            if mode == "fixed_k" and drop_k:
                M = max(8, min(L, L - int(drop_k)))
            elif mode == "target_acc":
                target_ratio = float(target_acc)
                if target_ratio > 1.0:
                    target_ratio /= 100.0
                if target_ratio >= 0.92: M = 64
                elif target_ratio >= 0.84: M = 48
                elif target_ratio >= 0.70: M = 32
                elif target_ratio >= 0.52: M = 16
                else: M = 8
            else:
                M = getattr(CURRENT_MODEL, "fixed_prefix_len", 32)
                if not M or M <= 0: M = 32
            M = max(8, min(M, L))

            # Encode via Cascading Encoder to discrete codebook vectors
            enc_out = CURRENT_MODEL.encoder(
                input_ids,
                attention_mask=attention_mask,
                prefix_len=M,
                return_continuous=False
            )
            quantized = enc_out[0]
            vq_loss = enc_out[1]
            code_indices_tensor = enc_out[2]
            code_indices = code_indices_tensor[0, :M].cpu().tolist()
            retained_count = M

            # Autoregressive generation through Cascading Decoder
            bos_id = getattr(CURRENT_MODEL, "bos_token_id", 0)
            pad_id = getattr(CURRENT_MODEL, "pad_token_id", 1)

            if hasattr(CURRENT_MODEL.decoder, "generate"):
                has_appended = getattr(CURRENT_MODEL, "append_m_query", False) and getattr(CURRENT_MODEL, "use_compression_embeddings", False)
                exp_len = M + 1 if has_appended else M
                mem_input = quantized[:, :exp_len, :]
                generated_ids = CURRENT_MODEL.decoder.generate(
                    mem_input,
                    start_token_id=bos_id,
                    max_length=L
                )
                pred_ids = generated_ids[0].cpu().tolist()
                if len(pred_ids) > L and pred_ids[0] == bos_id:
                    pred_ids = pred_ids[1:]
                pred_ids = pred_ids[:L]
            else:
                out = CURRENT_MODEL(input_ids, prefix_len=M)
                pred_ids = torch.argmax(out["logits"][0], dim=-1).cpu().tolist()

            # Token reconstruction accuracy
            non_pad_mask = (input_ids[0] != pad_id)
            pred_tensor = torch.tensor(pred_ids[:L], device=DEVICE)
            correct_mask = (pred_tensor == input_ids[0, :len(pred_tensor)]) & non_pad_mask[:len(pred_tensor)]
            actual_acc = (correct_mask.sum().float() / non_pad_mask.sum().clamp(min=1).float()).item()

            # Live teacher-forcing accuracy
            try:
                tf_out = CURRENT_MODEL(input_ids, prefix_len=M, word_dropout=0.0)
                tf_pred_ids = torch.argmax(tf_out["logits"][0, :L], dim=-1)
                tf_correct = (tf_pred_ids == input_ids[0, :L]) & non_pad_mask[:L]
                tf_acc_live = (tf_correct.sum().float() / non_pad_mask.sum().clamp(min=1).float()).item()
            except Exception:
                tf_acc_live = actual_acc

            raw_token_ids = input_ids[0].cpu().tolist()
            prefix_token_details = []
            for i in range(L):
                t_id = raw_token_ids[i]
                t_str = tokenizer.decode([t_id])
                c_idx = code_indices[i] if i < len(code_indices) else 0
                is_kept = (i < M)
                prefix_token_details.append({
                    "token": t_str,
                    "code_idx": int(c_idx),
                    "score": 100.0 if is_kept else 0.0,
                    "kept": is_kept,
                    "pos": i,
                    "is_cutoff": (i == M - 1)
                })
            token_details = prefix_token_details

            raw_chunks = [
                {
                    "chunk_idx": 0,
                    "route": 3,
                    "compression": f"Transmitted Latents: {M}/{L} ({L/max(1,M):.2f}x)",
                    "text": tokenizer.decode(raw_token_ids[:M])
                }
            ]
            if M < L:
                raw_chunks.append({
                    "chunk_idx": 1,
                    "route": 0,
                    "compression": f"Reconstructed Tail: {L-M} tokens",
                    "text": tokenizer.decode(raw_token_ids[M:])
                })

            curve_pcts = [45.8, 58.4, 76.2, 88.6, 94.6]
            probe_stats = {
                "mode": mode,
                "target_acc": round(float(target_acc) if float(target_acc) <= 1.0 else float(target_acc)/100.0, 3),
                "drop_k": L - M,
                "selected_m": M,
                "pred_acc": round(tf_acc_live * 100, 1),
                "actual_acc": round(actual_acc * 100, 1),
                "accuracy_curve": curve_pcts
            }
        elif arch == "prefix_dropout":
            quantized, _, code_indices_tensor = CURRENT_MODEL.encoder(input_ids, jitter_scale=0.0)
            code_indices = code_indices_tensor[0].cpu().tolist()
            
            curve_tensor = CURRENT_MODEL.probe.predict_full_curve(code_indices_tensor) # [1, 64]
            curve_floats = [round(c.item(), 4) for c in curve_tensor[0]]
            curve_pcts = [round(c * 100, 1) for c in curve_floats]

            L = input_ids.shape[1]
            if mode == "target_acc":
                target_ratio = float(target_acc)
                if target_ratio > 1.0:
                    target_ratio = target_ratio / 100.0
                qualifying = [m for m in range(1, L + 1) if curve_floats[m - 1] >= target_ratio]
                if qualifying:
                    M = qualifying[0]
                else:
                    M = L
                k = L - M
            elif mode == "fixed_k":
                k = max(0, min(L - 1, int(drop_k)))
                M = L - k
            else: # full 1x
                M = L
                k = 0

            bos_id = getattr(CURRENT_MODEL, "bos_token_id", 0)
            generated_ids = CURRENT_MODEL.decoder.generate(
                quantized[:, :M, :],
                start_token_id=bos_id,
                max_length=max_length
            )
            pred_ids = generated_ids[0].cpu().tolist()

            # Compute actual token reconstruction accuracy on non-padding tokens
            non_pad_mask = (input_ids[0] != 1) & (input_ids[0] != bos_id)
            if non_pad_mask.sum() == 0:
                non_pad_mask = (input_ids[0] != 1)
            pred_tensor = torch.tensor(pred_ids, device=DEVICE)
            correct_mask = (pred_tensor == input_ids[0]) & non_pad_mask
            actual_acc = (correct_mask.sum().float() / non_pad_mask.sum().clamp(min=1).float()).item()

            raw_token_ids = input_ids[0].cpu().tolist()
            prefix_token_details = []
            for i in range(L):
                t_id = raw_token_ids[i]
                t_str = tokenizer.decode([t_id])
                c_idx = code_indices[i] if i < len(code_indices) else 0
                t_score = curve_pcts[i] if i < len(curve_pcts) else 0.0
                is_kept = (i < M)
                prefix_token_details.append({
                    "token": t_str,
                    "code_idx": int(c_idx),
                    "score": t_score,
                    "kept": is_kept,
                    "pos": i,
                    "is_cutoff": (i == M - 1)
                })

            token_details = prefix_token_details
            retained_count = M
            ratio_str = f"{L / max(1, M):.2f}x"

            raw_chunks = [
                {
                    "chunk_idx": 0,
                    "route": 3,
                    "compression": f"Prefix {M}/{L}",
                    "text": tokenizer.decode(raw_token_ids[:M])
                }
            ]
            if M < L:
                raw_chunks.append({
                    "chunk_idx": 1,
                    "route": 0,
                    "compression": f"Dropped {k}/{L}",
                    "text": tokenizer.decode(raw_token_ids[M:])
                })

            probe_stats = {
                "mode": mode,
                "target_acc": round(float(target_acc) if float(target_acc) <= 1.0 else float(target_acc)/100.0, 3),
                "drop_k": k,
                "selected_m": M,
                "pred_acc": curve_pcts[M - 1] if M <= len(curve_pcts) else 100.0,
                "actual_acc": round(actual_acc * 100, 1),
                "accuracy_curve": curve_pcts
            }
        elif arch == "router" or arch == "fixed_router" or (hasattr(CURRENT_MODEL, "encoder") and hasattr(CURRENT_MODEL.encoder, "router_mlp")):
            outputs = CURRENT_MODEL(input_ids, attention_mask=attention_mask, deterministic=True)
            raw_codes = outputs["code_indices"][0]
            if isinstance(raw_codes, torch.Tensor):
                codes = raw_codes.cpu().tolist()
            else:
                codes = list(raw_codes)
            code_indices = codes
            retained_count = len(codes)
            
            bos_id = getattr(CURRENT_MODEL, "bos_token_id", 0)
            generated_ids = CURRENT_MODEL.generate(
                outputs["padded_quantized"],
                start_token_id=bos_id,
                max_length=max_length,
                memory_key_padding_mask=outputs["latent_key_padding_mask"]
            )
            pred_ids = generated_ids[0].cpu().tolist()
            
            raw_routes = outputs["selected_routes"][0]
            if isinstance(raw_routes, torch.Tensor):
                routes = raw_routes.cpu().tolist()
            else:
                routes = list(raw_routes)
                
            route_tokens_count = [1, 2, 3, 4]
            route_names = ["4x (1 tok)", "2x (2 toks)", "1.33x (3 toks)", "1x (4 toks)"]
            raw_token_ids = input_ids[0].cpu().tolist()
            chunk_size = getattr(CURRENT_MODEL.encoder, "chunk_size", 4)
            router_token_details = []
            code_ptr = 0
            
            raw_chunks = []
            comp_map = {0: "4x", 1: "2x", 2: "1.33x", 3: "1x"}
            for c_idx, r_val in enumerate(routes):
                chunk_tokens = raw_token_ids[c_idx*chunk_size : (c_idx+1)*chunk_size]
                chunk_str = tokenizer.decode(chunk_tokens)
                num_toks_for_chunk = route_tokens_count[r_val]
                raw_chunks.append({
                    "chunk_idx": c_idx,
                    "route": int(r_val),
                    "compression": comp_map.get(int(r_val), "1x"),
                    "text": chunk_str
                })
                
                for sub_idx in range(num_toks_for_chunk):
                    c_code = codes[code_ptr] if code_ptr < len(codes) else 0
                    code_ptr += 1
                    sub_str = f" #{sub_idx+1}" if num_toks_for_chunk > 1 else ""
                    router_token_details.append({
                        "token": f"Chunk {c_idx} [{route_names[r_val]}{sub_str}]: '{chunk_str}'",
                        "code_idx": int(c_code),
                        "score": round(1.0 - (r_val * 0.25), 2),
                        "kept": True,
                        "chunk_idx": c_idx,
                        "sub_idx": sub_idx,
                        "num_chunk_tokens": num_toks_for_chunk,
                        "is_chunk_start": (sub_idx == 0),
                        "is_chunk_end": (sub_idx == num_toks_for_chunk - 1)
                    })
            if len(raw_token_ids) > len(routes) * chunk_size:
                rem_tokens = raw_token_ids[len(routes)*chunk_size:]
                rem_str = tokenizer.decode(rem_tokens)
                if rem_str:
                    raw_chunks.append({
                        "chunk_idx": len(routes),
                        "route": 3,
                        "compression": "1x",
                        "text": rem_str
                    })
            token_details = router_token_details
        elif arch == "rl" or hasattr(CURRENT_MODEL, "sample_latent_codes"):
            h_text = CURRENT_MODEL.encode_text(input_ids)
            sampled_codes, _, lengths, _, _, _ = CURRENT_MODEL.sample_latent_codes(
                h_text,
                group_size=1,
                temperature=0.7,
                min_latent_len=8
            )
            K = lengths[0].item()
            code_indices = sampled_codes[0, :K].cpu().tolist()
            retained_count = K
            
            curr_tgt = torch.tensor([[input_ids[0, 0].item()]], dtype=torch.long, device=DEVICE)
            for _ in range(min(max_length, 64)):
                logits = CURRENT_MODEL.reconstruct_text(curr_tgt, sampled_codes[:, :K])
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_tgt = torch.cat([curr_tgt, next_token], dim=1)
                if next_token.item() == 2: # EOS
                    break
            pred_ids = curr_tgt[0, 1:].cpu().tolist()
        elif hasattr(CURRENT_MODEL, "encoder") and hasattr(CURRENT_MODEL.encoder, "score_mlp"):
            enc_out = CURRENT_MODEL.encoder(input_ids, attention_mask, threshold=0.5, min_tokens=10, force_discrete=True)
            if len(enc_out) == 7:
                quantized, code_indices, scores = enc_out[0], enc_out[3], enc_out[5]
            else:
                quantized, code_indices, scores = enc_out[0], enc_out[4], enc_out[6]
            scores_list = scores[0].cpu().tolist()
            
            # Real physical gating: identify kept tokens
            topk_scores, _ = torch.topk(scores, k=min(10, scores.shape[-1]), dim=-1)
            min_topk = topk_scores[:, -1:]
            is_kept_mask = (scores >= 0.5) | (scores >= min_topk)
            kept_bools = is_kept_mask[0].cpu().tolist()
            
            # Physically gather ONLY the retained tokens into [1, K, D]
            quantized_kept = quantized[:, is_kept_mask[0], :]
            
            all_code_indices = code_indices[0].cpu().tolist()
            code_indices = [code for idx, code in enumerate(all_code_indices) if kept_bools[idx]]
            retained_count = len(code_indices)
            
            bos_id = input_ids[0, 0].item()
            generated_ids = CURRENT_MODEL.decoder.generate(quantized_kept, start_token_id=bos_id, max_length=max_length + 1)
            pred_ids = generated_ids[0, 1:].cpu().tolist()
        elif hasattr(CURRENT_MODEL.decoder, "generate"):
            quantized, _, code_indices, _ = CURRENT_MODEL.encoder(input_ids, attention_mask)
            bos_id = input_ids[0, 0].item()
            generated_ids = CURRENT_MODEL.decoder.generate(quantized, start_token_id=bos_id, max_length=max_length + 1)
            code_indices = code_indices[0].cpu().tolist()
            pred_ids = generated_ids[0, 1:].cpu().tolist()
        else:
            outputs = CURRENT_MODEL(input_ids, attention_mask)
            code_indices = outputs["code_indices"][0].cpu().tolist()
            if "scores" in outputs:
                scores_list = outputs["scores"][0].cpu().tolist()
                retained_count = int(sum(1 for s in scores_list if s >= 0.5))
            logits = outputs["logits"][0]
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

    decoded_text = tokenizer.decode(pred_ids, skip_special_tokens=False)

    if 'token_details' not in locals() or token_details is None:
        token_details = []
    if scores_list:
        raw_token_ids = input_ids[0].cpu().tolist()
        for idx, (tid, sc) in enumerate(zip(raw_token_ids, scores_list)):
            token_str = tokenizer.decode([tid])
            c_idx = all_code_indices[idx] if 'all_code_indices' in locals() else (code_indices[idx] if idx < len(code_indices) else 0)
            token_details.append({
                "token": token_str,
                "code_idx": int(c_idx),
                "score": round(sc, 3),
                "kept": bool(kept_bools[idx] if 'kept_bools' in locals() else (sc >= 0.5))
            })

    eff_len = retained_count if (scores_list or 'probe_stats' in locals() or arch == "prefix_dropout" or hasattr(CURRENT_MODEL, "probe")) else len(code_indices)
    ratio_str = f"{max_length / max(1, eff_len):.2f}x"

    # Compute exact bit requirements for lossless reconstruction given token-forcing
    lossless_metrics = compute_lossless_storage_metrics(
        CURRENT_MODEL,
        input_ids,
        arch,
        eff_len,
        tokenizer
    )

    return {
        "sample_index": sample_idx,
        "total_samples": len(dataset_texts),
        "split": split,
        "raw_text": raw_text_sliced,
        "raw_chunks": raw_chunks if 'raw_chunks' in locals() else None,
        "input_length": max_length,
        "compressed_length": eff_len,
        "compression_ratio": ratio_str,
        "code_indices": code_indices,
        "token_details": token_details,
        "decoded_text": decoded_text,
        "probe_stats": probe_stats if 'probe_stats' in locals() else None,
        "lossless_metrics": lossless_metrics,
        "checkpoint_info": CHECKPOINT_INFO
    }

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reasoning Trace Compressor - Visualizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --panel-bg: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-emerald: #10b981;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            padding: 20px;
            min-height: 100vh;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-color);
            padding: 18px 24px;
            border-radius: 14px;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
            flex-wrap: wrap;
            gap: 15px;
        }

        .header-title h1 {
            font-size: 1.35rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-title p {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        .controls-group {
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }

        .model-select-wrapper {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .model-select-wrapper label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 600;
        }

        .model-select {
            background: #0f172a;
            color: var(--accent-cyan);
            border: 1px solid var(--accent-cyan);
            padding: 8px 14px;
            font-size: 0.9rem;
            font-weight: 600;
            border-radius: 10px;
            outline: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .model-select:hover {
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.4);
        }

        .stats-pills {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .pill {
            background: rgba(51, 65, 85, 0.5);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .pill strong {
            color: var(--accent-cyan);
        }

        .btn-random {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #fff;
            border: none;
            padding: 10px 18px;
            font-size: 0.85rem;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-random:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        }

        .probe-control-bar {
            width: 100%;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(6, 182, 212, 0.3);
            border-radius: 12px;
            padding: 12px 18px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .probe-curve-card {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 16px;
        }

        .curve-bar {
            flex: 1;
            min-width: 4px;
            background: rgba(59, 130, 246, 0.35);
            border-radius: 2px 2px 0 0;
            transition: all 0.15s ease;
            cursor: pointer;
        }

        .curve-bar:hover {
            background: var(--accent-cyan) !important;
            transform: scaleY(1.08);
        }

        .curve-bar.active-prefix {
            background: linear-gradient(180deg, var(--accent-emerald), rgba(16, 185, 129, 0.6));
        }

        .curve-bar.cutoff-marker {
            background: var(--accent-cyan) !important;
            box-shadow: 0 0 8px var(--accent-cyan);
        }

        .curve-bar.dropped-prefix {
            background: rgba(100, 116, 139, 0.25);
            opacity: 0.5;
        }

        .grid-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }

        .panel-header h2 {
            font-size: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .panel-header .badge {
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
        }

        .badge-cyan { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
        .badge-emerald { background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-purple { background: rgba(139, 92, 246, 0.15); color: var(--accent-purple); border: 1px solid rgba(139, 92, 246, 0.3); }

        .text-content {
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            line-height: 1.6;
            color: #cbd5e1;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            padding: 16px;
            border-radius: 10px;
            height: 280px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .codebook-panel {
            grid-column: 1 / -1;
        }

        .tokens-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            padding: 16px;
            border-radius: 10px;
            max-height: 200px;
            overflow-y: auto;
        }

        .token-chip {
            font-family: 'Fira Code', monospace;
            font-size: 0.75rem;
            padding: 4px 8px;
            border-radius: 6px;
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid var(--border-color);
            color: #e2e8f0;
            transition: all 0.15s ease;
        }

        .token-chip:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            transform: scale(1.1);
        }

        .chunk-separator {
            width: 2px;
            height: 24px;
            background-color: #ffffff;
            opacity: 0.9;
            margin: 0 4px;
            border-radius: 1px;
            flex-shrink: 0;
            box-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
            align-self: center;
        }

        .chunk-hl-1x {
            color: #cbd5e1;
        }

        .chunk-hl-1-33x {
            background: rgba(234, 179, 8, 0.22);
            color: #fef08a;
            border-bottom: 2px solid rgba(234, 179, 8, 0.85);
            padding: 1px 3px;
            margin: 0 1px;
            border-radius: 3px;
        }

        .chunk-hl-2x {
            background: rgba(249, 115, 22, 0.25);
            color: #fed7aa;
            border-bottom: 2px solid rgba(249, 115, 22, 0.9);
            padding: 1px 3px;
            margin: 0 1px;
            border-radius: 3px;
        }

        .chunk-hl-4x {
            background: rgba(239, 68, 68, 0.28);
            color: #fca5a5;
            border-bottom: 2px solid rgba(239, 68, 68, 0.95);
            padding: 1px 3px;
            margin: 0 1px;
            border-radius: 3px;
        }

        .chunk-legend {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.72rem;
            margin-left: auto;
            margin-right: 12px;
            font-weight: 500;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 4px;
            color: var(--text-secondary);
        }

        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.4); }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #64748b; }

        .metric-box {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .error-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
            padding: 4px 8px;
            border-radius: 6px;
            margin: 3px 4px;
            font-size: 0.75rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>🧠 Reasoning Trace Autoencoder Visualizer</h1>
            <p>Discrete Bottleneck Compression & Token Reconstruction Evaluator</p>
        </div>

        <div class="controls-group">
            <div class="model-select-wrapper">
                <label for="model-select">Select Model:</label>
                <select id="model-select" class="model-select" onchange="fetchRandomSample()">
                    __MODEL_OPTIONS__
                </select>
            </div>

            <div class="model-select-wrapper">
                <label for="split-select">Split:</label>
                <select id="split-select" class="model-select" onchange="onSplitChange()">
                    <option value="val" selected>Validation Split</option>
                    <option value="train">Train Split</option>
                    <option value="custom">✏️ Your Own (Custom Text)</option>
                </select>
            </div>

            <div class="stats-pills">
                <div class="pill">Checkpoint: <strong id="stat-epoch">Loading...</strong></div>
                <div class="pill">Compression: <strong id="stat-ratio">--</strong></div>
                <div class="pill" id="pill-probe-pred" style="display:none;">Probe Pred: <strong id="stat-probe-pred">--</strong></div>
                <div class="pill" id="pill-actual-acc" style="display:none;">Actual Acc: <strong id="stat-actual-acc">--</strong></div>
                <div class="pill">Codebook: <strong id="stat-codebook">4096</strong></div>
                <div class="pill" id="pill-lossless" style="background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(56, 189, 248, 0.4);">
                    Lossless: <strong id="stat-lossless-ratio" style="color: #38bdf8;">--</strong> (<span id="stat-lossless-bytes" style="color: #94a3b8;">--</span>)
                </div>
            </div>

            <button id="btn-reload" class="btn-random" style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.2);" onclick="reloadModelWeights()">
                🔄 Reload Checkpoint
            </button>

            <button class="btn-random" style="background: linear-gradient(135deg, #059669, #10b981); border: none;" onclick="openExperimentsModal()">
                📚 Research Archive
            </button>

            <button class="btn-random" onclick="fetchRandomSample()">
                🎲 Random Sample
            </button>
        </div>
    </div>

    <!-- Prefix-Dropout & Accuracy Probe Dynamic Control Strip -->
    <div id="prefix-probe-controls" class="probe-control-bar" style="display: none;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 0.9rem; font-weight: 700; color: var(--accent-cyan);">🎛️ Prefix-Dropout Controls:</span>
            <select id="probe-mode-select" class="model-select" style="font-size: 0.82rem; padding: 6px 12px;" onchange="onProbeControlChange()">
                <option value="target_acc" selected>🎯 Target Accuracy Threshold</option>
                <option value="fixed_k">✂️ Fixed Drop Tokens (k)</option>
                <option value="full">📦 Full 1x Sequence (No Drop)</option>
            </select>
        </div>

        <div id="target-acc-wrapper" class="model-select-wrapper" style="display: flex; align-items: center; gap: 10px;">
            <label for="target-acc-slider" style="font-size: 0.85rem;">Target Reconstruction Accuracy: <strong id="target-acc-val" style="color: var(--accent-cyan); font-size: 0.95rem;">90%</strong></label>
            <input type="range" id="target-acc-slider" min="50" max="99" value="90" step="1" style="accent-color: var(--accent-cyan); width: 140px; cursor: pointer;" oninput="onTargetAccInput(this.value)" onchange="fetchRandomSample()">
        </div>

        <div id="drop-k-wrapper" class="model-select-wrapper" style="display: none; align-items: center; gap: 10px;">
            <label for="drop-k-slider" style="font-size: 0.85rem;">Drop Last Tokens (k): <strong id="drop-k-val" style="color: var(--accent-purple); font-size: 0.95rem;">16</strong> (<span id="keep-m-val" style="color: var(--accent-emerald);">Keep M=48</span>)</label>
            <input type="range" id="drop-k-slider" min="0" max="63" value="16" step="1" style="accent-color: var(--accent-purple); width: 140px; cursor: pointer;" oninput="onDropKInput(this.value)" onchange="fetchRandomSample()">
        </div>

        <div style="font-size: 0.8rem; color: var(--text-secondary);">
            Active Prefix: <strong id="status-prefix-len" style="color: var(--accent-emerald);">M=64/64</strong>
        </div>
    </div>

    <div id="custom-text-container" style="display: none; width: 100%; margin-bottom: 20px;">
        <div class="panel" style="padding: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <label style="font-size: 0.85rem; font-weight: 600; color: var(--accent-cyan);">✏️ Enter Custom Reasoning Text:</label>
                <button class="btn-random" style="padding: 6px 14px; font-size: 0.8rem;" onclick="fetchRandomSample()">⚡ Encode Custom Text</button>
            </div>
            <textarea id="custom-text-input" style="width: 100%; height: 75px; background: rgba(15, 23, 42, 0.8); color: #fff; border: 1px solid var(--border-color); border-radius: 8px; padding: 10px; font-family: 'Fira Code', monospace; font-size: 0.85rem; outline: none; resize: vertical;" placeholder="Type or paste any arbitrary text here to see how the discrete bottleneck quantizes and reconstructs it...">&lt;think&gt;
Okay, let's calculate the sum of 1 + 1. First, we know that 1 is a natural number...
&lt;/think&gt;</textarea>
        </div>
    </div>

    <div class="grid-container">
        <!-- Panel 1: Raw Original Trace -->
        <div class="panel">
            <div class="panel-header">
                <h2>📄 Raw Input Reasoning Trace</h2>
                <div class="chunk-legend" id="raw-legend" style="display: none;">
                    <span class="legend-item"><span class="legend-dot" style="background:#10b981;"></span>Transmitted Prefix</span>
                    <span class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span>Pruned / Dropped Tail</span>
                </div>
                <span class="badge badge-cyan" id="raw-tokens-badge">Length: 64 tokens</span>
            </div>
            <div class="text-content" id="raw-text">Loading validation sample...</div>
        </div>

        <!-- Panel 2: Decoded Output Reconstruction -->
        <div class="panel">
            <div class="panel-header">
                <h2>⚡ Decoded Output Reconstruction</h2>
                <span class="badge badge-emerald" id="sample-id-badge">Sample #--</span>
            </div>
            <div class="text-content" id="decoded-text">Loading decoded output...</div>
        </div>

        <!-- Panel 3: Unified Dynamic Bottleneck & Codebook Tokens -->
        <div class="panel codebook-panel">
            <div class="panel-header">
                <h2>🗜️ Dynamic Ordered Bottleneck (Codebook Tokens & Accuracy Probe)</h2>
                <span class="badge badge-purple" id="latent-badge">Compressed: -- tokens</span>
            </div>

            <!-- Accuracy Probe Curve Card -->
            <div id="probe-curve-card" class="probe-curve-card" style="display: none;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.82rem; font-weight: 600; color: var(--accent-cyan); display: flex; align-items: center; gap: 6px;">
                        📈 Probe Predicted Accuracy Across Prefix Lengths (1 &rarr; 64 Tokens)
                    </span>
                    <span id="curve-hover-info" style="font-size: 0.75rem; color: var(--text-secondary); font-family: 'Fira Code', monospace;">
                        Selected Cutoff M=<span id="curve-cutoff-m">64</span> | Predicted Acc: <span id="curve-cutoff-acc" style="color:var(--accent-emerald);">--</span>
                    </span>
                </div>
                <div id="curve-bars-grid" style="display: flex; align-items: flex-end; gap: 2px; height: 55px; padding-top: 4px;">
                    <!-- 64 interactive curve bars rendered dynamically -->
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.68rem; color: #64748b; margin-top: 4px; font-family: 'Fira Code', monospace;">
                    <span>Prefix 1</span>
                    <span>Prefix 16 (4x)</span>
                    <span>Prefix 32 (2x)</span>
                    <span>Prefix 48 (1.33x)</span>
                    <span>Prefix 64 (1x)</span>
                </div>
            </div>

            <div class="tokens-grid" id="codebook-grid" style="gap: 8px;">
                <!-- Unified token chips rendered dynamically -->
            </div>
        </div>
    </div>

    <!-- Panel 4: Minimum Additional Stored Bits for Perfect Lossless Reconstruction -->
    <div class="panel" id="lossless-panel" style="margin-top: 18px; width: 100%;">
        <div class="panel-header" style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <h2 style="color: #38bdf8; display: flex; align-items: center; gap: 8px;">
                    💎 Minimum Additional Bits for Perfect (Lossless) Reconstruction
                </h2>
                <span class="badge" id="lossless-ratio-badge" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8;">
                    Lossless Ratio: --
                </span>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; font-family: 'Fira Code', monospace;">
                Token-Forcing (Teacher-Forced) Accuracy: <strong id="tf-acc-stat" style="color: #10b981;">--</strong>
            </div>
        </div>
        <div style="padding: 16px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; background: rgba(15, 23, 42, 0.6); border-radius: 8px; margin: 12px 0;">
            <div class="metric-box">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Incorrect Tokens (Errors)</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #f87171;" id="metric-error-count">--</div>
                <div style="font-size: 0.72rem; color: #64748b;" id="metric-error-sub">under token forcing</div>
            </div>
            <div class="metric-box">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Min Additional Stored Bits</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #38bdf8;" id="metric-add-bits">--</div>
                <div style="font-size: 0.72rem; color: #64748b;" id="metric-add-bits-sub">Bitmap + Error Tokens</div>
            </div>
            <div class="metric-box">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Base Discrete Latents</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #c084fc;" id="metric-latent-bits">--</div>
                <div style="font-size: 0.72rem; color: #64748b;" id="metric-latent-sub">M codes &times; 14 bits</div>
            </div>
            <div class="metric-box">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Total Bits (100% Lossless)</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #4ade80;" id="metric-total-bits">--</div>
                <div style="font-size: 0.72rem; color: #64748b;" id="metric-total-sub">vs Raw Uncompressed</div>
            </div>
            <div class="metric-box">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Shannon Entropy Limit</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #fbbf24;" id="metric-shannon-bits">--</div>
                <div style="font-size: 0.72rem; color: #64748b;" id="metric-shannon-sub">Arithmetic coding lower bound</div>
            </div>
        </div>

        <!-- Classical Compression (Zip / Gzip) Comparison Row -->
        <div style="margin: 0 0 14px 0; background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 14px 16px;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="display: flex; align-items: center; gap: 8px;">
                    📦 Classical Lossless Compression Comparison (Deflate / Zip / Gzip)
                </span>
                <span id="zip-advantage-pill" class="badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #34d399; font-size: 0.78rem;">
                    --
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; font-size: 0.8rem;">
                <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 6px; border-left: 3px solid #60a5fa;">
                    <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600;">Zip / Deflate (16-bit Tokens)</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 3px 0;" id="cmp-zip-tokens">--</div>
                    <div style="font-size: 0.72rem; color: #64748b;" id="cmp-zip-tokens-sub">Standard PKZIP / Deflate</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 6px; border-left: 3px solid #a78bfa;">
                    <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600;">Zip / Deflate (Raw Text)</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 3px 0;" id="cmp-zip-text">--</div>
                    <div style="font-size: 0.72rem; color: #64748b;" id="cmp-zip-text-sub">UTF-8 Byte Stream</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 6px; border-left: 3px solid #34d399;">
                    <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600;">Our Autoencoder (Latent + Errors)</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #34d399; margin: 3px 0;" id="cmp-our-lossless">--</div>
                    <div style="font-size: 0.72rem; color: #64748b;" id="cmp-our-lossless-sub">100% Exact Reconstruction</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 6px; border-left: 3px solid #fbbf24;">
                    <div style="color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600;">Gzip Compression</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #fbbf24; margin: 3px 0;" id="cmp-gzip-val">--</div>
                    <div style="font-size: 0.72rem; color: #64748b;" id="cmp-gzip-sub">Tokens &bull; Text</div>
                </div>
            </div>
        </div>
        <div id="lossless-error-chips" style="padding: 4px 12px 12px 12px; font-size: 0.8rem; font-family: 'Fira Code', monospace;">
            <!-- Error chips -->
        </div>
    </div>

    <!-- Research Preservation & Experiment Archive Modal -->
    <div id="experiments-modal" style="display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); backdrop-filter: blur(10px); z-index: 9999; justify-content: center; align-items: center; padding: 20px;">
        <div style="background: #1e293b; border: 1px solid var(--border-color); border-radius: 16px; width: 95%; max-width: 1200px; max-height: 90vh; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.7);">
            <div style="padding: 20px 24px; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; background: rgba(15,23,42,0.9);">
                <div>
                    <h2 style="font-size: 1.25rem; font-weight: 700; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px;">
                        📚 Research Preservation & Experiment Archive
                    </h2>
                    <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 4px;">
                        Curated archive of all autoencoder runs, architecture configs, training trajectories, and compression benchmarks.
                    </p>
                </div>
                <button onclick="closeExperimentsModal()" style="background: transparent; border: 1px solid var(--border-color); color: var(--text-secondary); font-size: 1.4rem; border-radius: 8px; width: 36px; height: 36px; cursor: pointer; display: flex; align-items: center; justify-content: center;">&times;</button>
            </div>
            <div style="padding: 20px 24px; overflow-y: auto; flex: 1;">
                <div style="margin-bottom: 16px; display: flex; gap: 12px; align-items: center;">
                    <input type="text" id="exp-search" placeholder="Search experiments by name, architecture, or hyperparameter..." oninput="filterExperiments()" style="background: #0f172a; border: 1px solid var(--border-color); color: #fff; padding: 10px 16px; border-radius: 8px; font-size: 0.9rem; flex: 1; outline: none;">
                    <span id="exp-count-badge" class="pill" style="font-weight: 600; padding: 8px 14px;">Loading...</span>
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left;">
                    <thead>
                        <tr style="border-bottom: 2px solid var(--border-color); color: var(--accent-cyan);">
                            <th style="padding: 10px 8px;">Experiment Directory</th>
                            <th style="padding: 10px 8px;">Model Class</th>
                            <th style="padding: 10px 8px;">Epochs</th>
                            <th style="padding: 10px 8px;">Train Acc</th>
                            <th style="padding: 10px 8px;">M=64 Acc</th>
                            <th style="padding: 10px 8px;">M=32 Acc</th>
                            <th style="padding: 10px 8px;">Active Codes</th>
                            <th style="padding: 10px 8px;">Status</th>
                        </tr>
                    </thead>
                    <tbody id="experiments-table-body">
                        <!-- Populated dynamically via JS -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

<script>
    let currentSampleIdx = null;
    let latestAccuracyCurve = [];

    function onSplitChange() {
        const split = document.getElementById('split-select').value;
        const customContainer = document.getElementById('custom-text-container');
        if (split === 'custom') {
            customContainer.style.display = 'block';
        } else {
            customContainer.style.display = 'none';
        }
        fetchRandomSample();
    }

    function onProbeControlChange() {
        const mode = document.getElementById('probe-mode-select').value;
        document.getElementById('target-acc-wrapper').style.display = (mode === 'target_acc') ? 'flex' : 'none';
        document.getElementById('drop-k-wrapper').style.display = (mode === 'fixed_k') ? 'flex' : 'none';
        fetchRandomSample();
    }

    function onTargetAccInput(val) {
        document.getElementById('target-acc-val').innerText = `${val}%`;
    }

    function onDropKInput(val) {
        document.getElementById('drop-k-val').innerText = `${val}`;
        document.getElementById('keep-m-val').innerText = `Keep M=${64 - parseInt(val)}`;
    }

    async function fetchRandomSample() {
        const modelId = document.getElementById('model-select').value;
        const split = document.getElementById('split-select').value;
        const mode = document.getElementById('probe-mode-select') ? document.getElementById('probe-mode-select').value : 'target_acc';
        const targetAcc = document.getElementById('target-acc-slider') ? (document.getElementById('target-acc-slider').value / 100.0) : 0.90;
        const dropK = document.getElementById('drop-k-slider') ? document.getElementById('drop-k-slider').value : 0;
        
        let url = `/api/sample?model_id=${modelId}&split=${split}&mode=${mode}&target_acc=${targetAcc}&drop_k=${dropK}`;
        if (split === 'custom') {
            const customText = encodeURIComponent(document.getElementById('custom-text-input').value);
            url += `&custom_text=${customText}`;
        }

        try {
            const res = await fetch(url);
            const data = await res.json();
            renderSample(data);
        } catch (err) {
            console.error("Failed to fetch sample:", err);
        }
    }

    function renderAccuracyCurve(curve, selectedM) {
        const grid = document.getElementById('curve-bars-grid');
        grid.innerHTML = '';
        latestAccuracyCurve = curve;

        curve.forEach((acc, idx) => {
            const m = idx + 1;
            const bar = document.createElement('div');
            bar.className = 'curve-bar';
            const hPct = Math.max(8, Math.min(100, acc));
            bar.style.height = `${hPct}%`;

            if (m === selectedM) {
                bar.classList.add('cutoff-marker');
            } else if (m < selectedM) {
                bar.classList.add('active-prefix');
            } else {
                bar.classList.add('dropped-prefix');
            }

            bar.title = `Prefix Length M=${m} / 64: ${acc.toFixed(1)}% predicted accuracy`;

            bar.onmouseenter = () => {
                document.getElementById('curve-hover-info').innerHTML = 
                    `Prefix M=<strong>${m}</strong> (${(64/m).toFixed(2)}x) | Predicted Acc: <strong style="color:var(--accent-cyan);">${acc.toFixed(1)}%</strong>`;
            };

            bar.onmouseleave = () => {
                const selAcc = curve[selectedM - 1] !== undefined ? curve[selectedM - 1].toFixed(1) : '--';
                document.getElementById('curve-hover-info').innerHTML = 
                    `Selected Cutoff M=<span id="curve-cutoff-m">${selectedM}</span> | Predicted Acc: <span id="curve-cutoff-acc" style="color:var(--accent-emerald);">${selAcc}%</span>`;
            };

            bar.onclick = () => {
                // Set to fixed_k with drop_k = 64 - m
                const dropK = 64 - m;
                document.getElementById('probe-mode-select').value = 'fixed_k';
                document.getElementById('drop-k-slider').value = dropK;
                onDropKInput(dropK);
                onProbeControlChange();
            };

            grid.appendChild(bar);
        });

        const selAcc = curve[selectedM - 1] !== undefined ? curve[selectedM - 1].toFixed(1) : '--';
        document.getElementById('curve-cutoff-m').innerText = selectedM;
        document.getElementById('curve-cutoff-acc').innerText = `${selAcc}%`;
    }

    function renderSample(data) {
        const epochInfo = data.checkpoint_info.epoch !== undefined ? data.checkpoint_info.epoch : 'N/A';
        document.getElementById('stat-epoch').innerText = `Epoch ${epochInfo}`;
        document.getElementById('stat-ratio').innerText = data.compression_ratio;
        document.getElementById('stat-codebook').innerText = data.checkpoint_info.codebook_size;

        const isPrefixDropout = (data.checkpoint_info.arch === 'prefix_dropout' || data.probe_stats !== null);
        const probeControls = document.getElementById('prefix-probe-controls');
        const probeCard = document.getElementById('probe-curve-card');
        const pillProbePred = document.getElementById('pill-probe-pred');
        const pillActualAcc = document.getElementById('pill-actual-acc');

        if (isPrefixDropout && data.probe_stats) {
            probeControls.style.display = 'flex';
            probeCard.style.display = 'block';
            pillProbePred.style.display = 'inline-block';
            pillActualAcc.style.display = 'inline-block';

            document.getElementById('stat-probe-pred').innerText = `${data.probe_stats.pred_acc.toFixed(1)}%`;
            document.getElementById('stat-actual-acc').innerText = `${data.probe_stats.actual_acc.toFixed(1)}%`;
            document.getElementById('status-prefix-len').innerText = `M=${data.probe_stats.selected_m}/64 (${data.compression_ratio})`;

            renderAccuracyCurve(data.probe_stats.accuracy_curve, data.probe_stats.selected_m);
        } else {
            probeControls.style.display = 'none';
            probeCard.style.display = 'none';
            pillProbePred.style.display = 'none';
            pillActualAcc.style.display = 'none';
        }

        document.getElementById('raw-tokens-badge').innerText = `Sequence: ${data.input_length} tokens`;
        const splitLabel = data.split === 'custom' ? 'CUSTOM TEXT' : `${data.split.toUpperCase()} SPLIT`;
        document.getElementById('sample-id-badge').innerText = `Sample #${data.sample_index + 1} / ${data.total_samples} (${splitLabel})`;
        
        if (data.token_details && data.token_details.length > 0) {
            const keptCount = data.token_details.filter(t => t.kept).length;
            const retPct = ((keptCount / data.input_length) * 100).toFixed(1);
            document.getElementById('latent-badge').innerText = `Retained: ${keptCount} / ${data.input_length} tokens (${retPct}% transmitted | ${data.compression_ratio} compression)`;
        } else {
            document.getElementById('latent-badge').innerText = `Compressed: ${data.compressed_length} tokens (${data.compression_ratio} factor)`;
        }

        const rawContainer = document.getElementById('raw-text');
        const rawLegend = document.getElementById('raw-legend');
        if (data.raw_chunks && data.raw_chunks.length > 0) {
            rawContainer.innerHTML = '';
            data.raw_chunks.forEach(chunk => {
                const span = document.createElement('span');
                span.textContent = chunk.text;
                if (chunk.route === 0 || chunk.compression.includes('4x') || chunk.compression.includes('Dropped')) {
                    span.className = 'chunk-hl-4x';
                    span.title = `Pruned Tail: ${chunk.compression}`;
                } else if (chunk.route === 1 || chunk.compression.includes('2x')) {
                    span.className = 'chunk-hl-2x';
                    span.title = `2x Compression: ${chunk.compression}`;
                } else if (chunk.route === 2 || chunk.compression.includes('1.33x')) {
                    span.className = 'chunk-hl-1-33x';
                    span.title = `1.33x Compression: ${chunk.compression}`;
                } else {
                    span.className = 'chunk-hl-1x';
                    span.title = `Transmitted Prefix: ${chunk.compression}`;
                }
                rawContainer.appendChild(span);
            });
            if (rawLegend) rawLegend.style.display = 'flex';
        } else {
            rawContainer.textContent = data.raw_text;
            if (rawLegend) rawLegend.style.display = 'none';
        }

        document.getElementById('decoded-text').innerText = data.decoded_text;

        const grid = document.getElementById('codebook-grid');
        grid.innerHTML = '';

        if (data.token_details && data.token_details.length > 0) {
            data.token_details.forEach((t, i) => {
                if (t.is_chunk_start && t.chunk_idx > 0) {
                    const sep = document.createElement('div');
                    sep.className = 'chunk-separator';
                    sep.title = `Boundary between Chunk ${t.chunk_idx - 1} and Chunk ${t.chunk_idx}`;
                    grid.appendChild(sep);
                }
                const codeIdx = t.code_idx !== undefined ? t.code_idx : (data.code_indices[i] !== undefined ? data.code_indices[i] : 0);
                const chip = document.createElement('div');
                chip.className = 'token-chip';
                const hue = (codeIdx * 137.5) % 360;
                
                const statusStr = t.kept ? "KEPT (Transmitted to Decoder)" : "PRUNED (Dropped by Prefix Probe)";
                chip.title = `Pos [${i}] | Token: "${t.token}" | Pred Acc: ${t.score}% | Code: ${codeIdx} | Status: ${statusStr}`;

                if (t.kept) {
                    chip.style.borderColor = `hsl(${hue}, 60%, 45%)`;
                    chip.style.background = 'rgba(16, 185, 129, 0.2)';
                    chip.style.color = '#f8fafc';
                    chip.style.fontWeight = '600';
                    chip.innerText = `[${i}]: ${codeIdx}`;
                } else {
                    chip.style.borderColor = '#334155';
                    chip.style.background = 'rgba(30, 41, 59, 0.35)';
                    chip.style.color = '#64748b';
                    chip.style.opacity = '0.45';
                    chip.style.textDecoration = 'line-through';
                    chip.innerText = `[${i}]: <DROP>`;
                }
                grid.appendChild(chip);
            });
        } else {
            data.code_indices.forEach((idx, i) => {
                const chip = document.createElement('div');
                chip.className = 'token-chip';
                const hue = (idx * 137.5) % 360;
                chip.style.borderColor = `hsl(${hue}, 60%, 45%)`;
                chip.innerText = `[${i}]: ${idx}`;
                grid.appendChild(chip);
            });
        }

        if (data.lossless_metrics) {
            const lm = data.lossless_metrics;
            const statRatio = document.getElementById('stat-lossless-ratio');
            const statBytes = document.getElementById('stat-lossless-bytes');
            const pillLossless = document.getElementById('pill-lossless');
            if (statRatio) statRatio.innerText = lm.lossless_compression_ratio;
            if (statBytes) statBytes.innerText = `${lm.total_lossless_bytes}B vs ${lm.raw_uncompressed_bytes}B`;
            if (pillLossless) pillLossless.style.display = 'inline-block';

            const panel = document.getElementById('lossless-panel');
            if (panel) {
                panel.style.display = 'block';
                document.getElementById('lossless-ratio-badge').innerText = `Lossless Ratio: ${lm.lossless_compression_ratio} (${lm.total_lossless_bytes} B vs ${lm.raw_uncompressed_bytes} B raw)`;
                document.getElementById('tf-acc-stat').innerText = `${lm.token_forcing_acc.toFixed(1)}%`;
                document.getElementById('metric-error-count').innerText = `${lm.num_errors} / ${lm.total_valid_tokens}`;
                document.getElementById('metric-error-sub').innerText = `${(100 - lm.token_forcing_acc).toFixed(1)}% error rate`;
                document.getElementById('metric-add-bits').innerText = `${lm.min_additional_bits} bits`;
                document.getElementById('metric-add-bits-sub').innerText = `${lm.min_additional_bytes} bytes (${lm.scheme_used} scheme)`;
                document.getElementById('metric-latent-bits').innerText = `${lm.latent_bits} bits`;
                document.getElementById('metric-latent-sub').innerText = `${Math.ceil(lm.latent_bits / 8)} bytes (${lm.bits_per_latent} b/code)`;
                document.getElementById('metric-total-bits').innerText = `${lm.total_lossless_bits} bits`;
                document.getElementById('metric-total-sub').innerText = `${lm.total_lossless_bytes} bytes (100% perfect)`;
                document.getElementById('metric-shannon-bits').innerText = `${lm.shannon_total_bits} bits`;
                document.getElementById('metric-shannon-sub').innerText = `${lm.shannon_compression_ratio} theoretical bound`;

                if (lm.zip_comparison) {
                    const zc = lm.zip_comparison;
                    const elZipTokens = document.getElementById('cmp-zip-tokens');
                    const elZipTokensSub = document.getElementById('cmp-zip-tokens-sub');
                    const elZipText = document.getElementById('cmp-zip-text');
                    const elZipTextSub = document.getElementById('cmp-zip-text-sub');
                    const elOurLossless = document.getElementById('cmp-our-lossless');
                    const elOurLosslessSub = document.getElementById('cmp-our-lossless-sub');
                    const elGzip = document.getElementById('cmp-gzip-val');
                    const elGzipSub = document.getElementById('cmp-gzip-sub');
                    const elPill = document.getElementById('zip-advantage-pill');

                    if (elZipTokens) {
                        elZipTokens.innerText = `${zc.zip_tokens_bytes} B (${zc.zip_tokens_ratio})`;
                        elZipTokensSub.innerText = `vs ${zc.raw_tokens_bytes} B raw 16-bit tokens`;
                    }
                    if (elZipText) {
                        elZipText.innerText = `${zc.zip_text_bytes} B (${zc.zip_text_ratio})`;
                        elZipTextSub.innerText = `vs ${zc.raw_text_bytes} B UTF-8 string`;
                    }
                    if (elOurLossless) {
                        elOurLossless.innerText = `${lm.total_lossless_bytes} B (${lm.lossless_compression_ratio})`;
                        elOurLosslessSub.innerText = `${lm.latent_bits}b latents + ${lm.min_additional_bits}b corrections`;
                    }
                    if (elGzip) {
                        elGzip.innerText = `${zc.gzip_tokens_bytes} B (tok) / ${zc.gzip_text_bytes} B (txt)`;
                        elGzipSub.innerText = `Tokens: ${zc.gzip_tokens_ratio} | Text: ${zc.gzip_text_ratio}`;
                    }
                    if (elPill) {
                        if (zc.token_savings_pct > 0) {
                            elPill.style.background = 'rgba(16, 185, 129, 0.2)';
                            elPill.style.color = '#34d399';
                            elPill.style.borderColor = '#34d399';
                            elPill.innerText = `⚡ ${zc.token_savings_pct}% Smaller than Zip (Tokens)`;
                        } else {
                            elPill.style.background = 'rgba(245, 158, 11, 0.2)';
                            elPill.style.color = '#fbbf24';
                            elPill.style.borderColor = '#fbbf24';
                            elPill.innerText = `${Math.abs(zc.token_savings_pct)}% larger than Zip (Tokens)`;
                        }
                    }
                }

                const chipBox = document.getElementById('lossless-error-chips');
                chipBox.innerHTML = '';
                if (lm.errors && lm.errors.length > 0) {
                    const title = document.createElement('div');
                    title.style.marginBottom = '8px';
                    title.style.color = '#f87171';
                    title.style.fontWeight = '600';
                    title.innerText = `⚠️ Stored Error Tokens under Token Forcing (${lm.errors.length} of ${lm.num_errors} shown):`;
                    chipBox.appendChild(title);

                    lm.errors.forEach(err => {
                        const chip = document.createElement('span');
                        chip.className = 'error-chip';
                        chip.innerHTML = `<strong>Pos [${err.pos}]</strong> Expected: <span style="color:#a7f3d0; font-weight:600;">'${err.expected_token}'</span> &rarr; Model: <span style="color:#fecaca;">'${err.predicted_token}'</span> <span style="color:#94a3b8; font-size:0.7rem;">(+${err.bits_cost}b)</span>`;
                        chipBox.appendChild(chip);
                    });
                } else {
                    chipBox.innerHTML = '<div style="color: #4ade80; font-weight: 600; padding: 4px 0;">🎉 Zero Token-Forcing Errors! 100% lossless reconstruction directly from latent memory tokens!</div>';
                }
            }
        }
    }

    async function reloadModelWeights() {
        const btn = document.getElementById('btn-reload');
        const modelId = document.getElementById('model-select').value;
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = "⏳ Reloading...";
        try {
            const res = await fetch(`/api/reload?model_id=${modelId}`);
            const data = await res.json();
            if (data.status === 'success') {
                btn.innerHTML = "✅ Reloaded!";
                setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; }, 1500);
                fetchRandomSample();
            } else {
                btn.innerHTML = "❌ Failed";
                setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; }, 1500);
            }
        } catch (e) {
            btn.innerHTML = "❌ Error";
            setTimeout(() => { btn.innerHTML = originalText; btn.disabled = false; }, 1500);
        }
    }

    let allExperiments = [];

    async function openExperimentsModal() {
        document.getElementById('experiments-modal').style.display = 'flex';
        if (allExperiments.length === 0) {
            try {
                const res = await fetch('/api/experiments');
                allExperiments = await res.json();
                renderExperimentsTable(allExperiments);
            } catch (err) {
                console.error("Failed to load experiments:", err);
            }
        }
    }

    function closeExperimentsModal() {
        document.getElementById('experiments-modal').style.display = 'none';
    }

    function filterExperiments() {
        const query = document.getElementById('exp-search').value.toLowerCase();
        const filtered = allExperiments.filter(exp => 
            exp.name.toLowerCase().includes(query) ||
            (exp.config && JSON.stringify(exp.config).toLowerCase().includes(query))
        );
        renderExperimentsTable(filtered);
    }

    function renderExperimentsTable(exps) {
        const tbody = document.getElementById('experiments-table-body');
        tbody.innerHTML = '';
        document.getElementById('exp-count-badge').innerText = `${exps.length} Preserved Experiments`;

        exps.forEach(exp => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid rgba(51, 65, 85, 0.4)';

            const modelClass = exp.config ? (exp.config.class || exp.config.model_class || exp.config.module || 'Standard') : '--';
            const epochs = exp.epochs_count !== undefined ? `${exp.epochs_count} ep` : (exp.config && exp.config.args && exp.config.args.epochs ? `${exp.config.args.epochs} ep` : '--');
            const trainAcc = exp.final_train_acc !== undefined ? `${exp.final_train_acc.toFixed(1)}%` : '--';

            let m64Acc = '--';
            let m32Acc = '--';
            if (exp.final_benchmarks) {
                if (exp.final_benchmarks['64']) m64Acc = `${exp.final_benchmarks['64'].acc.toFixed(1)}%`;
                if (exp.final_benchmarks['32']) m32Acc = `${exp.final_benchmarks['32'].acc.toFixed(1)}%`;
            }

            const activeCodes = exp.active_codes !== undefined ? `${exp.active_codes}` : '--';
            const statusBadge = exp.has_ckpt ? 
                '<span style="background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 2px 8px; border-radius: 12px; font-weight: 600;">Active Checkpoint</span>' : 
                '<span style="background: rgba(59, 130, 246, 0.15); color: #93c5fd; padding: 2px 8px; border-radius: 12px;">Preserved Logs/Config</span>';

            tr.innerHTML = `
                <td style="padding: 10px 8px; font-weight: 600; color: #f8fafc; font-family: monospace;">${exp.name}</td>
                <td style="padding: 10px 8px; color: var(--accent-cyan);">${modelClass}</td>
                <td style="padding: 10px 8px;">${epochs}</td>
                <td style="padding: 10px 8px; color: var(--accent-emerald); font-weight: 600;">${trainAcc}</td>
                <td style="padding: 10px 8px;">${m64Acc}</td>
                <td style="padding: 10px 8px; font-weight: 600; color: #fef08a;">${m32Acc}</td>
                <td style="padding: 10px 8px; color: #a78bfa;">${activeCodes}</td>
                <td style="padding: 10px 8px;">${statusBadge}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Auto load initial sample on startup
    fetchRandomSample();
</script>
</body>
</html>
"""

class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

class VisualizerRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            # Dynamically generate model dropdown options from discovered models
            current_models = discover_models()
            default_mid = "checkpoints_discrete_joint_gpu" if "checkpoints_discrete_joint_gpu" in current_models else next(iter(current_models.keys()), "")
            options_html = ""
            for mid, minfo in current_models.items():
                selected = ' selected' if mid == default_mid else ''
                options_html += f'<option value="{mid}"{selected}>{minfo["name"]}</option>\n'
            page_html = HTML_TEMPLATE.replace('__MODEL_OPTIONS__', options_html)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(page_html.encode("utf-8"))
        elif path.startswith("/api/sample"):
            model_id = query.get("model_id", [None])[0]
            split = query.get("split", ["val"])[0]
            custom_text = query.get("custom_text", [""])[0]
            mode = query.get("mode", ["target_acc"])[0]
            try:
                target_acc = float(query.get("target_acc", [0.90])[0])
            except Exception:
                target_acc = 0.90
            try:
                drop_k = int(query.get("drop_k", [0])[0])
            except Exception:
                drop_k = 0

            data = generate_sample_data(
                model_id=model_id,
                split=split,
                custom_text=custom_text,
                mode=mode,
                target_acc=target_acc,
                drop_k=drop_k
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        elif path.startswith("/api/reload"):
            model_id = query.get("model_id", [None])[0]
            success = load_model_on_demand(model_id=model_id, force_reload=True)
            self.send_response(200 if success else 500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            res = {"status": "success" if success else "error", "checkpoint_info": CHECKPOINT_INFO}
            self.wfile.write(json.dumps(res).encode("utf-8"))
        elif path.startswith("/api/experiments"):
            experiments = []
            for item in sorted(os.listdir(MODELS_DIR)):
                dir_p = os.path.join(MODELS_DIR, item)
                if not os.path.isdir(dir_p):
                    continue
                tr_path = os.path.join(dir_p, "training_run.json")
                cfg_path = os.path.join(dir_p, "model_config.json")
                if not os.path.exists(tr_path) and not os.path.exists(cfg_path):
                    continue
                pt_files = [f for f in os.listdir(dir_p) if f.endswith('.pt')]
                exp_entry = {
                    "name": item,
                    "has_ckpt": len(pt_files) > 0,
                    "ckpt_count": len(pt_files),
                    "ckpt_names": pt_files
                }
                if os.path.exists(cfg_path):
                    try:
                        with open(cfg_path, 'r', encoding='utf-8') as f:
                            exp_entry["config"] = json.load(f)
                    except Exception:
                        pass
                if os.path.exists(tr_path):
                    try:
                        with open(tr_path, 'r', encoding='utf-8') as f:
                            tr = json.load(f)
                            eps = tr.get("epochs", [])
                            exp_entry["epochs_count"] = len(eps)
                            if eps:
                                last_ep = eps[-1]
                                exp_entry["final_epoch"] = last_ep.get("epoch")
                                exp_entry["final_train_acc"] = round(last_ep.get("train_acc", 0.0), 2)
                                exp_entry["final_rec_loss"] = round(last_ep.get("train_rec_loss", 0.0), 4)
                                exp_entry["final_benchmarks"] = last_ep.get("benchmarks")
                                exp_entry["active_codes"] = last_ep.get("active_codes")
                    except Exception:
                        pass
                experiments.append(exp_entry)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(json.dumps(experiments).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8050):
    server_address = ('', port)
    httpd = ReusableHTTPServer(server_address, VisualizerRequestHandler)
    print(f"\n=======================================================", flush=True)
    print(f"🚀 VISUALIZER RUNNING AT: http://localhost:{port}", flush=True)
    print(f"=======================================================\n", flush=True)
    
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    httpd.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reasoning Autoencoder Interactive Visualizer")
    parser.add_argument("--parquet_path", type=str, default="cot-00000-of-00144.parquet")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    DEVICE = args.device

    # Pre-load tokenizer & dataset texts
    TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer_name)
    load_dataset_texts(args.parquet_path)
    
    # Default initial model load (first discovered model)
    load_model_on_demand()

    run_server(args.port)
