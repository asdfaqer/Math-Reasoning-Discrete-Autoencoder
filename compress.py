"""
Reasoning Trace Discrete Compression & Reconstruction CLI.

Encodes raw reasoning text or pre-tokenized sequences into discrete codebook tokens
using the Cascading Memory Discrete Autoencoder, and autoregressively reconstructs
the reasoning chain with controllable compression budgets M in [8, 16, 24, 32, 48, 64].
"""

import os
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer

from cascading_memory_model import SmallCascadingMemoryAutoencoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})

    model = SmallCascadingMemoryAutoencoder(
        vocab_size=50265,
        d_model=saved_args.get("d_model", 72),
        nhead=saved_args.get("nhead", 4),
        num_layers=saved_args.get("num_layers", 3),
        decoder_num_layers=saved_args.get("decoder_num_layers", 3),
        text_encoder_layers=saved_args.get("text_encoder_layers", 3),
        codebook_size=saved_args.get("codebook_size", 16384),
        max_length=saved_args.get("max_length", 64),
        use_cosine=saved_args.get("use_cosine", True),
        append_m_query=saved_args.get("append_m_query", True),
        use_compression_embeddings=saved_args.get("use_compression_embeddings", True),
        causal_cross_attn=saved_args.get("causal_cross_attn", True),
        causal_encoder_queries=saved_args.get("causal_encoder_queries", True)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, checkpoint.get("epoch", "N/A"), checkpoint.get("benchmarks", {})


def compress_text(model, tokenizer, text, prefix_len=32, device="cpu"):
    # Tokenize input text to fixed max_length
    enc = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    orig_seq_len = input_ids.shape[1]

    # Pad if sequence length < 64
    if orig_seq_len < 64:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
        pad_len = 64 - orig_seq_len
        input_ids = torch.cat([input_ids, torch.full((1, pad_len), pad_id, dtype=torch.long, device=device)], dim=1)

    eval_len = min(orig_seq_len, 64)

    t0 = time.time()
    with torch.no_grad():
        # 1. Encode into discrete codes
        enc_out = model.encoder(input_ids, prefix_len=prefix_len, return_continuous=False)
        quantized = enc_out[0]
        code_indices_tensor = enc_out[2]
        t_enc = (time.time() - t0) * 1000

        has_appended = getattr(model, "append_m_query", False) and getattr(model, "use_compression_embeddings", False)
        mem_len = prefix_len + 1 if has_appended else prefix_len
        memory = quantized[:, :mem_len, :]

        # 2. Teacher-forced reconstruction (parallel pass)
        tf_out = model(input_ids, prefix_len=prefix_len, word_dropout=0.0)
        tf_preds = torch.argmax(tf_out["logits"][0, :eval_len], dim=-1)
        tf_correct = (tf_preds == input_ids[0, :eval_len]).sum().item()
        tf_acc = (tf_correct / max(1, eval_len)) * 100.0

        # 3. Autoregressive reconstruction (step-by-step decoding)
        t1 = time.time()
        bos_id = getattr(model, "bos_token_id", 0)
        generated_ids = model.decoder.generate(memory, start_token_id=bos_id, max_length=64).squeeze(0)
        t_dec = (time.time() - t1) * 1000

    target_tokens = input_ids[0, :eval_len]
    pred_tokens = generated_ids[:eval_len]
    ar_correct = (pred_tokens == target_tokens).sum().item()
    ar_acc = (ar_correct / max(1, eval_len)) * 100.0

    orig_text_decoded = tokenizer.decode(target_tokens, skip_special_tokens=True)
    recon_tf_decoded = tokenizer.decode(tf_preds, skip_special_tokens=True)
    recon_ar_decoded = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    retained_codes = code_indices_tensor[0, :prefix_len].cpu().tolist()

    return {
        "original_tokens": target_tokens.cpu().tolist(),
        "original_text": orig_text_decoded,
        "compressed_codes": retained_codes,
        "compressed_tokens_count": prefix_len,
        "compression_factor": 64.0 / float(prefix_len),
        "reconstructed_tf_text": recon_tf_decoded,
        "reconstructed_ar_text": recon_ar_decoded,
        "token_accuracy_tf": tf_acc,
        "token_accuracy_ar": ar_acc,
        "encode_latency_ms": t_enc,
        "decode_latency_ms": t_dec
    }



def main():
    parser = argparse.ArgumentParser(description="Compress and reconstruct reasoning traces with Discrete Autoencoder")
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints_discrete_joint_gpu/joint_discrete_epoch_15.pt",
                        help="Path to trained autoencoder checkpoint")
    parser.add_argument("--tokenizer_name", type=str, default="roberta-base",
                        help="HuggingFace tokenizer name")
    parser.add_argument("--prefix_len", type=int, default=32, choices=[8, 16, 24, 32, 48, 64],
                        help="Number of discrete latent tokens to retain (compression budget M)")
    parser.add_argument("--text", type=str, default=None,
                        help="Input reasoning text to compress (if None, uses default reasoning prompt)")
    parser.add_argument("--data_cache", type=str, default="tokenized_fast_tensor_64.pt",
                        help="Path to tokenized cache tensor for batch evaluation")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to evaluate if evaluating from cache")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 68)
    print("  MATHEMATICAL REASONING DISCRETE AUTOENCODER COMPRESSOR")
    print("=" * 68)
    print(f"Device: {args.device.upper()}")
    print(f"Loading checkpoint: {args.checkpoint}...")
    model, epoch, benchmarks = load_model(args.checkpoint, args.device)
    print(f"Loaded successfully (Trained Epoch: {epoch})")
    if benchmarks:
        print("\nModel Pre-computed Benchmarks:")
        for k, v in benchmarks.items():
            acc = v.get("acc", 0.0) if isinstance(v, dict) else v
            ratio = 64.0 / float(k)
            print(f"  * M = {str(k):>2s} tokens ({ratio:4.2f}x): {acc:5.2f}% token accuracy")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    cache_path = args.data_cache
    if not os.path.exists(cache_path):
        alt_path = os.path.join("data", cache_path)
        if os.path.exists(alt_path):
            cache_path = alt_path

    if args.text is not None:
        texts = [args.text]
    elif os.path.exists(cache_path):
        print(f"\nSampling {args.num_samples} reasoning traces from {cache_path}...")
        tensors = torch.load(cache_path, map_location="cpu").long()
        val_samples = tensors[-args.num_samples:]
        texts = [tokenizer.decode(sample, skip_special_tokens=True) for sample in val_samples]
    else:
        texts = [
            "Let x be a positive real number such that x^2 + 5x + 6 = 0. We can factor this as (x+2)(x+3) = 0, giving roots -2 and -3.",
            "To prove that the sequence converges, we apply the Cauchy criterion. For any epsilon > 0, there exists N such that |a_n - a_m| < epsilon.",
            "Suppose there exists an integer n such that 2^n - 1 is divisible by 7. By Fermat's Little Theorem, 2^3 = 8 = 1 (mod 7)."
        ]

    print("\n" + "-" * 68)
    for idx, sample_text in enumerate(texts):
        result = compress_text(model, tokenizer, sample_text, prefix_len=args.prefix_len, device=args.device)
        print(f"\n[Sample {idx + 1}] (Compression: {result['compression_factor']:.2f}x | Retained: {result['compressed_tokens_count']}/64 tokens)")
        print(f"Original Text:\n  {result['original_text']}")
        print(f"Discrete Codes (M={args.prefix_len} from Codebook Size 16,384):\n  {result['compressed_codes']}")
        print(f"Reconstructed (Teacher-Forcing):\n  {result['reconstructed_tf_text']}")
        print(f"Reconstructed (Autoregressive Generation):\n  {result['reconstructed_ar_text']}")
        print(f"Accuracy: Teacher-Forcing {result['token_accuracy_tf']:.2f}% | Autoregressive {result['token_accuracy_ar']:.2f}%")
        print(f"Latency: Enc {result['encode_latency_ms']:.1f}ms, Dec {result['decode_latency_ms']:.1f}ms")
        print("-" * 68)


if __name__ == "__main__":
    main()
