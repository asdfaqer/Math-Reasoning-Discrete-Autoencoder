import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import math
import gzip
import struct
import json
from transformers import AutoTokenizer
from cascading_memory_model import SmallCascadingMemoryAutoencoder

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(root_dir, "models", "checkpoints_discrete_joint_gpu", "joint_discrete_epoch_15.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SmallCascadingMemoryAutoencoder(
        vocab_size=50265, d_model=72, nhead=4, num_layers=3, decoder_num_layers=3,
        text_encoder_layers=3, codebook_size=16384, max_length=64, use_cosine=True,
        append_m_query=True, use_compression_embeddings=True, causal_cross_attn=True,
        causal_encoder_queries=True, normalize_prefix_pos=False
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cache_path = os.path.join(root_dir, "data", "tokenized_fast_tensor_64.pt")
    data = torch.load(cache_path, map_location="cpu")[-200:].long()

    m_vals = [8, 16, 24, 32, 48, 64]
    results = {}

    for m in m_vals:
        total_raw_bytes = 0
        total_gzip_tokens_bytes = 0
        total_gzip_text_bytes = 0
        total_raw_text_bytes = 0
        total_latent_bytes = 0
        total_neural_lossless_bytes = 0
        total_shannon_bytes = 0
        total_acc = 0

        with torch.no_grad():
            for i in range(len(data)):
                x = data[i:i+1].to(device)
                out = model(x, prefix_len=m, word_dropout=0.0)
                logits = out["logits"][0]
                preds = torch.argmax(logits, dim=-1)

                correct = (preds == x[0]).sum().item()
                total_acc += correct / 64.0

                # 1. Raw 16-bit packed token bytes (64 * 2 = 128 bytes)
                raw_bytes = struct.pack("<64H", *[int(t.item()) for t in x[0]])
                total_raw_bytes += len(raw_bytes)

                # 2. Gzip on 16-bit tokens
                gz_tokens = gzip.compress(raw_bytes, compresslevel=9)
                total_gzip_tokens_bytes += len(gz_tokens)

                # 3. Raw UTF-8 text and Gzip on text
                text = tokenizer.decode(x[0], skip_special_tokens=False).encode("utf-8")
                total_raw_text_bytes += len(text)
                gz_text = gzip.compress(text, compresslevel=9)
                total_gzip_text_bytes += len(gz_text)

                # 4. Neural discrete latent bytes
                latent_bits = m * 14
                total_latent_bytes += latent_bits / 8.0

                # 5. Neural lossless total (latents + exact sparse error correction)
                errors = (preds != x[0]).nonzero(as_tuple=True)[0]
                num_err = len(errors)
                min_err_bits = min(64 + num_err * 16, num_err * 22)
                lossless_bits = latent_bits + min_err_bits
                total_neural_lossless_bytes += lossless_bits / 8.0

                # 6. Shannon theoretical limit
                log_p = F.log_softmax(logits.float(), dim=-1)
                target_p = log_p.gather(dim=-1, index=x[0].unsqueeze(-1)).squeeze(-1)
                shannon_bits = latent_bits + float((-target_p / math.log(2.0)).sum().item())
                total_shannon_bytes += shannon_bits / 8.0

        n = len(data)
        results[m] = {
            "m": m,
            "compression_factor": 64.0 / float(m),
            "token_accuracy_pct": round((total_acc / n) * 100.0, 2),
            "raw_tokens_bytes": round(total_raw_bytes / n, 1),
            "raw_text_bytes": round(total_raw_text_bytes / n, 1),
            "gzip_tokens_bytes": round(total_gzip_tokens_bytes / n, 1),
            "gzip_text_bytes": round(total_gzip_text_bytes / n, 1),
            "latent_bytes": round(total_latent_bytes / n, 1),
            "neural_lossless_bytes": round(total_neural_lossless_bytes / n, 1),
            "shannon_limit_bytes": round(total_shannon_bytes / n, 1),
            "gzip_tokens_ratio": round((total_raw_bytes / n) / (total_gzip_tokens_bytes / n), 2),
            "gzip_text_ratio": round((total_raw_text_bytes / n) / (total_gzip_text_bytes / n), 2),
            "neural_lossless_ratio": round((total_raw_bytes / n) / (total_neural_lossless_bytes / n), 2),
            "shannon_ratio": round((total_raw_bytes / n) / (total_shannon_bytes / n), 2)
        }

    out_file = os.path.join(os.path.dirname(__file__), "lossless_benchmark_stats.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for m, r in results.items():
        print(f"M={m:2d} (Acc: {r['token_accuracy_pct']:5.1f}%): Raw={r['raw_tokens_bytes']:.0f}B | GzipTokens={r['gzip_tokens_bytes']:.1f}B ({r['gzip_tokens_ratio']:.2f}x) | GzipText={r['gzip_text_bytes']:.1f}B ({r['gzip_text_ratio']:.2f}x) | NeuralLossless={r['neural_lossless_bytes']:.1f}B ({r['neural_lossless_ratio']:.2f}x) | Shannon={r['shannon_limit_bytes']:.1f}B ({r['shannon_ratio']:.2f}x)")

if __name__ == "__main__":
    main()
