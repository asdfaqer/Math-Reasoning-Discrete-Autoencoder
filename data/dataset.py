"""
Dynamic Reasoning Trace Dataset Loader and Preprocessing Utilities.

Supports loading raw parquet files (GSM8k, NuminaMath, Open-R1 reasoning traces)
and tokenizing them into fixed-length spans for autoencoder training and evaluation.
"""

import os
import sys
import glob
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class DynamicReasoningTraceDataset(Dataset):
    """
    Loads reasoning traces from Parquet files and dynamically extracts spans
    for autoencoder sequence compression.
    """
    def __init__(
        self,
        parquet_path,
        tokenizer,
        target_span_length=64,
        max_seq_length=1024,
        split_ratio=0.9,
        is_train=True,
        cache_dir="."
    ):
        self.target_span_length = target_span_length
        self.is_train = is_train
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        split_name = "train" if is_train else "val"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"tokenized_{split_name}_dynamic{max_seq_length}.pt")

        if os.path.exists(cache_file):
            print(f"[CACHE] Loading pre-tokenized sequences from {cache_file}...", flush=True)
            data = torch.load(cache_file)
            self.input_ids_list = data["input_ids_list"]
            self.texts = data["texts"]
            self.text_indices = data.get("text_indices", [i for i in range(len(self.texts))])
            print(f"[CACHE] Loaded {len(self.input_ids_list)} dynamic pre-tokenized samples into memory!", flush=True)
        else:
            print(f"Pre-tokenizing dataset ({split_name}) up to max_seq_length={max_seq_length}...", flush=True)
            if '*' in parquet_path or '?' in parquet_path:
                files = glob.glob(parquet_path)
            elif os.path.isdir(parquet_path):
                files = glob.glob(os.path.join(parquet_path, "*.parquet"))
            else:
                files = [parquet_path]
            
            print(f"Loading data from {len(files)} parquet file(s)...", flush=True)
            dfs = [pd.read_parquet(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            
            if "problem" in df.columns and "generated_solution" in df.columns:
                df["full_text"] = "Problem:\n" + df["problem"].fillna("") + "\n\nSolution:\n" + df["generated_solution"].fillna("")
                texts = df["full_text"].dropna().tolist()
            elif "generated_solution" in df.columns:
                texts = df["generated_solution"].dropna().tolist()
            else:
                texts = df["problem"].dropna().tolist()

            split_idx = int(len(texts) * split_ratio)
            self.texts = texts[:split_idx] if is_train else texts[split_idx:]

            chunk_size = 1000
            self.input_ids_list = []
            self.text_indices = []
            total_chunks = (len(self.texts) + chunk_size - 1) // chunk_size

            for i in range(0, len(self.texts), chunk_size):
                chunk_texts = self.texts[i:i+chunk_size]
                chunk_idx = i // chunk_size + 1
                enc = tokenizer(
                    chunk_texts,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False
                )
                for idx, ids in enumerate(enc["input_ids"]):
                    for j in range(0, len(ids), max_seq_length):
                        chunk_ids = ids[j:j+max_seq_length]
                        self.input_ids_list.append(torch.tensor(chunk_ids, dtype=torch.long))
                        self.text_indices.append(i + idx)

                print(f"   [Pre-tokenizing {split_name}] Chunk {chunk_idx}/{total_chunks} ({min(chunk_idx*chunk_size, len(self.texts))}/{len(self.texts)} items)", flush=True)

            torch.save({
                "input_ids_list": self.input_ids_list,
                "texts": self.texts,
                "text_indices": self.text_indices
            }, cache_file)
            print(f"[CACHE] Saved pre-tokenized cache -> {cache_file}", flush=True)

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        seq = self.input_ids_list[idx]
        L = len(seq)

        if self.is_train:
            # Random Span Cropping
            if L > self.target_span_length:
                max_start = L - self.target_span_length
                start = torch.randint(0, max_start + 1, (1,)).item()
                input_ids = seq[start : start + self.target_span_length]
                attention_mask = torch.ones(self.target_span_length, dtype=torch.long)
            else:
                pad_len = self.target_span_length - L
                input_ids = F.pad(seq, (0, pad_len), value=self.pad_token_id)
                attention_mask = F.pad(torch.ones(L, dtype=torch.long), (0, pad_len), value=0)
        else:
            # Deterministic Slicing for Validation
            if L >= self.target_span_length:
                input_ids = seq[: self.target_span_length]
                attention_mask = torch.ones(self.target_span_length, dtype=torch.long)
            else:
                pad_len = self.target_span_length - L
                input_ids = F.pad(seq, (0, pad_len), value=self.pad_token_id)
                attention_mask = F.pad(torch.ones(L, dtype=torch.long), (0, pad_len), value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw_text": self.texts[self.text_indices[idx]]
        }


def get_dataloaders(parquet_path, tokenizer_name="roberta-base", batch_size=16, max_length=64, max_seq_length=1024, cache_dir="."):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_ds = DynamicReasoningTraceDataset(
        parquet_path,
        tokenizer,
        target_span_length=max_length,
        max_seq_length=max_seq_length,
        is_train=True,
        cache_dir=cache_dir
    )
    val_ds = DynamicReasoningTraceDataset(
        parquet_path,
        tokenizer,
        target_span_length=max_length,
        max_seq_length=max_seq_length,
        is_train=False,
        cache_dir=cache_dir
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer
