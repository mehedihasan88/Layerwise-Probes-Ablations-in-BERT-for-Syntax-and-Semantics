#!/usr/bin/env python3
"""
Layer-wise probing and ablation for BERT (POS tagging + STS-B semantic similarity).

Designed for Kaggle GPU: PyTorch + Hugging Face transformers/datasets + sklearn + matplotlib.
BERT is frozen (no weight updates), but every probe training step still runs a full BERT forward on
the batch; that forward dominates wall time. Only the small linear probe is trained (backward
through the probe, not through BERT).

BERT is not trained, but each linear probe is: it maps hidden states to labels on a finite training
set. More examples stabilize the probe weights and reported F1 / correlation; they do not teach
BERT anything. For qualitative layer comparisons, modest caps are often enough—use None only when
you care about tighter metrics or publication-style numbers.

Usage:
  python layerwise_bert_pos_stsb.py

Use FORCE_QUICK_DEBUG / QUICK_DEBUG=1 for smoke runs: only the constants below change (data size, batch, epochs, max length).
"""

from __future__ import annotations

import csv
import os
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download

import matplotlib

matplotlib.use("Agg")  # headless / notebook-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Fast smoke test: `QUICK_DEBUG=1 python layerwise_bert_pos_stsb.py`, or set FORCE_QUICK_DEBUG = True.
FORCE_QUICK_DEBUG = True
QUICK_DEBUG = FORCE_QUICK_DEBUG or (os.environ.get("QUICK_DEBUG", "0") == "1")
RANDOM_SEED = 42
MODEL_NAME = "bert-base-uncased"

# Hub repo ships raw .conllu files only (no dataset_infos / builder) — see load_pos_dataset().
POS_DATASET = "Qwerty66/UD_english-EWT"
POS_HUB_CONLLU_FILES = {
    "train": "en_ewt-ud-train.conllu",
    "validation": "en_ewt-ud-dev.conllu",
    "test": "en_ewt-ud-test.conllu",
}
GLUE_CONFIG = "glue"
GLUE_SUBSET = "stsb"

# Shorter sequences in quick mode = faster BERT forwards on CPU.
POS_MAX_LEN = 96 if QUICK_DEBUG else 128
STSB_MAX_LEN = 96 if QUICK_DEBUG else 128

POS_BATCH_SIZE = 12 if not QUICK_DEBUG else 16
STSB_BATCH_SIZE = 12 if not QUICK_DEBUG else 16

# Train/val row caps for *probe* fitting (maybe_subsample). None = use full HF split.
# Quick mode uses small caps; when QUICK_DEBUG is False you can still set integers here for faster
# exploratory runs (e.g. 4096) without touching BERT—only the linear head sees fewer rows.
POS_TRAIN_SAMPLES = 256 if QUICK_DEBUG else None
POS_EVAL_SAMPLES = 64 if QUICK_DEBUG else None
STSB_TRAIN_SAMPLES = 256 if QUICK_DEBUG else None
STSB_EVAL_SAMPLES = 64 if QUICK_DEBUG else None

PROBE_EPOCHS_POS = 1 if QUICK_DEBUG else 3
PROBE_EPOCHS_STSB = 1 if QUICK_DEBUG else 3
PROBE_LR = 2e-3

OUTPUT_DIR = "."

# Pooling for STS-B: "cls" or "mean"
STSB_POOLING = "mean"


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Dataset splits (HF)
# -----------------------------------------------------------------------------


def print_dataset_structure(ds: DatasetDict, name: str) -> None:
    print(f"\n=== {name} ===")
    print(ds)
    print("Splits:", list(ds.keys()))
    for split in ds.keys():
        print(f"  [{split}] columns: {ds[split].column_names}, n={len(ds[split])}")


def ensure_validation_split(
    ds: DatasetDict, train_key: str = "train", val_size: float = 0.1, seed: int = RANDOM_SEED
) -> DatasetDict:
    """If 'validation' is missing, split train into train/validation."""
    if "validation" in ds:
        return ds
    if train_key not in ds:
        raise KeyError(f"Expected '{train_key}' split in dataset")
    print(
        f"[{train_key}] No validation split found — creating validation "
        f"({val_size:.0%}) from '{train_key}' (seed={seed})."
    )
    split = ds[train_key].train_test_split(test_size=val_size, seed=seed)
    new_ds = DatasetDict(
        {
            train_key: split["train"],
            "validation": split["test"],
        }
    )
    # Preserve other splits (e.g. test)
    for k, v in ds.items():
        if k != train_key:
            new_ds[k] = v
    return new_ds


def normalize_split_names(ds: DatasetDict) -> DatasetDict:
    """Map common UD/HF names: dev -> validation."""
    ds = DatasetDict(dict(ds))
    if "validation" not in ds and "dev" in ds:
        ds["validation"] = ds["dev"]
        print("Renamed split 'dev' -> 'validation' for a consistent API.")
    return ds


def ensure_test_split(ds: DatasetDict) -> DatasetDict:
    """If 'test' is missing, reuse validation with a warning."""
    if "test" in ds:
        return ds
    if "validation" not in ds:
        raise KeyError("Need 'validation' or 'test' split")
    print("WARNING: No 'test' split — reusing 'validation' as test (for reporting only).")
    ds = DatasetDict(dict(ds))
    ds["test"] = ds["validation"]
    return ds


def maybe_subsample(ds: Dataset, n: Optional[int], seed: int = RANDOM_SEED) -> Dataset:
    if n is None or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


# -----------------------------------------------------------------------------
# POS: load UD English EWT (Hub has .conllu only — no auto load_dataset)
# -----------------------------------------------------------------------------


def parse_conllu_file(path: str) -> List[Dict[str, List[str]]]:
    """
    Parse a CoNLL-U file into sentence dicts with word-level tokens and UPOS tags.
    Skips multiword token lines (e.g. id 1-2) and empty nodes (e.g. 5.1).
    """
    sentences: List[Dict[str, List[str]]] = []
    cur_tok: List[str] = []
    cur_pos: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                if cur_tok:
                    sentences.append({"tokens": cur_tok, "upos": cur_pos})
                cur_tok, cur_pos = [], []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            tid = parts[0]
            if re.fullmatch(r"\d+-\d+", tid):
                continue
            if re.fullmatch(r"\d+\.\d+", tid):
                continue
            form, upos = parts[1], parts[3]
            cur_tok.append(form)
            cur_pos.append(upos)
    if cur_tok:
        sentences.append({"tokens": cur_tok, "upos": cur_pos})
    return sentences


def load_ud_ewt_from_hub_conllu(repo_id: str = POS_DATASET) -> DatasetDict:
    """Download official .conllu splits from the Hub and build a DatasetDict."""
    hf_token = os.environ.get("HF_TOKEN")  # optional; improves Hub rate limits when set
    paths: Dict[str, str] = {}
    for split, fname in POS_HUB_CONLLU_FILES.items():
        paths[split] = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            token=hf_token,
        )
    return DatasetDict(
        {
            "train": Dataset.from_list(parse_conllu_file(paths["train"])),
            "validation": Dataset.from_list(parse_conllu_file(paths["validation"])),
            "test": Dataset.from_list(parse_conllu_file(paths["test"])),
        }
    )


def load_pos_dataset() -> DatasetDict:
    """
    Prefer datasets' auto loader; fall back to CoNLL-U files (required for Qwerty66/UD_english-EWT on Hub).
    """
    try:
        return load_dataset(POS_DATASET)
    except Exception as e:
        if type(e).__name__ != "DataFilesNotFoundError" and "No (supported) data files found" not in str(e):
            raise
        print(
            f"Note: {POS_DATASET} has no packaged dataset script (raw .conllu only). "
            "Loading train/dev/test CoNLL-U files from the Hub."
        )
        return load_ud_ewt_from_hub_conllu(POS_DATASET)


# -----------------------------------------------------------------------------
# POS: token / label columns (UD HF datasets vary slightly)
# -----------------------------------------------------------------------------


def detect_pos_columns(example: Dict[str, Any]) -> Tuple[str, str]:
    keys = set(example.keys())
    # Common patterns
    if "tokens" in keys and "upos" in keys:
        return "tokens", "upos"
    if "tokens" in keys and "pos_tags" in keys:
        return "tokens", "pos_tags"
    if "tokens" in keys and "xpos" in keys:
        return "tokens", "xpos"
    raise ValueError(f"Could not detect token/POS columns. Keys: {sorted(keys)}")


def build_label_maps(
    dataset: Dataset, tokens_key: str, labels_key: str
) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_set: List[str] = []
    seen = set()
    for ex in tqdm(dataset, desc="Scanning POS labels"):
        for t in ex[labels_key]:
            s = str(t)
            if s not in seen:
                seen.add(s)
                label_set.append(s)
    label_set = sorted(label_set)
    label2id = {l: i for i, l in enumerate(label_set)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def align_labels_with_tokenizer(
    tokenizer: AutoTokenizer,
    tokens: List[str],
    labels: List[str],
    label2id: Dict[str, int],
    max_length: int,
) -> BatchEncoding:
    """
    Tokenize with is_split_into_words=True; align labels:
    - first subword gets gold label
    - other subwords and special tokens = -100
    """
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    word_ids = enc.word_ids(batch_index=0)
    label_ids: List[int] = []
    previous_word_idx: Optional[int] = None
    for wid in word_ids:
        if wid is None:
            label_ids.append(-100)
            previous_word_idx = None
            continue
        if wid != previous_word_idx:
            lab = labels[wid]
            label_ids.append(label2id[str(lab)])
            previous_word_idx = wid
        else:
            label_ids.append(-100)

    enc["labels"] = torch.tensor(label_ids, dtype=torch.long).unsqueeze(0)
    return enc


def preprocess_pos_dataset(
    raw: Dataset,
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    tokens_key: str,
    labels_key: str,
    max_length: int,
) -> Dataset:
    def _map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        tokens = ex[tokens_key]
        labels = ex[labels_key]
        enc = align_labels_with_tokenizer(tokenizer, tokens, labels, label2id, max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0).tolist(),
            "attention_mask": enc["attention_mask"].squeeze(0).tolist(),
            "token_type_ids": enc["token_type_ids"].squeeze(0).tolist()
            if "token_type_ids" in enc
            else [0] * max_length,
            "labels": enc["labels"].squeeze(0).tolist(),
        }

    return raw.map(_map_fn, remove_columns=raw.column_names)


# -----------------------------------------------------------------------------
# STS-B preprocessing
# -----------------------------------------------------------------------------


def preprocess_stsb_dataset(raw: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        enc = tokenizer(
            ex["sentence1"],
            ex["sentence2"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Normalize label to [0, 1] for stability (original 0–5)
        y = float(ex["label"]) / 5.0
        enc["labels"] = y
        return enc

    cols_to_remove = [c for c in raw.column_names if c not in ("sentence1", "sentence2", "label")]
    return raw.map(_map_fn, remove_columns=cols_to_remove)


# -----------------------------------------------------------------------------
# Collate
# -----------------------------------------------------------------------------


def collate_pos(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "token_type_ids": torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def collate_stsb(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    enc = {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "token_type_ids": torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long),
    }
    if "labels" in batch[0]:
        enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.float32)
    return enc


# -----------------------------------------------------------------------------
# BERT forward helpers (frozen) + ablation hook
# -----------------------------------------------------------------------------


def freeze_bert(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


@torch.no_grad()
def forward_hidden_states(
    model: AutoModel, batch: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, ...]:
    out = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        token_type_ids=batch["token_type_ids"].to(device),
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states: tuple length 13 for bert-base: embeddings + 12 layers
    hs = out.hidden_states
    assert hs is not None
    return hs


def make_layer_ablation_hook(layer_idx: int) -> Callable[..., None]:
    """
    Zero-out the output of encoder layer `layer_idx` (0..num_layers-1).
    """

    def hook(_module: nn.Module, _inp: Any, output: Any) -> None:
        if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
            output[0].zero_()
        elif torch.is_tensor(output):
            output.zero_()

    return hook


def register_ablation_hook(model: AutoModel, layer_idx: int) -> torch.utils.hooks.RemovableHandle:
    layer = model.encoder.layer[layer_idx]
    return layer.register_forward_hook(make_layer_ablation_hook(layer_idx))


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.size < 2:
        return float("nan")
    xm = x.mean()
    ym = y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    if den == 0:
        return float("nan")
    return float(num / den)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman via Pearson on rank-transformed values (no scipy)."""
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return pearson_r(rx, ry)


# -----------------------------------------------------------------------------
# Probes
# -----------------------------------------------------------------------------


class TokenProbe(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)


class RegProbe(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(pooled).squeeze(-1)


def pool_sentence_pair(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    hidden: (B, L, H)
    For sentence A (first segment): use tokens where token_type_ids==0 or first segment until sep.
    We use simple masked mean over full sequence (standard for STS) or CLS.
    """
    if pooling == "cls":
        return hidden[:, 0, :]
    # mean over tokens (mask padding)
    mask = attention_mask.unsqueeze(-1).to(device=hidden.device, dtype=hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def train_pos_probe_for_layer(
    layer_tensor_idx: int,
    model: AutoModel,
    probe: TokenProbe,
    train_loader: DataLoader,
    device: torch.device,
    num_labels: int,
    epochs: int,
    lr: float,
) -> None:
    probe.train()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    # BERT under no_grad: no BERT grads, but full encoder forward still runs (main cost per batch).
    for _ in range(epochs):
        for batch in train_loader:
            with torch.no_grad():
                hs = forward_hidden_states(model, batch, device)
                h = hs[layer_tensor_idx]  # (B,L,H)
            logits = probe(h)
            loss = loss_fn(logits.view(-1, num_labels), batch["labels"].to(device).view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def eval_pos_probe(
    layer_tensor_idx: int,
    model: AutoModel,
    probe: TokenProbe,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
    ablation_layer: Optional[int] = None,
) -> Tuple[float, float]:
    probe.eval()
    preds_all: List[int] = []
    gold_all: List[int] = []
    handle = None
    if ablation_layer is not None:
        handle = register_ablation_hook(model, ablation_layer)
    try:
        for batch in loader:
            hs = forward_hidden_states(model, batch, device)
            h = hs[layer_tensor_idx]
            logits = probe(h)
            pred = logits.argmax(dim=-1).cpu().numpy().ravel()
            gold = batch["labels"].cpu().numpy().ravel()
            mask = gold != -100
            preds_all.extend(pred[mask].tolist())
            gold_all.extend(gold[mask].tolist())
    finally:
        if handle is not None:
            handle.remove()

    acc = accuracy_score(gold_all, preds_all) if gold_all else 0.0
    f1 = f1_score(gold_all, preds_all, average="macro", zero_division=0) if gold_all else 0.0
    return float(acc), float(f1)


def train_stsb_probe_for_layer(
    layer_tensor_idx: int,
    model: AutoModel,
    probe: RegProbe,
    train_loader: DataLoader,
    device: torch.device,
    pooling: str,
    epochs: int,
    lr: float,
) -> None:
    probe.train()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # Same as POS: frozen BERT forward every batch dominates time.
    for _ in range(epochs):
        for batch in train_loader:
            with torch.no_grad():
                hs = forward_hidden_states(model, batch, device)
                h = hs[layer_tensor_idx]
                pooled = pool_sentence_pair(h, batch["attention_mask"], pooling)
            pred = probe(pooled)
            loss = loss_fn(pred, batch["labels"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def eval_stsb_probe(
    layer_tensor_idx: int,
    model: AutoModel,
    probe: RegProbe,
    loader: DataLoader,
    device: torch.device,
    pooling: str,
    ablation_layer: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Returns Pearson, Spearman, MSE (labels in [0,1])."""
    probe.eval()
    preds: List[float] = []
    golds: List[float] = []
    handle = None
    if ablation_layer is not None:
        handle = register_ablation_hook(model, ablation_layer)
    try:
        for batch in loader:
            hs = forward_hidden_states(model, batch, device)
            h = hs[layer_tensor_idx]
            pooled = pool_sentence_pair(h, batch["attention_mask"], pooling)
            pred = probe(pooled)
            preds.extend(pred.detach().cpu().numpy().tolist())
            golds.extend(batch["labels"].numpy().tolist())
    finally:
        if handle is not None:
            handle.remove()

    y_hat = np.array(preds, dtype=np.float64)
    y = np.array(golds, dtype=np.float64)
    mse = float(mean_squared_error(y, y_hat)) if y.size else float("nan")
    pr = pearson_r(y_hat, y)
    sp = spearman_rho(y_hat, y)
    return pr, sp, mse


def save_rows_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved: {path}")


def plot_layer_curve(
    xs: List[int],
    ys: Dict[str, List[float]],
    title: str,
    ylabel: str,
    path: str,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for name, series in ys.items():
        plt.plot(xs, series, marker="o", label=name)
    plt.xlabel("Encoder layer index (0 = bottom)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved figure: {path}")


def print_model_info(model: AutoModel) -> None:
    n_layers = model.config.num_hidden_layers
    h = model.config.hidden_size
    print(f"BERT layers (encoder): {n_layers}, hidden size: {h}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,} (frozen)")


def run() -> None:
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")
    print(
        "Effective limits (None = use full split): "
        f"POS train/val caps={POS_TRAIN_SAMPLES}/{POS_EVAL_SAMPLES}, "
        f"STS-B train/val caps={STSB_TRAIN_SAMPLES}/{STSB_EVAL_SAMPLES}, "
        f"batch POS/STSB={POS_BATCH_SIZE}/{STSB_BATCH_SIZE}, "
        f"epochs POS/STSB={PROBE_EPOCHS_POS}/{PROBE_EPOCHS_STSB}, "
        f"max_len POS/STSB={POS_MAX_LEN}/{STSB_MAX_LEN}."
    )

    # --- Model + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # disable dropout when using frozen BERT representations
    freeze_bert(model)
    print_model_info(model)
    hidden_size = model.config.hidden_size
    n_enc_layers = model.config.num_hidden_layers
    # hidden_states[0]=embeddings; hidden_states[k]=output after encoder layer k-1 (k=1..L)

    # --- POS data ---
    pos_raw = load_pos_dataset()
    print_dataset_structure(pos_raw, "UD English EWT (POS)")
    pos_raw = normalize_split_names(pos_raw)
    pos_raw = ensure_validation_split(pos_raw, train_key="train", val_size=0.1, seed=RANDOM_SEED)
    pos_raw = ensure_test_split(pos_raw)

    ex0 = pos_raw["train"][0]
    tokens_key, labels_key = detect_pos_columns(ex0)
    print(f"POS columns: tokens='{tokens_key}', labels='{labels_key}'")

    pos_train = maybe_subsample(pos_raw["train"], POS_TRAIN_SAMPLES)
    pos_val = maybe_subsample(pos_raw["validation"], POS_EVAL_SAMPLES)
    # Full train split for label inventory (avoids missing labels when subsampling).
    label2id, _ = build_label_maps(pos_raw["train"], tokens_key, labels_key)
    num_labels = len(label2id)
    print(f"num_labels (POS): {num_labels}")

    pos_train_tok = preprocess_pos_dataset(pos_train, tokenizer, label2id, tokens_key, labels_key, POS_MAX_LEN)
    pos_val_tok = preprocess_pos_dataset(pos_val, tokenizer, label2id, tokens_key, labels_key, POS_MAX_LEN)

    pos_train_loader = DataLoader(
        pos_train_tok,
        batch_size=POS_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pos,
        num_workers=0,
    )
    pos_val_loader = DataLoader(
        pos_val_tok,
        batch_size=POS_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_pos,
        num_workers=0,
    )

    # --- STS-B data ---
    glue = load_dataset(GLUE_CONFIG, GLUE_SUBSET)
    print_dataset_structure(glue, "GLUE STS-B")
    glue = normalize_split_names(glue)
    glue = ensure_validation_split(glue, train_key="train", val_size=0.1, seed=RANDOM_SEED)
    glue = ensure_test_split(glue)

    stsb_train = maybe_subsample(glue["train"], STSB_TRAIN_SAMPLES)
    stsb_val = maybe_subsample(glue["validation"], STSB_EVAL_SAMPLES)

    stsb_train_ds = preprocess_stsb_dataset(stsb_train, tokenizer, STSB_MAX_LEN)
    stsb_val_ds = preprocess_stsb_dataset(stsb_val, tokenizer, STSB_MAX_LEN)

    stsb_train_loader = DataLoader(
        stsb_train_ds,
        batch_size=STSB_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_stsb,
        num_workers=0,
    )
    stsb_val_loader = DataLoader(
        stsb_val_ds,
        batch_size=STSB_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_stsb,
        num_workers=0,
    )

    # Debug shapes
    pb = next(iter(pos_train_loader))
    sb = next(iter(stsb_train_loader))
    print("[debug] POS batch input_ids:", tuple(pb["input_ids"].shape))
    print("[debug] STS-B batch input_ids:", tuple(sb["input_ids"].shape))

    layer_axis = list(range(n_enc_layers))

    # --- Probing: train + eval per encoder layer ---
    pos_probe_rows: List[Dict[str, Any]] = []
    stsb_probe_rows: List[Dict[str, Any]] = []

    pos_f1_by_layer: List[float] = []
    pos_acc_by_layer: List[float] = []
    stsb_pearson_by_layer: List[float] = []
    stsb_spearman_by_layer: List[float] = []
    stsb_mse_by_layer: List[float] = []

    pos_probes: List[TokenProbe] = []
    stsb_probes: List[RegProbe] = []

    for enc_i in tqdm(range(n_enc_layers), desc="Probing per layer"):
        tensor_idx = enc_i + 1  # hidden_states after encoder layer enc_i

        # POS
        pos_probe = TokenProbe(hidden_size, num_labels).to(device)
        train_pos_probe_for_layer(
            tensor_idx,
            model,
            pos_probe,
            pos_train_loader,
            device,
            num_labels,
            PROBE_EPOCHS_POS,
            PROBE_LR,
        )
        pos_probes.append(pos_probe)
        acc, f1 = eval_pos_probe(tensor_idx, model, pos_probe, pos_val_loader, device, num_labels)
        pos_acc_by_layer.append(acc)
        pos_f1_by_layer.append(f1)
        pos_probe_rows.append(
            {
                "layer": enc_i,
                "accuracy": acc,
                "macro_f1": f1,
            }
        )

        # STS-B
        reg_probe = RegProbe(hidden_size).to(device)
        train_stsb_probe_for_layer(
            tensor_idx,
            model,
            reg_probe,
            stsb_train_loader,
            device,
            STSB_POOLING,
            PROBE_EPOCHS_STSB,
            PROBE_LR,
        )
        stsb_probes.append(reg_probe)
        pr, sp, mse = eval_stsb_probe(
            tensor_idx, model, reg_probe, stsb_val_loader, device, STSB_POOLING
        )
        stsb_pearson_by_layer.append(pr)
        stsb_spearman_by_layer.append(sp)
        stsb_mse_by_layer.append(mse)
        stsb_probe_rows.append(
            {
                "layer": enc_i,
                "pearson": pr,
                "spearman": sp,
                "mse": mse,
                "pooling": STSB_POOLING,
            }
        )

    # --- Ablation: reuse probes trained above; ablate encoder layer k, evaluate same-layer probe ---
    pos_ablation_rows: List[Dict[str, Any]] = []
    stsb_ablation_rows: List[Dict[str, Any]] = []

    pos_ablation_drop_f1: List[float] = []
    stsb_ablation_drop_pearson: List[float] = []

    for enc_i in tqdm(range(n_enc_layers), desc="Ablation per layer"):
        tensor_idx = enc_i + 1
        pos_probe = pos_probes[enc_i]
        reg_probe = stsb_probes[enc_i]

        base_acc, base_f1 = eval_pos_probe(tensor_idx, model, pos_probe, pos_val_loader, device, num_labels)
        ab_acc, ab_f1 = eval_pos_probe(
            tensor_idx,
            model,
            pos_probe,
            pos_val_loader,
            device,
            num_labels,
            ablation_layer=enc_i,
        )
        pos_ablation_rows.append(
            {
                "probe_layer": enc_i,
                "ablated_layer": enc_i,
                "accuracy_baseline": base_acc,
                "accuracy_ablated": ab_acc,
                "f1_baseline": base_f1,
                "f1_ablated": ab_f1,
                "accuracy_drop": base_acc - ab_acc,
                "f1_drop": base_f1 - ab_f1,
            }
        )
        pos_ablation_drop_f1.append(base_f1 - ab_f1)

        base_pr, base_sp, base_mse = eval_stsb_probe(
            tensor_idx, model, reg_probe, stsb_val_loader, device, STSB_POOLING
        )
        ab_pr, ab_sp, ab_mse = eval_stsb_probe(
            tensor_idx,
            model,
            reg_probe,
            stsb_val_loader,
            device,
            STSB_POOLING,
            ablation_layer=enc_i,
        )
        stsb_ablation_rows.append(
            {
                "probe_layer": enc_i,
                "ablated_layer": enc_i,
                "pearson_baseline": base_pr,
                "pearson_ablated": ab_pr,
                "spearman_baseline": base_sp,
                "spearman_ablated": ab_sp,
                "mse_baseline": base_mse,
                "mse_ablated": ab_mse,
                "pearson_drop": base_pr - ab_pr,
                "mse_increase": ab_mse - base_mse,
            }
        )
        stsb_ablation_drop_pearson.append(base_pr - ab_pr)

    # --- Save CSV ---
    save_rows_csv(
        os.path.join(OUTPUT_DIR, "pos_probe_results.csv"),
        ["layer", "accuracy", "macro_f1"],
        pos_probe_rows,
    )
    save_rows_csv(
        os.path.join(OUTPUT_DIR, "stsb_probe_results.csv"),
        ["layer", "pearson", "spearman", "mse", "pooling"],
        stsb_probe_rows,
    )
    save_rows_csv(
        os.path.join(OUTPUT_DIR, "pos_ablation_results.csv"),
        [
            "probe_layer",
            "ablated_layer",
            "accuracy_baseline",
            "accuracy_ablated",
            "f1_baseline",
            "f1_ablated",
            "accuracy_drop",
            "f1_drop",
        ],
        pos_ablation_rows,
    )
    save_rows_csv(
        os.path.join(OUTPUT_DIR, "stsb_ablation_results.csv"),
        [
            "probe_layer",
            "ablated_layer",
            "pearson_baseline",
            "pearson_ablated",
            "spearman_baseline",
            "spearman_ablated",
            "mse_baseline",
            "mse_ablated",
            "pearson_drop",
            "mse_increase",
        ],
        stsb_ablation_rows,
    )

    # --- Plots ---
    plot_layer_curve(
        layer_axis,
        {"macro F1": pos_f1_by_layer},
        "POS probing: macro F1 by layer",
        "Macro F1 (validation)",
        os.path.join(OUTPUT_DIR, "pos_layer_vs_f1.png"),
    )
    plot_layer_curve(
        layer_axis,
        {"Pearson r": stsb_pearson_by_layer},
        "STS-B probing: Pearson correlation by layer",
        "Pearson r (validation, labels scaled to [0,1])",
        os.path.join(OUTPUT_DIR, "stsb_layer_vs_pearson.png"),
    )
    plot_layer_curve(
        layer_axis,
        {"F1 drop (macro)": pos_ablation_drop_f1},
        "POS ablation: drop in macro F1 when layer output is zeroed",
        "F1(baseline) − F1(ablated)",
        os.path.join(OUTPUT_DIR, "pos_layer_vs_ablation_drop.png"),
    )
    plot_layer_curve(
        layer_axis,
        {"Pearson drop": stsb_ablation_drop_pearson},
        "STS-B ablation: drop in Pearson when layer output is zeroed",
        "Pearson(baseline) − Pearson(ablated)",
        os.path.join(OUTPUT_DIR, "stsb_layer_vs_ablation_drop.png"),
    )

    # Optional combined normalized overlay
    def _norm(a: List[float]) -> List[float]:
        a = np.array(a, dtype=np.float64)
        if a.size == 0 or np.nanmax(a) == np.nanmin(a):
            return [0.0 for _ in a]
        return ((a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a) + 1e-9)).tolist()

    plt.figure(figsize=(8, 4.5))
    plt.plot(layer_axis, _norm(pos_f1_by_layer), marker="o", label="POS macro F1 (norm)")
    plt.plot(layer_axis, _norm(stsb_pearson_by_layer), marker="s", label="STS-B Pearson (norm)")
    plt.xlabel("Encoder layer index (0 = bottom)")
    plt.ylabel("Normalized score")
    plt.title("Probing curves (min–max normalized per task)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p_combined = os.path.join(OUTPUT_DIR, "combined_probing_normalized.png")
    plt.savefig(p_combined, dpi=150)
    plt.close()
    print(f"Saved figure: {p_combined}")

    # --- Summary ---
    best_pos = int(np.argmax(pos_f1_by_layer))
    best_stsb = int(np.argmax(stsb_pearson_by_layer))
    worst_pos_ab = int(np.argmax(pos_ablation_drop_f1))
    worst_stsb_ab = int(np.argmax(stsb_ablation_drop_pearson))

    print("\n=== RESULTS SUMMARY ===")
    print(f"Best POS macro F1 layer (validation): {best_pos} (F1={pos_f1_by_layer[best_pos]:.4f})")
    print(f"Best STS-B Pearson layer (validation): {best_stsb} (r={stsb_pearson_by_layer[best_stsb]:.4f})")
    print(f"Largest POS ablation F1 drop at layer: {worst_pos_ab} (drop={pos_ablation_drop_f1[worst_pos_ab]:.4f})")
    print(f"Largest STS-B ablation Pearson drop at layer: {worst_stsb_ab} (drop={stsb_ablation_drop_pearson[worst_stsb_ab]:.4f})")

    print("\n=== INTERPRETATION (safe wording) ===")
    print(
        f"POS tagging probe quality (macro F1) is highest around layer {best_pos}, suggesting that "
        "representations in that region encode token-level syntactic information useful for the task."
    )
    print(
        f"Semantic similarity probe quality (Pearson on STS-B) peaks near layer {best_stsb}, suggesting that "
        "deeper layers better support sentence-level semantic similarity under this setup."
    )
    print(
        "If earlier layers peak for POS and later layers for STS-B, that pattern would be consistent with the "
        "hypothesis that lower/middle layers carry more local syntactic cues while deeper layers carry more "
        "information aligned with semantic similarity — without claiming that any layer 'understands' text."
    )


if __name__ == "__main__":
    run()
