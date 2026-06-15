"""Reusable fine-tuning experiment for function-level vulnerability detection.

A single ``run_experiment`` call fine-tunes a sequence-classification model on
one processed dataset split set, selects the best checkpoint by validation F1,
evaluates on the test split, and returns the full metric suite. It is model- and
dataset-agnostic so the same code serves the CodeBERT baseline, the VulBERTa
domain model, and the key-context-extraction variants.

Class imbalance is handled with class-weighted cross-entropy (weights derived
from the training split), which matters for the highly imbalanced DiverseVul.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from data.jsonl_dataset import PROCESSED_ROOT, load_processed
from metrics import make_compute_metrics, metrics_from_logits

TOKENIZED_ROOT = PROCESSED_ROOT.parent / "tokenized"


def _sanitize(name: str) -> str:
    """Turn a model id or path into a filesystem-safe token."""
    return re.sub(r"[^0-9A-Za-z._-]+", "__", name)


# Per-process tokenizer cache so multi-process map workers each build their own
# tokenizer instead of receiving an unpicklable one (VulBERTa holds ctypes
# pointers via libclang, which cannot cross a process boundary).
_WORKER_TOK: dict = {}

# Hard character cap applied before tokenization. VulBERTa's libclang tokenizer
# parses the whole string before truncating, so pathologically long functions
# (DiverseVul has some >250k chars) can stall a worker. This cap is far larger
# than max_len tokens' worth of code, so the first max_len tokens are unchanged.
_CHAR_CAP = 20_000


def _worker_tokenize(batch, model_name, trust_remote_code, text_field, max_len):
    """Tokenize one batch, lazily creating a per-process tokenizer.

    Args:
        batch: A batch dict from datasets.map.
        model_name: Model id used to build the tokenizer in this process.
        trust_remote_code: Whether to allow the model's custom tokenizer code.
        text_field: Record field holding the input code.
        max_len: Maximum sequence length.

    Returns:
        Dict with input_ids, attention_mask and labels.
    """
    key = (model_name, trust_remote_code)
    tok = _WORKER_TOK.get(key)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        _WORKER_TOK[key] = tok
    texts = [t[:_CHAR_CAP] for t in batch[text_field]]
    enc = tok(texts, truncation=True, max_length=max_len)
    enc["labels"] = [int(x) for x in batch["label"]]
    return enc


def get_tokenized(
    dataset: str,
    model_name: str,
    max_len: int,
    text_field: str,
    num_proc: int = 1,
    trust_remote_code: bool = False,
):
    """Tokenize a dataset's splits, caching the result to disk for reuse.

    The cache key includes the model, max length and text field, so different
    backbones or input variants do not collide. Because some tokenizers (e.g.
    VulBERTa's libclang pipeline) are slow, caching lets multiple seeds share a
    single tokenization pass.

    Args:
        dataset: Processed dataset name.
        model_name: Model id, used both to build the tokenizer and as a cache key.
        max_len: Maximum sequence length.
        text_field: Record field holding the input code.
        num_proc: Number of worker processes for the map step.
        trust_remote_code: Whether to allow the model's custom tokenizer code.

    Returns:
        Mapping from split name to a tokenized Dataset (columns input_ids,
        attention_mask, labels).
    """
    cache_dir = TOKENIZED_ROOT / f"{dataset}__{_sanitize(model_name)}__len{max_len}__{text_field}"

    def _valid(d: Path) -> bool:
        return all((d / s).exists() for s in ("train", "validation", "test"))

    def _load(d: Path) -> dict:
        return {s: load_from_disk(str(d / s)) for s in ("train", "validation", "test")}

    if _valid(cache_dir):
        return _load(cache_dir)

    # Build into a private temp dir, then atomically publish via os.replace so
    # that concurrent jobs sharing a cache key cannot read or write a partially
    # built cache. If another process publishes first, discard our temp copy.
    TOKENIZED_ROOT.mkdir(parents=True, exist_ok=True)
    raw = load_processed(dataset)
    fn_kwargs = {
        "model_name": model_name,
        "trust_remote_code": trust_remote_code,
        "text_field": text_field,
        "max_len": max_len,
    }
    tmp_dir = Path(tempfile.mkdtemp(prefix=cache_dir.name + ".tmp.", dir=TOKENIZED_ROOT))
    splits = {}
    for name, ds in raw.items():
        ds = ds.map(
            _worker_tokenize,
            batched=True,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc if num_proc > 1 else None,
            remove_columns=ds.column_names,
        )
        ds.save_to_disk(str(tmp_dir / name))
        splits[name] = ds

    if _valid(cache_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        try:
            os.replace(tmp_dir, cache_dir)
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return _load(cache_dir) if _valid(cache_dir) else splits


class PadCollator:
    """Dynamic padding collator that stores only the pad id, not a tokenizer.

    Avoiding a ``.tokenizer`` attribute keeps the Trainer from trying to
    serialize a non-serializable custom tokenizer (e.g. VulBERTa) into every
    checkpoint, and keeps the collator trivially picklable for dataloader workers.
    """

    def __init__(self, pad_token_id: int):
        """Store the padding token id.

        Args:
            pad_token_id: Token id used to right-pad input_ids.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        """Right-pad a batch of tokenized features to equal length.

        Args:
            features: List of dicts with torch tensors input_ids, attention_mask,
                labels.

        Returns:
            Batched dict of stacked, padded tensors.
        """
        max_len = max(f["input_ids"].size(0) for f in features)
        input_ids, attention, labels = [], [], []
        for f in features:
            ids, mask = f["input_ids"], f["attention_mask"]
            pad = max_len - ids.size(0)
            if pad:
                ids = torch.cat([ids, torch.full((pad,), self.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad, dtype=mask.dtype)])
            input_ids.append(ids)
            attention.append(mask)
            labels.append(f["labels"])
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention),
            "labels": torch.stack(labels),
        }


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-class focal loss with optional per-class alpha weighting.

    Focal loss down-weights well-classified examples by a factor of
    ``(1 - p_t) ** gamma``, focusing training on the hard, often minority-class
    examples. With ``gamma == 0`` it reduces to (weighted) cross-entropy.

    Args:
        logits: Class logits of shape (n, 2).
        labels: Ground-truth labels of shape (n,).
        gamma: Focusing parameter (>= 0); larger values focus more on hard cases.
        alpha: Optional per-class weight tensor of shape (2,).

    Returns:
        Scalar mean focal loss.
    """
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, labels.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()
    loss = -((1.0 - p_t) ** gamma) * logp_t
    if alpha is not None:
        loss = alpha.gather(0, labels) * loss
    return loss.mean()


class WeightedTrainer(Trainer):
    """Trainer variant supporting class-weighted CE or focal loss.

    The loss is selected by ``loss_type``:
      - ``"ce_weighted"``: cross-entropy weighted by inverse class frequency.
      - ``"focal"``: focal loss with ``focal_gamma`` and ``class_weights`` as the
        per-class alpha (set ``class_weights=None`` for unweighted focal).
      - ``"ce"``: plain unweighted cross-entropy (used with oversampling so the
        minority class is not corrected twice).
    """

    def __init__(
        self,
        *args,
        class_weights: torch.Tensor | None = None,
        loss_type: str = "ce_weighted",
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        """Store loss configuration, then defer to the base Trainer.

        Args:
            class_weights: Tensor of per-class weights, or None for unweighted.
            loss_type: One of "ce_weighted", "focal", "ce".
            focal_gamma: Focusing parameter for focal loss.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute the configured classification loss.

        Args:
            model: The model being trained.
            inputs: Batch dict; "labels" is consumed here.
            return_outputs: Whether to also return model outputs.
            **kwargs: Forward-compatible extras passed by the Trainer.

        Returns:
            The loss, or (loss, outputs) when ``return_outputs`` is True.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        if self.loss_type == "focal":
            loss = focal_loss(logits, labels, gamma=self.focal_gamma, alpha=weight)
        elif self.loss_type == "ce":
            loss = F.cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights normalized to mean 1.

    Args:
        labels: Training-split binary labels.

    Returns:
        Float tensor of shape (2,) with per-class weights.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (2 * counts.get(c, 1)) for c in (0, 1)]
    return torch.tensor(weights, dtype=torch.float)


def oversample_minority(train_ds, seed: int):
    """Duplicate minority-class rows so the training set is class-balanced.

    Random oversampling is an alternative to loss reweighting: instead of scaling
    the loss, the minority class is physically repeated until both classes are
    roughly equinumerous. It is applied only to the training split and should be
    paired with an unweighted loss to avoid correcting the imbalance twice.

    Args:
        train_ds: Tokenized training Dataset with a "labels" column.
        seed: Seed for the final shuffle.

    Returns:
        A new shuffled Dataset with the minority class oversampled to balance.
    """
    labels = list(train_ds["labels"])
    counts = Counter(labels)
    if len(counts) < 2:
        return train_ds
    majority = max(counts, key=counts.get)
    n_major = counts[majority]
    parts = [train_ds]
    for cls, n_cls in counts.items():
        if cls == majority or n_cls == 0:
            continue
        idx = [i for i, y in enumerate(labels) if y == cls]
        deficit = n_major - n_cls
        reps = [idx[i % len(idx)] for i in range(deficit)]
        parts.append(train_ds.select(reps))
    return concatenate_datasets(parts).shuffle(seed=seed)


def run_experiment(
    model_name: str,
    dataset: str,
    seed: int,
    output_dir: str | Path,
    text_field: str = "func",
    max_len: int = 512,
    epochs: int = 4,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    use_class_weights: bool = True,
    loss_type: str = "ce_weighted",
    focal_gamma: float = 2.0,
    sampler: str = "none",
    trust_remote_code: bool = False,
    tokenize_num_proc: int = 1,
) -> dict:
    """Fine-tune one model on one dataset for one seed and evaluate on test.

    Args:
        model_name: HuggingFace model id or local path.
        dataset: Processed dataset name ("devign" or "diversevul").
        seed: Random seed for reproducibility.
        output_dir: Directory for checkpoints and the metrics file.
        text_field: Record field holding the input code.
        max_len: Maximum tokenized sequence length.
        epochs: Number of training epochs.
        train_batch_size: Per-device training batch size.
        eval_batch_size: Per-device evaluation batch size.
        lr: Learning rate.
        weight_decay: Weight decay.
        use_class_weights: Whether to weight the loss by inverse class frequency.
        loss_type: Loss to optimize ("ce_weighted", "focal", or "ce").
        focal_gamma: Focusing parameter when ``loss_type == "focal"``.
        sampler: Training sampling strategy ("none" or "oversample"). When
            "oversample", the minority class is duplicated to balance the train
            split; pair it with an unweighted loss (handled automatically below).
        trust_remote_code: Pass through for models with custom code (e.g. VulBERTa).
        tokenize_num_proc: Worker processes for the (cached) tokenization step.

    Returns:
        Dict of test-split metrics, also written to ``output_dir/metrics.json``.
    """
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tok_splits = get_tokenized(
        dataset, model_name, max_len, text_field,
        num_proc=tokenize_num_proc, trust_remote_code=trust_remote_code,
    )

    # Random oversampling balances the train split by duplicating minority rows.
    # It is mutually exclusive with loss reweighting (to avoid double-correcting),
    # so when oversampling we drop class weights and use the requested/plain loss.
    if sampler == "oversample":
        tok_splits = dict(tok_splits)
        tok_splits["train"] = oversample_minority(tok_splits["train"], seed)
        use_class_weights = False
        if loss_type == "ce_weighted":
            loss_type = "ce"

    keep = ["input_ids", "attention_mask", "labels"]
    tokenized = {
        name: ds.with_format("torch", columns=[c for c in keep if c in ds.column_names])
        for name, ds in tok_splits.items()
    }

    class_weights = (
        compute_class_weights(tok_splits["train"]["labels"]) if use_class_weights else None
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, trust_remote_code=trust_remote_code
    )

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        bf16=torch.cuda.is_available(),
        seed=seed,
        dataloader_num_workers=4,
    )

    # processing_class is intentionally omitted and a tokenizer-free PadCollator
    # is used: some custom tokenizers (VulBERTa) cannot be serialized into a
    # checkpoint, and the tokenizer is not needed there.
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=PadCollator(pad_id),
        compute_metrics=make_compute_metrics(),
        class_weights=class_weights,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
    )

    trainer.train()

    pred = trainer.predict(tokenized["test"])
    test_metrics = metrics_from_logits(pred.predictions, pred.label_ids)
    test_metrics.update(
        {"model": model_name, "dataset": dataset, "seed": seed,
         "max_len": max_len, "epochs": epochs, "text_field": text_field,
         "loss_type": loss_type, "sampler": sampler}
    )
    (output_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    np.save(output_dir / "test_logits.npy", pred.predictions)
    np.save(output_dir / "test_labels.npy", pred.label_ids)
    return test_metrics
