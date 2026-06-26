#!/usr/bin/env python3
"""
Continued-pretraining (CPT) of a Qwen3 base model on K, via unsloth QLoRA.

  uv run python scripts/train_cpt.py            # full run (reads configs/train.yaml)
  uv run python scripts/train_cpt.py --smoke    # 8-step end-to-end smoke test

Loads pre-packed fixed-length token blocks from data/packed/, trains LoRA adapters
(optionally embeddings), evaluates K perplexity on val with early stopping, and
saves adapters to train.output_dir.
"""
import argparse
import json
import os

import torch
from datasets import load_from_disk
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

# unsloth must be imported before transformers/trl for its patches to apply.
from unsloth import FastLanguageModel

from kcpt import paths
from kcpt.config import load_train_config, snapshot_to
from kcpt.metrics import perplexity_from_loss


class Collator:
    """Pre-packed fixed-length input_ids -> causal-LM batch (labels = input_ids)."""
    def __call__(self, feats):
        ids = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

class CPTTrainer(Trainer):
    """Adds eval_perplexity to the metrics dict *before* it's dispatched to the
    reporting integrations, so it reaches TensorBoard/W&B. A TrainerCallback's
    on_evaluate runs after self.log(), too late to be logged."""
    def log(self, logs, *args, **kwargs):
        if "eval_loss" in logs and "eval_perplexity" not in logs:
            logs["eval_perplexity"] = perplexity_from_loss(logs["eval_loss"])
        super().log(logs, *args, **kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test")
    args = ap.parse_args()
    cfg = load_train_config()
    m, lo, tr = cfg.model, cfg.lora, cfg.train

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m.name, max_seq_length=m.max_seq_length,
        load_in_4bit=m.load_in_4bit, dtype=m.dtype,
    )
    targets = list(lo.target_modules)
    if lo.train_embeddings:
        targets += ["embed_tokens", "lm_head"]
    model = FastLanguageModel.get_peft_model(
        model, r=lo.r, lora_alpha=lo.alpha, lora_dropout=lo.dropout,
        target_modules=targets, bias="none",
        use_gradient_checkpointing=lo.use_gradient_checkpointing,
        random_state=tr.seed,
    )

    train_ds = load_from_disk(os.path.join(paths.PACKED, "train"))
    val_ds = load_from_disk(os.path.join(paths.PACKED, "val"))

    stats = json.load(open(os.path.join(paths.PACKED, "stats.json")))
    if not args.smoke:
        assert stats.get("tokenizer") == m.name, (
            f"packed data tokenizer {stats.get('tokenizer')} != train model {m.name}; re-run pack_dataset")
        assert stats.get("seq_len") == m.max_seq_length, (
            f"packed seq_len {stats.get('seq_len')} != {m.max_seq_length}")

    if args.smoke:
        train_ds = train_ds.select(range(min(64, len(train_ds))))
        val_ds = val_ds.select(range(min(16, len(val_ds))))

    targs = TrainingArguments(
        output_dir=tr.output_dir,
        num_train_epochs=1 if args.smoke else tr.epochs,
        max_steps=8 if args.smoke else -1,
        per_device_train_batch_size=tr.per_device_batch_size,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,     # only eval_loss is needed (perplexity); avoids
                                       # materializing 2048 x 151k-vocab logits -> OOM
        gradient_accumulation_steps=1 if args.smoke else tr.grad_accum,
        learning_rate=tr.learning_rate,
        lr_scheduler_type=tr.lr_scheduler, warmup_ratio=tr.warmup_ratio,
        weight_decay=tr.weight_decay, max_grad_norm=tr.max_grad_norm,
        bf16=True, logging_steps=1 if args.smoke else tr.logging_steps,
        eval_strategy="steps", eval_steps=4 if args.smoke else tr.eval_steps,
        save_strategy="steps", save_steps=8 if args.smoke else tr.save_steps,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        seed=tr.seed, optim="adamw_8bit",
        # Experiment tracking (default tensorboard -> <output_dir>/runs/). Disabled
        # for --smoke so throwaway test runs don't litter the log dir.
        report_to="none" if args.smoke else tr.report_to,
        run_name=os.path.basename(tr.output_dir),
    )
    trainer = CPTTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tr.early_stopping_patience)],
    )
    print(f"VRAM before: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    trainer.train()
    if not args.smoke:
        model.save_pretrained(os.path.join(tr.output_dir, "adapter"))
        tokenizer.save_pretrained(os.path.join(tr.output_dir, "adapter"))
        # final test perplexity
        test_ds = load_from_disk(os.path.join(paths.PACKED, "test"))
        res = trainer.evaluate(test_ds)
        res["test_perplexity"] = perplexity_from_loss(res["eval_loss"])
        json.dump(res, open(os.path.join(tr.output_dir, "test_metrics.json"), "w"), indent=2)
        print("TEST:", res)
        snapshot_to(tr.output_dir, os.path.join(paths.ROOT, "configs", "train.yaml"),
                    os.path.join(paths.ROOT, "configs", "data.yaml"))
    print("done. peak VRAM:", f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB")

if __name__ == "__main__":
    main()
