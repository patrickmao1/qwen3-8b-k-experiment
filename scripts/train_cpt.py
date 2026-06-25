#!/usr/bin/env python3
"""
Continued-pretraining (CPT) of a Qwen3 base model on K, via unsloth QLoRA.

  .venv/bin/python scripts/train_cpt.py            # full run (reads configs/cpt.yaml)
  .venv/bin/python scripts/train_cpt.py --smoke    # 8-step end-to-end smoke test

Loads pre-packed fixed-length token blocks from data/packed/, trains LoRA adapters
(optionally embeddings), evaluates K perplexity on val with early stopping, and
saves adapters to train.output_dir.
"""
import argparse, json, math, os

# unsloth must be imported before transformers/trl for its patches to apply.
from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from transformers import (Trainer, TrainingArguments, EarlyStoppingCallback,
                          TrainerCallback)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKED = os.path.join(ROOT, "data", "packed")

def load_cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "configs", "cpt.yaml")))

class Collator:
    """Pre-packed fixed-length input_ids -> causal-LM batch (labels = input_ids)."""
    def __call__(self, feats):
        ids = torch.tensor([f["input_ids"] for f in feats], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics and "eval_loss" in metrics:
            metrics["eval_perplexity"] = math.exp(min(metrics["eval_loss"], 20))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test")
    args = ap.parse_args()
    cfg = load_cfg()
    m, lo, tr = cfg["model"], cfg["lora"], cfg["train"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m["name"], max_seq_length=m["max_seq_length"],
        load_in_4bit=m["load_in_4bit"], dtype=m["dtype"],
    )
    targets = list(lo["target_modules"])
    if lo.get("train_embeddings"):
        targets += ["embed_tokens", "lm_head"]
    model = FastLanguageModel.get_peft_model(
        model, r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
        target_modules=targets, bias="none",
        use_gradient_checkpointing=lo["use_gradient_checkpointing"],
        random_state=tr["seed"],
    )

    train_ds = load_from_disk(os.path.join(PACKED, "train"))
    val_ds = load_from_disk(os.path.join(PACKED, "val"))
    if args.smoke:
        train_ds = train_ds.select(range(min(64, len(train_ds))))
        val_ds = val_ds.select(range(min(16, len(val_ds))))

    targs = TrainingArguments(
        output_dir=tr["output_dir"],
        num_train_epochs=1 if args.smoke else tr["epochs"],
        max_steps=8 if args.smoke else -1,
        per_device_train_batch_size=tr["per_device_batch_size"],
        per_device_eval_batch_size=1,
        prediction_loss_only=True,     # only eval_loss is needed (perplexity); avoids
                                       # materializing 2048 x 151k-vocab logits -> OOM
        gradient_accumulation_steps=1 if args.smoke else tr["grad_accum"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler"], warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"], max_grad_norm=tr["max_grad_norm"],
        bf16=True, logging_steps=1 if args.smoke else tr["logging_steps"],
        eval_strategy="steps", eval_steps=4 if args.smoke else tr["eval_steps"],
        save_strategy="steps", save_steps=8 if args.smoke else tr["save_steps"],
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        seed=tr["seed"], report_to="none", optim="adamw_8bit",
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(),
        callbacks=[PerplexityCallback(),
                   EarlyStoppingCallback(early_stopping_patience=tr["early_stopping_patience"])],
    )
    print(f"VRAM before: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    trainer.train()
    if not args.smoke:
        model.save_pretrained(os.path.join(tr["output_dir"], "adapter"))
        tokenizer.save_pretrained(os.path.join(tr["output_dir"], "adapter"))
        # final test perplexity
        test_ds = load_from_disk(os.path.join(PACKED, "test"))
        res = trainer.evaluate(test_ds)
        res["test_perplexity"] = math.exp(min(res["eval_loss"], 20))
        json.dump(res, open(os.path.join(tr["output_dir"], "test_metrics.json"), "w"), indent=2)
        print("TEST:", res)
    print("done. peak VRAM:", f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB")

if __name__ == "__main__":
    main()
