"""LoRA fine-tuning of Qwen3.8-27B on A-OKVQA, sharded over several GPUs with FSDP.

Started by torchrun on every node (see job.sbatch). The 27B model in bfloat16 (54 GB)
doesn't fit one 48 GB L40S; FSDP splits its weights across the GPUs, and LoRA trains
small adapter matrices only, so optimizer state stays tiny.
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

MAX_SIDE = 448  # downscale images: fewer image tokens, less memory and traffic per step


def to_messages(question, answer=None):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


class Collator:
    """Builds chat-formatted batches; the loss is computed on the answer tokens only."""

    def __init__(self, processor):
        self.p = processor

    def __call__(self, rows):
        images, full, prompts = [], [], []
        for r in rows:
            img = r["images"][0].convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            images.append([img])
            turn = r["texts"][0]
            kw = dict(tokenize=False, chat_template_kwargs={"enable_thinking": False})
            full.append(self.p.apply_chat_template(to_messages(turn["user"], turn["assistant"]), **kw))
            prompts.append(self.p.apply_chat_template(to_messages(turn["user"]), add_generation_prompt=True, **kw))
        batch = self.p(text=full, images=images, return_tensors="pt", padding=True)
        prompt_lens = [len(x) for x in self.p(text=prompts, images=images, padding=False)["input_ids"]]
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i, n in enumerate(prompt_lens):
            labels[i, :n] = -100  # don't learn to reproduce the question
        batch["labels"] = labels
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4, help="per GPU")
    p.add_argument("--eval-examples", type=int, default=200)
    a = p.parse_args()

    processor = AutoProcessor.from_pretrained(a.model)
    processor.tokenizer.padding_side = "right"

    # A-OKVQA from HuggingFaceM4/the_cauldron: questions about COCO photos that need
    # common sense ("what is the man by the bags waiting for?"), 16.5k examples, 0.9 GB.
    ds = load_dataset("HuggingFaceM4/the_cauldron", "aokvqa", split="train")
    ds = ds.train_test_split(test_size=a.eval_examples, seed=0)

    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    if int(os.environ.get("RANK", 0)) == 0:
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(a.out),
        max_steps=a.max_steps,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=1,  # with FSDP every micro-batch re-gathers the weights: avoid
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=a.max_steps,      # evaluate at the end...
        eval_on_start=True,          # ...and before training, to compare
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        report_to="tensorboard",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "use_orig_params": True,
            "activation_checkpointing": True,
            "cpu_ram_efficient_loading": True,  # only rank 0 reads the full weights into memory
            "sync_module_states": True,
            "state_dict_type": "SHARDED_STATE_DICT",
        },
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
                      data_collator=Collator(processor))
    trainer.train()

    # Save the LoRA adapter (about 100 MB) as one ordinary file set.
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(str(a.out / "adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(a.out / "adapter"))


if __name__ == "__main__":
    main()
