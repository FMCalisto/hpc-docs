"""Describe a folder of images with a vision-language model, using vLLM on one GPU.

Runs inside the vLLM container (see job.sbatch):
    python3 caption.py --images DIR --out captions.jsonl
"""

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path

from vllm import LLM, SamplingParams

PROMPT = (
    "You are helping a robotics lab label its data. Look at the image and answer with JSON only: "
    '{"caption": one sentence describing the scene, '
    '"objects": list of the main objects, '
    '"environment": one of "indoor", "outdoor", "underwater", "aerial", "lab", "other", '
    '"hazards": list of anything a mobile robot should avoid}'
)


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--limit", type=int, default=0, help="only the first N images (0 = all)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=32768)
    a = p.parse_args()

    files = sorted(f for f in a.images.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if a.limit:
        files = files[: a.limit]
    if not files:
        raise SystemExit(f"No images in {a.images}")
    print(f"{len(files)} images, model {a.model}")

    t0 = time.time()
    llm = LLM(
        model=a.model,
        max_model_len=a.max_model_len,  # the model supports 262k; less leaves more memory for batching
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 1},
    )
    print(f"Model loaded in {time.time() - t0:.0f}s")

    # Qwen's recommended settings for answers without the thinking phase.
    params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=400)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(a.out, "w") as out:
        for i in range(0, len(files), a.batch):
            chunk = files[i : i + a.batch]
            conversations = [
                [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url(f)}},
                    {"type": "text", "text": PROMPT},
                ]}]
                for f in chunk
            ]
            results = llm.chat(
                conversations, params,
                chat_template_kwargs={"enable_thinking": False},  # straight answers, no reasoning trace
                use_tqdm=False,
            )
            for f, r in zip(chunk, results):
                text = r.outputs[0].text.strip().removeprefix("```json").removesuffix("```").strip()
                try:
                    answer = json.loads(text)
                except json.JSONDecodeError:
                    answer = {"raw": text}
                out.write(json.dumps({"image": str(f), **answer}) + "\n")
            done = i + len(chunk)
            print(f"{done}/{len(files)} images, {done / (time.time() - t0):.1f} images/s", flush=True)

    print("Wrote", a.out)


if __name__ == "__main__":
    main()
