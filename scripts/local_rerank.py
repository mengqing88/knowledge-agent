"""CPU CrossEncoder experiment in a separate environment; JSONL input/output protocol."""
import argparse
import json
import math
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(4)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, use_safetensors=True).to(device).eval()
    load_seconds = time.monotonic() - started
    with args.output.open("w", encoding="utf-8") as output:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            started, scores, truncated = time.monotonic(), [], 0
            for index in range(0, len(row["candidates"]), args.batch_size):
                candidates = row["candidates"][index:index + args.batch_size]
                pairs = [(row["question"], candidate["text"]) for candidate in candidates]
                raw = tokenizer(pairs, truncation=False, add_special_tokens=True)["input_ids"]
                truncated += sum(len(ids) > 1024 for ids in raw)
                tokens = tokenizer(pairs, padding=True, truncation="only_second", max_length=1024,
                                   return_tensors="pt")
                tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
                with torch.inference_mode():
                    values = model(**tokens).logits.view(-1).float().tolist()
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Non-finite reranker output")
                scores.extend({"id": candidate["id"], "score": score} for candidate, score in zip(candidates, values))
            result = {"id": row["id"], "scores": sorted(scores, key=lambda x: (-x["score"], x["id"])),
                      "seconds": round(time.monotonic() - started, 4), "truncated_pairs": truncated,
                      "load_seconds": round(load_seconds, 3), "torch": torch.__version__,
                      "transformers": transformers.__version__, "max_length": 1024, "device": device}
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(row["id"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
