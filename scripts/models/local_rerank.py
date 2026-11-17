"""Local CrossEncoder experiment in a separate environment; JSONL protocol."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def request_digest(row):
    payload = {"question": row["question"], "candidates": row["candidates"]}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "float32", "float16"), default="float32")
    parser.add_argument("--direct-cuda-load", action="store_true",
                        help="Load weights directly onto CUDA with accelerate to reduce CPU memory")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output path; prior scores are immutable")
    if args.batch_size < 1:
        raise ValueError("A positive batch size is required")
    print("loading inference dependencies", flush=True)
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(4)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    precision = args.precision
    if precision == "auto":
        precision = "float16" if device == "cuda" else "float32"
    if precision == "float16" and device != "cuda":
        raise ValueError("float16 is supported only on CUDA")
    if args.direct_cuda_load and device != "cuda":
        raise ValueError("Direct CUDA loading requires a CUDA device")
    dtype = torch.float16 if precision == "float16" else torch.float32
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    print(f"loading model on {device} in {precision}", flush=True)
    loading_options = {"device_map": "cuda"} if args.direct_cuda_load else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=dtype, **loading_options)
    if not args.direct_cuda_load:
        model = model.to(device)
    model.eval()
    load_seconds = time.monotonic() - started
    print(f"loaded in {load_seconds:.1f}s", flush=True)
    manifest_path = args.model / "download-manifest.json"
    model_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None
    with args.output.open("x", encoding="utf-8") as output, args.input.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            candidate_ids = [candidate["id"] for candidate in row["candidates"]]
            if (not row["question"].strip() or not candidate_ids or
                    len(candidate_ids) != len(set(candidate_ids))):
                raise ValueError("A question and unique candidates are required")
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
            result = {"id": row["id"], "request_sha256": request_digest(row),
                      "model_manifest_sha256": model_manifest_sha256,
                      "scores": sorted(scores, key=lambda x: (-x["score"], x["id"])),
                      "seconds": round(time.monotonic() - started, 4), "truncated_pairs": truncated,
                      "load_seconds": round(load_seconds, 3), "torch": torch.__version__,
                      "transformers": transformers.__version__, "max_length": 1024, "device": device,
                      "precision": precision, "direct_cuda_load": args.direct_cuda_load}
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(row["id"], result["seconds"], flush=True)


if __name__ == "__main__":
    main()
