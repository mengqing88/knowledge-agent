"""Run the existing pinned BGE worker in its isolated Python environment."""
import hashlib
import json
import os
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from app.core.config import ROOT
from app.core.errors import BudgetExhausted, ProviderError
from app.knowledge.context_selection import validated_order

WORKER = ROOT / "scripts" / "models" / "local_rerank.py"


def _stop_process(process):
    # Windows venv python.exe launches a second interpreter. Kill that owned
    # process tree as well, otherwise a cancelled GPU inference keeps running.
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    if process.poll() is None:
        process.kill()


def request_digest(row):
    return hashlib.sha256(json.dumps({"question": row["question"], "candidates": row["candidates"]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def local_model_identity(config):
    manifest = Path(config["local_reranker_model"]) / "download-manifest.json"
    return {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
            "worker_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest()}


@lru_cache(maxsize=2)
def _verify_files(model_path, manifest_bytes, file_stats):
    # Stat identities invalidate the cache after a model file is replaced.
    model = Path(model_path).resolve()
    manifest = json.loads(manifest_bytes)
    if manifest.get("model") != "BAAI/bge-reranker-v2-m3" or not manifest.get("files"):
        raise ValueError("Unexpected model manifest")
    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "sentencepiece.bpe.model"}
    if set(manifest["files"]) != required:
        raise ValueError("Incomplete model manifest")
    for name, expected in manifest["files"].items():
        path = model / name
        if not path.resolve().is_relative_to(model) or path.stat().st_size != expected["bytes"]:
            raise ValueError("Model size or location differs")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                raise ValueError("Model file hash differs")


def validate_local_model(config):
    try:
        python, model = Path(config["local_reranker_python"]), Path(config["local_reranker_model"])
        if not python.is_file():
            raise ValueError("Missing isolated Python")
        identity = local_model_identity(config)
        if identity != config.get("retrieval_model_identity", identity) or not identity["manifest_sha256"]:
            raise ValueError("Model identity changed")
        manifest_bytes = (model / "download-manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        stats = tuple((name, (model / name).stat().st_size, (model / name).stat().st_mtime_ns)
                      for name in sorted(manifest["files"]))
        _verify_files(str(model.resolve()), manifest_bytes, stats)
        return identity
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ProviderError("本地 BGE 环境或模型不完整/已改变；请准备模型后新建问题，或显式回退 legacy",
                            "local_reranker_config") from exc


def rerank_local(requests, config, *, guard, deadline, observe=None):
    guard()
    identity = validate_local_model(config)
    guard()
    stop = min(deadline, time.monotonic() + config["local_reranker_timeout_seconds"])
    if time.monotonic() >= stop:
        raise BudgetExhausted("time")
    if (not 1 <= len(requests) <= 3 or len({r["id"] for r in requests}) != len(requests) or
            any(not 1 <= len(r["candidates"]) <= 300 for r in requests)):
        raise ProviderError("本地重排输入超出范围", "local_reranker_input")
    if observe:
        observe("local_rerank_started", {"candidate_count": len(requests[0]["candidates"]),
                "query_count": len(requests), **identity})
    root = ROOT / "tmp" / "runtime-rerank"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="request-", dir=root) as directory:
        folder = Path(directory)
        source, target = folder / "input.jsonl", folder / "scores.jsonl"
        source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests), encoding="utf-8")
        command = [config["local_reranker_python"], "-X", "utf8", str(WORKER), str(source), str(target),
                   "--model", config["local_reranker_model"], "--device", config["local_reranker_device"],
                   "--precision", "auto", "--batch-size", "2"]
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false"}
        started = time.monotonic()
        with (folder / "worker.log").open("wb") as log:
            try:
                process = subprocess.Popen(command, stdout=log, stderr=log, env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except OSError as exc:
                raise ProviderError("本地 BGE 进程无法启动", "local_reranker_config") from exc
            try:
                while True:
                    guard()
                    if time.monotonic() >= stop:
                        raise BudgetExhausted("time")
                    try:
                        process.wait(timeout=.2)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                guard()
                if process.returncode != 0:
                    raise ProviderError("本地 BGE 执行失败，请检查独立环境及可用内存", "local_reranker_failed")
            finally:
                if process.poll() is None:
                    _stop_process(process)
                process.wait(timeout=5)
        try:
            results = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            if len(results) != len(requests) or {r["id"] for r in results} != {r["id"] for r in requests}:
                raise ValueError("Missing inference result")
            by_id = {r["id"]: r for r in results}
            orders = []
            for request in requests:
                result = by_id[request["id"]]
                if result.get("model_manifest_sha256") != identity["manifest_sha256"]:
                    raise ValueError("Model identity differs")
                orders.append(validated_order(result, request, request_digest(request)))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError("本地 BGE 结果未通过完整性校验", "local_reranker_result") from exc
        if local_model_identity(config) != identity:
            raise ProviderError("本地 BGE 模型在推理中改变", "local_reranker_config")
        if observe:
            observe("local_rerank_finished", {"seconds": round(time.monotonic() - started, 3),
                "query_count": len(requests), "pair_count": sum(len(r["candidates"]) for r in requests),
                "truncated_pairs": sum(r["truncated_pairs"] for r in results),
                "request_sha256": [request_digest(r) for r in requests],
                "device": results[0]["device"], "precision": results[0]["precision"], **identity})
        return orders
