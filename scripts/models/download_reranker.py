"""Download only pinned inference files; verify their published SHA-256 or Git blob IDs."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "sentencepiece.bpe.model"}
DESTINATION = Path(__file__).resolve().parents[2] / "tmp" / "rag-models" / "bge-reranker-v2-m3"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true",
                                timeout=30) as response:
        metadata = json.load(response)
    manifest = {"model": MODEL, "revision": REVISION, "files": {}}
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        if name not in FILES:
            continue
        target = DESTINATION / name
        expected_size = entry["size"]
        expected_hash = entry.get("lfs", {}).get("sha256")
        if not target.exists() or target.stat().st_size != expected_size:
            part = target.with_suffix(target.suffix + ".partial")
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            print(f"downloading {name}: {expected_size / 1e6:.1f} MB", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as output:
                count, reported = 0, time.monotonic()
                while data := response.read(4 * 1024 * 1024):
                    output.write(data)
                    count += len(data)
                    if time.monotonic() - reported >= 15:
                        print(f"  {count / 1e6:.0f}/{expected_size / 1e6:.0f} MB", flush=True)
                        reported = time.monotonic()
            if part.stat().st_size != expected_size:
                raise ValueError(f"Incomplete download: {name}")
            part.replace(target)
        sha = hashlib.sha256()
        git = hashlib.sha1(f"blob {expected_size}\0".encode())
        with target.open("rb") as data:
            while chunk := data.read(4 * 1024 * 1024):
                sha.update(chunk)
                git.update(chunk)
        if (expected_hash and sha.hexdigest() != expected_hash) or (
                not expected_hash and git.hexdigest() != entry["blobId"]):
            raise ValueError(f"Hash mismatch: {name}")
        manifest["files"][name] = {"bytes": expected_size, "sha256": sha.hexdigest()}
    (DESTINATION / "download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("verified", DESTINATION, flush=True)


if __name__ == "__main__":
    main()
