"""Upload explicit fictional fixtures through the real API and create a fixed demo snapshot."""
import argparse
import getpass
import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--username", help="Login account; password is read without echo")
    parser.add_argument("--token-env", default="KB_EVALUATION_TOKEN", help="Environment variable holding a session token")
    args = parser.parse_args(argv)
    fixtures = ROOT / "fixtures"
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    with httpx.Client(base_url=args.api, timeout=30, trust_env=False) as client:
        token = os.environ.get(args.token_env)
        if token:
            client.headers["Authorization"] = f"Bearer {token}"
        else:
            username = args.username or input("Username: ")
            login = client.post("/api/auth/login", json={"username": username, "password": getpass.getpass("Password: ")})
            login.raise_for_status()
        corpora = client.get("/api/corpora")
        corpora.raise_for_status()
        existing = next((c for c in corpora.json() if c["name"] == "虚构教学论文集"), None)
        if existing:
            corpus_id = existing["id"]
        else:
            response = client.post("/api/corpora", json={"name": "虚构教学论文集"})
            response.raise_for_status()
            corpus_id = response.json()["id"]
        versions = []
        for paper in manifest["papers"]:
            with (fixtures / paper["filename"]).open("rb") as stream:
                response = client.post(f"/api/corpora/{corpus_id}/papers", data={"title": paper["title"]},
                                       files={"file": (paper["filename"], stream, "application/pdf")})
            response.raise_for_status()
            versions.append(response.json()["id"])
        started = time.monotonic()
        while True:
            states = []
            for version_id in versions:
                response = client.get(f"/api/paper-versions/{version_id}")
                response.raise_for_status()
                states.append(response.json())
            if any(s["status"] == "failed" for s in states):
                raise SystemExit("入库失败：" + "; ".join(s.get("error") or "" for s in states))
            if all(s["status"] == "ready" for s in states):
                break
            if time.monotonic()-started > 600:
                raise SystemExit("等待入库超时；请确认独立 worker 已启动")
            time.sleep(2)
        response = client.post(f"/api/corpora/{corpus_id}/snapshots", json={"paper_version_ids": versions})
        response.raise_for_status()
        snapshot_id = response.json()["id"]
    questions = [
        ("fact_a", "论文 A 在数据集 X 上，M 和基线的 F1 各是多少？", "supported"),
        ("compare_ab", "对比论文 A、B 的 F1，是否能说明 M 在所有场景都优于基线？", "supported"),
        ("latency_b", "论文 B 中 M 和基线的推理延迟各是多少，增加多少？", "supported"),
        ("missing_memory", "这些材料报告了 M 的 GPU 显存占用具体数值吗？", "insufficient"),
        ("partial", "M 的准确性、推理延迟和 GPU 显存代价分别怎样？", "partial"),
    ]
    rows = [{"question_id": qid, "question": question, "snapshot_id": snapshot_id,
             "expected_action": status, "fictional": True, "human_reviewed": False}
            for qid, question, status in questions]
    output = ROOT / "data" / "teaching_dev_questions.jsonl"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in rows)+"\n", encoding="utf-8")
    print(json.dumps({"corpus_id": corpus_id, "snapshot_id": snapshot_id, "fictional": True,
                      "development_questions": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
