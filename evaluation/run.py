"""Run selected configurations on frozen questions; preserve every failure."""
import argparse
import getpass
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

MODES = ("baseline_rag", "full_agent", "without_verifier")


def case_order(questions, modes):
    return [(number, question, mode) for number, question in enumerate(questions, 1) for mode in modes]


def evaluate_case(client, q, mode, timeout):
    record = {"question_id": q["question_id"], "mode": mode}
    start = time.monotonic()
    run_id = None
    try:
        response = client.post("/api/question-runs", headers={"Idempotency-Key": uuid4().hex}, json={
            "snapshot_id": q["snapshot_id"], "question": q["question"], "mode": mode})
        response.raise_for_status()
        run = response.json()
        run_id = record["run_id"] = run["id"]
        record["execution_manifest"] = run.get("config", {})
        while run["status"] not in ("completed", "failed", "cancelled", "review_pending", "rejected"):
            if time.monotonic()-start > timeout:
                raise TimeoutError("Evaluation timeout")
            time.sleep(1)
            response = client.get(f"/api/question-runs/{run_id}")
            response.raise_for_status()
            run = response.json()
        record["run"] = run
        if run["status"] in {"review_pending", "rejected"}:
            record["evaluation_blocked"] = "answer_human_review_required"
    except (httpx.HTTPError, TimeoutError) as exc:
        record["technical_error"] = type(exc).__name__
        if run_id:
            try:
                response = client.post(f"/api/question-runs/{run_id}/cancel")
                response.raise_for_status()
                record["cancellation"] = response.json().get("status", "requested")
            except httpx.HTTPError:
                record["cancellation"] = "unknown"
    finally:
        if run_id:
            try:
                trace = client.get(f"/api/question-runs/{run_id}/trace")
                trace.raise_for_status()
                record["trace"] = trace.json()
            except httpx.HTTPError as exc:
                record["trace_error"] = type(exc).__name__
        record["wall_seconds"] = round(time.monotonic()-start, 3)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("evaluation/runs"))
    parser.add_argument("--allow-dev", action="store_true", help="Allow a smaller dev set; never labels it frozen")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--review-source", choices=["human", "agent"], default="human")
    parser.add_argument("--snapshot-id", help="Bind a fictional development set to its uploaded teaching snapshot")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--username", help="Login account; password is read without echo")
    parser.add_argument("--token-env", default="KB_EVALUATION_TOKEN", help="Environment variable holding a session token")
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not questions or len({q["question_id"] for q in questions}) != len(questions):
        raise SystemExit("Empty set or duplicate question IDs")
    if len(set(args.modes)) != len(args.modes):
        raise SystemExit("Duplicate evaluation modes")
    if args.snapshot_id:
        if not args.allow_dev or any(q.get("fictional") is not True for q in questions):
            raise SystemExit("Snapshot rebinding is only allowed for fictional --allow-dev questions")
        questions = [{**q, "snapshot_id": args.snapshot_id} for q in questions]
    if any(q["snapshot_id"] == "BIND_TEACHING_SNAPSHOT" for q in questions):
        raise SystemExit("Bind the teaching snapshot with --allow-dev --snapshot-id before running")
    if args.review_source == "agent" and any(q.get("agent_reviewed") is not True
            or q.get("human_reviewed") is not False or q.get("fictional") is not False for q in questions):
        raise SystemExit("Agent evaluation requires explicitly agent-reviewed non-fictional questions; human_reviewed stays false")
    if args.review_source == "human" and not args.allow_dev and (
        len(questions) < 30
        or any(q.get("human_reviewed") is not True or q.get("fictional", False) is not False for q in questions)
    ):
        raise SystemExit("Formal evaluation requires >=30 independently human-reviewed, non-fictional questions")
    if len({q["snapshot_id"] for q in questions}) != 1:
        raise SystemExit("Use one frozen snapshot for a fair comparison")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    output = args.output / stamp
    output.mkdir(parents=True)
    manifest = {"question_set_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                "questions": questions, "mode_order": args.modes,
                "review_source": args.review_source, "dev_only": args.allow_dev or args.review_source == "agent",
                "created_at": stamp, "manual_semantic_review": "pending"}
    with httpx.Client(base_url=args.api, timeout=30, trust_env=False) as client:
        token = os.environ.get(args.token_env)
        if token:
            client.headers["Authorization"] = f"Bearer {token}"
        else:
            username = args.username or input("Username: ")
            login = client.post("/api/auth/login", json={"username": username, "password": getpass.getpass("Password: ")})
            login.raise_for_status()
        snapshot = client.get(f"/api/snapshots/{questions[0]['snapshot_id']}")
        snapshot.raise_for_status()
        manifest["snapshot"] = snapshot.json()
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        for mode in args.modes:
            directory = output / mode
            directory.mkdir()
        for number, q, mode in case_order(questions, args.modes):
            record = evaluate_case(client, q, mode, args.timeout)
            filename = f"{number:04d}.json"
            (output / mode / filename).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(mode, q["question_id"], record.get("run", {}).get("status", "technical_error"), flush=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
