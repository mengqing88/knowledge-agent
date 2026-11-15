"""Launch the normal Windows scripts in a disposable stack, then run browser QA.

All model HTTP is directed to a loopback-only deterministic server. It is never
selected by product settings. No paid requests or real .env changes are made.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "backend/tests")]

from scripts.acceptance.isolated_acceptance import acceptance_environment  # noqa: E402


def mock_server():
    from app.core.schemas import Analysis, EvidenceGate, Plan, Reranking, Verification
    from conftest import FakeProvider

    state = {"failed_once": False, "requests": []}
    schemas = dict(Planner=Plan, Reranker=Reranking, EvidenceGate=EvidenceGate, Analyst=Analysis, Verifier=Verification)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/embeddings":
                result = {"data": [{"index": i, "embedding": [1., 0., 0.]} for i in range(len(body["input"]))],
                          "usage": {"prompt_tokens": 10, "total_tokens": 10}}
            elif self.path == "/v1/responses":
                role = re.search(r"角色：(\w+)", body["instructions"])[1]
                data = json.loads(body["input"])
                question = data.get("question", "")
                state["requests"].append(dict(role=role, question=question))
                if role == "Planner" and question.startswith("验收失败重跑") and not state["failed_once"]:
                    state["failed_once"] = True
                    self.send_error(503, "Deliberately injected local acceptance failure")
                    return
                time.sleep(10 if role == "Planner" and question.startswith(("验收取消", "验收断线")) else .5)
                output, _ = FakeProvider().structured(role, "", data, schemas[role])
                result = {"status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": output.model_dump_json()}]}],
                    "usage": {"input_tokens": 10, "output_tokens": 10}}
            else:
                self.send_error(404)
                return
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/closure-3yuan/browser")
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--playwright", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--member-only", action="store_true", help="Targeted rerun after member pattern fix")
    args = parser.parse_args()
    if not args.project.startswith("kbaccept-"):
        raise SystemExit("Disposable acceptance project required")
    runtime = args.output.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    env = acceptance_environment(args.project, runtime)
    server, state = mock_server()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    env.update(CHAT_BASE_URL=base, EMBEDDING_BASE_URL=base, CHAT_API_STYLE="responses",
               PROVIDER_MAX_RETRIES="0", MAX_REVISION_ROUNDS="0", MAX_RETRIEVAL_ROUNDS="1",
               KB_BROWSER_URL=f"http://127.0.0.1:{args.port}", KB_BROWSER_RUNTIME=str(runtime),
               KB_PLAYWRIGHT=str(args.playwright), MODEL="browser-local-fake")
    env["KB_BROWSER_MEMBER_ONLY"] = "1" if args.member_only else "0"
    powershell = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    start = [*powershell, str(ROOT / "scripts/start.ps1"), "-SkipDatabase", "-RuntimeDir", str(runtime),
             "-Port", str(args.port)]
    stop = [*powershell, str(ROOT / "scripts/stop.ps1"), "-RuntimeDir", str(runtime)]
    evidence = dict(project=args.project, url=env["KB_BROWSER_URL"], model="loopback deterministic test server",
                    checks=[], paid_model_calls=0)
    logs = (runtime / ("launcher-member-fix.log" if args.member_only else "launcher.log")).open("w", encoding="utf-8")
    try:
        # This record refers only to a nonexistent PID and this harness process;
        # a stale record must be archived without terminating this live process.
        (runtime / "processes.json").write_text(json.dumps({"api": 2147483600, "other": os.getpid()}))
        subprocess.run(start, cwd=ROOT, env=env, stdout=logs, stderr=subprocess.STDOUT, check=True)
        deadline = time.monotonic() + 60
        with httpx.Client(trust_env=False, timeout=3) as client:
            while time.monotonic() < deadline:
                try:
                    if client.get(env["KB_BROWSER_URL"] + "/api/health").is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError("Started API did not become healthy")
        evidence["checks"].append("normal-start-with-stale-and-reused-PID-record")
        duplicate = subprocess.run(start, cwd=ROOT, env=env, stdout=logs, stderr=subprocess.STDOUT)
        if duplicate.returncode == 0:
            raise AssertionError("Duplicate startup must be blocked")
        evidence["checks"].append("live-duplicate-start-blocked")
        result = subprocess.run([str(args.node), str(ROOT / "scripts/acceptance/browser_acceptance.mjs")], cwd=ROOT,
                                env=env, stdout=logs, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Browser interaction failed; see browser report and launcher log")
        evidence["checks"].append("browser-interactions")
        evidence["status"] = "passed"
    except Exception as exc:
        evidence.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        subprocess.run(stop, cwd=ROOT, env=env, stdout=logs, stderr=subprocess.STDOUT)
        time.sleep(1)
        with httpx.Client(trust_env=False, timeout=2) as client:
            try:
                client.get(env["KB_BROWSER_URL"] + "/api/health")
                evidence["stop"] = "failed-still-listening"
            except httpx.HTTPError:
                evidence["stop"] = "passed-port-closed"
        server.shutdown()
        logs.close()
        if evidence.get("stop") != "passed-port-closed":
            evidence["status"] = "failed"
        evidence["mock_requests"] = state["requests"]
        report_name = "startup-member-fix-report.json" if args.member_only else "startup-report.json"
        (runtime / report_name).write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    if evidence["status"] != "passed":
        raise RuntimeError("Acceptance cleanup did not close the API port")
    print(json.dumps(evidence, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
