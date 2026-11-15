"""Stable Windows entry point, including Python 3.11 .pth files in non-ASCII paths."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["api", "worker", "doctor", "migrate", "migrate-vectors",
                                            "init-admin", "celery", "beat"])
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.command == "api":
        import uvicorn
        uvicorn.run("app.api.main:app", host="127.0.0.1", port=int(os.environ.get("KB_API_PORT", "8000")))
    elif args.command == "worker":
        from app.jobs.worker import main as worker
        worker()
    elif args.command in ("doctor", "migrate-vectors"):
        from app.cli import main as cli
        sys.argv.insert(1, args.command)
        return cli()
    elif args.command == "init-admin":
        from app.bootstrap import main as bootstrap
        return bootstrap()
    elif args.command in ("celery", "beat"):
        from app.jobs.tasks import celery_app
        if args.command == "celery":
            celery_app.worker_main(["worker", "--pool=solo", "--loglevel=INFO", *rest])
        else:
            schedule = Path(os.environ.get("KB_RUNTIME_DIR", str(ROOT / "data"))) / "celerybeat"
            celery_app.start(["beat", "--loglevel=INFO", f"--schedule={schedule}", *rest])
    else:
        from alembic import command
        from alembic.config import Config
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


if __name__ == "__main__":
    raise SystemExit(main())
