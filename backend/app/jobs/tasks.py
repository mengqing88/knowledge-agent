"""Celery schedules durable DB jobs. Leases protect against duplicate delivery and process loss."""
from celery import Celery

from app.core.config import get_settings

settings = get_settings()
celery_app = Celery("team_knowledge", broker=settings.celery_broker_url)
celery_app.conf.update(
    task_ignore_result=True, task_serializer="json", accept_content=["json"],
    broker_connection_retry_on_startup=True, worker_prefetch_multiplier=1, worker_concurrency=1,
    task_acks_late=True, task_reject_on_worker_lost=True,
    beat_schedule={"recover-and-dispatch": {"task": "knowledge.execute_next", "schedule": 3.0,
                                          "options": {"expires": 6}}},
    timezone="Asia/Shanghai",
)


@celery_app.task(name="knowledge.execute_next")
def execute_next():
    from contextlib import nullcontext

    from langgraph.checkpoint.postgres import PostgresSaver

    from app.core.db import Session
    from app.integrations.providers import ModelProvider
    from app.jobs.services import claim_job
    from app.jobs.worker import process_job

    job = claim_job(Session, settings)
    if not job:
        return
    cm = (PostgresSaver.from_conn_string(settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1))
          if settings.database_url.startswith("postgresql") else nullcontext(None))
    with cm as saver:
        if saver:
            saver.setup()
        process_job(Session, settings, ModelProvider(settings), job, saver)
