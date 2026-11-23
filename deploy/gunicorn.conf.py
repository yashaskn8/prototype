"""Gunicorn settings for the backend-only Servy API on a Linux VM.

One worker process: BGE, V5 and the Chroma client are loaded once per process
(~1-2 GB RAM), and RAG admission is a per-process lock (one AI turn at a time,
others get HTTP 503 rag_busy). Threads serve the non-AI endpoints and keep SSE
streams open concurrently. Scale by adding VMs behind a load balancer, not workers.
"""
import os

bind = os.getenv("SERVY_BIND", "127.0.0.1:8000")      # nginx terminates TLS in front
workers = int(os.getenv("SERVY_GUNICORN_WORKERS", "1"))
worker_class = "gthread"
threads = int(os.getenv("SERVY_GUNICORN_THREADS", "8"))
timeout = int(os.getenv("SERVY_GUNICORN_TIMEOUT", "120"))  # SSE streams send keep-alives every 12 s
graceful_timeout = 30
keepalive = 5
preload_app = False            # models warm in the worker (SERVY_PRELOAD_MODELS=1)
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("SERVY_GUNICORN_LOGLEVEL", "info")
# Never log request bodies or Authorization headers.
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms'
