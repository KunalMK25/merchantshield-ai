"""
In-process scheduler for the retraining job.

Meant for a single long-running API process. It runs `job` every `interval_seconds`
(first run one interval after start, not immediately), never overlaps two runs, survives
a failing run, and treats "another process holds the job lock" as a skip rather than an
error. If you run the API under several workers, or serverless, use cron instead:

    0 2 * * *  cd /path/to/merchantshield-ai && .venv/bin/python -m ml.retrain.job >> ml/data_store/cron.log 2>&1
"""

import logging
import threading
import time

from ml.retrain.job import JobAlreadyRunning

logger = logging.getLogger("merchantshield")


class RetrainScheduler:
    def __init__(self, job, interval_seconds: float):
        self._job = job
        self.interval = float(interval_seconds)
        self._stop = threading.Event()
        self._thread = None
        self._run_lock = threading.Lock()
        self._next_run_at = None
        self._running = False
        self._last_result = None
        self._last_error = None
        self._skipped = 0

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._next_run_at = time.time() + self.interval
        self._thread = threading.Thread(target=self._loop, name="retrain-scheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None
        self._next_run_at = None

    def _loop(self):
        while not self._stop.wait(max(self._next_run_at - time.time(), 0)):
            self._run_once()
            self._next_run_at = time.time() + self.interval

    def _run_once(self):
        if not self._run_lock.acquire(blocking=False):
            return
        try:
            self._running = True
            result = self._job()
            self._last_result = {"status": (result or {}).get("status", "ok"),
                                 "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            self._last_error = None
        except JobAlreadyRunning:
            self._skipped += 1
        except Exception as e:  # keep the schedule alive; the job already logged its own failure
            logger.error(f"Scheduled retraining failed: {e}")
            self._last_error = f"{type(e).__name__}: {e}"
        finally:
            self._running = False
            self._run_lock.release()

    def status(self) -> dict:
        active = self._thread is not None
        return {
            "enabled": active,
            "interval_hours": self.interval / 3600.0,
            "running": self._running,
            "next_run_at": (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._next_run_at))
                            if active and self._next_run_at else None),
            "last_result": self._last_result,
            "last_error": self._last_error,
            "skipped_runs": self._skipped,
        }
