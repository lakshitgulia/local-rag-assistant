"""
Personal-use background watcher: monitors LOCAL_SCAN_DIR continuously via
real filesystem events (watchdog) and calls the same incremental ingestion
logic in ingest.py for any new or changed supported file.

Separate process from app.py -- the query-serving path never imports this,
and this never touches the query path. Reuses ingest.py's PARSERS/IMAGE_EXTS/
run_ingestion directly rather than re-implementing any of that logic.

Run: python watcher.py
"""
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from core import save_watcher_status
from ingest import IMAGE_EXTS, PARSERS, run_ingestion

load_dotenv()

SCAN_DIR = Path(os.environ.get("LOCAL_SCAN_DIR", "~/Downloads")).expanduser()
SUPPORTED_EXTS = set(PARSERS.keys()) | IMAGE_EXTS
DEBOUNCE_SECONDS = 3  # let a file finish writing before ingesting it -- a
# raw "created" event can fire while a download is still in progress.


class DownloadsHandler(FileSystemEventHandler):
    def __init__(self):
        self._timers = {}

    def _schedule(self, path_str: str):
        path = Path(path_str)
        if path.suffix.lower() not in SUPPORTED_EXTS or not path.is_file():
            return
        existing = self._timers.get(path_str)
        if existing:
            existing.cancel()
        timer = threading.Timer(DEBOUNCE_SECONDS, self._ingest, args=(path,))
        self._timers[path_str] = timer
        timer.start()

    def _ingest(self, path: Path):
        print(f"[watcher] change detected: {path.name}")
        run_ingestion([path])
        save_watcher_status({
            "last_run_at": datetime.now(timezone.utc).isoformat(),
            "last_file": path.name,
        })

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)


if __name__ == "__main__":
    print(f"[watcher] watching {SCAN_DIR} (supported: {sorted(SUPPORTED_EXTS)})")
    observer = Observer()
    observer.schedule(DownloadsHandler(), str(SCAN_DIR), recursive=True)
    observer.start()
    try:
        while observer.is_alive():
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
