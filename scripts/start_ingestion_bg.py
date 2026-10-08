"""Detached Background Job Launcher for Full SAP CDS View Ingestion.

Launches scripts/ingest_all_views.py as an independent detached background process,
ensuring long-running extractions (50,000+ views) persist uninterrupted.
"""

import sys
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = BASE_DIR / "scripts" / "ingest_all_views.py"
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "full_ingestion.log"

os.makedirs(LOGS_DIR, exist_ok=True)


def launch_background_ingestion(cap: int = 200, workers: int = 5) -> int:
    """Launch ingest_all_views.py detached from current shell/session."""
    print("=" * 70)
    print("LAUNCHING PRODUCTION ADAPTIVE CRAWLER IN DETACHED BACKGROUND")
    print(f"Target Script: {SCRIPT_PATH}")
    print(f"Cap: {cap}, Workers: {workers}")
    print(f"Persistent Log: {LOG_FILE}")
    print("=" * 70)

    if sys.platform == "win32":
        # Windows detached process using cmd start /b
        cmd_str = f'start /b "" "{sys.executable}" "{SCRIPT_PATH}" --cap {cap} --workers {workers}'
        proc = subprocess.Popen(
            f'cmd.exe /c {cmd_str}',
            cwd=str(BASE_DIR),
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = proc.pid
    else:
        # Linux/macOS nohup background process
        cmd = [sys.executable, str(SCRIPT_PATH), "--cap", str(cap), "--workers", str(workers)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        pid = proc.pid

    print("SUCCESS: Background Ingestion Process launched successfully!")
    print(f"Spawn Process PID: {pid}")
    print("To monitor live progress, run:")
    print(f"  Get-Content logs/full_ingestion.log -Wait -Tail 30")
    print("=" * 70)
    return pid


if __name__ == "__main__":
    launch_background_ingestion()
