"""Enterprise Hard Reset Script for SAP CDS GraphRAG.

Automates sterile environment wipe:
1. Finds and terminates any background Python processes running ingest_all_views.py.
2. Safely deletes ChromaDB storage directory (./data/chroma_db) and local catalog cache (./data/cds_catalog.json).
3. Verifies complete removal and confirms sterile state.
"""

import sys
import os
import shutil
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CATALOG_FILE = DATA_DIR / "cds_catalog.json"


def terminate_crawler_processes():
    """Find and kill any background processes running ingest_all_views.py."""
    print("=" * 70)
    print("STEP 1: TERMINATING ACTIVE CRAWLER PROCESSES")
    print("=" * 70)

    if sys.platform == "win32":
        cmd = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -like '*ingest_all_views.py*' } | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output \"Terminated PID $($_.ProcessId)\" }"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", cmd],
            capture_output=True,
            text=True,
        )
        out = res.stdout.strip()
        if out:
            print(f"  [OK] Process Termination Result:\n{out}")
        else:
            print("  [INFO] No active ingest_all_views.py processes detected.")
    else:
        res = subprocess.run(["pkill", "-9", "-f", "ingest_all_views.py"], capture_output=True, text=True)
        print("  [OK] Executed pkill for ingest_all_views.py.")

    time.sleep(1.5)


def annihilate_databases():
    """Safely delete ChromaDB storage directory and local JSON seed catalog."""
    print("\n" + "=" * 70)
    print("STEP 2: DATABASE & CACHE ANNIHILATION")
    print("=" * 70)

    # 1. Delete catalog cache
    if CATALOG_FILE.exists():
        try:
            CATALOG_FILE.unlink()
            print(f"  [OK] Deleted offline catalog cache: {CATALOG_FILE}")
        except Exception as e:
            print(f"  [WARN] Could not delete {CATALOG_FILE}: {e}")
    else:
        print(f"  [INFO] Offline catalog cache {CATALOG_FILE} does not exist. (Clean)")

    # 2. Delete ChromaDB directory
    if CHROMA_DIR.exists():
        deleted = False
        for attempt in range(1, 4):
            try:
                shutil.rmtree(CHROMA_DIR)
                print(f"  [OK] Deleted ChromaDB directory: {CHROMA_DIR} (Attempt {attempt})")
                deleted = True
                break
            except Exception as e:
                time.sleep(0.5)

        if not deleted:
            # Force removal via Windows rmdir if locked by previous handle
            try:
                subprocess.run(["cmd.exe", "/c", f'rmdir /s /q "{CHROMA_DIR}"'], capture_output=True)
                if not CHROMA_DIR.exists():
                    print(f"  [OK] Force-deleted ChromaDB directory via rmdir.")
                    deleted = True
                else:
                    print(f"  [WARN] ChromaDB directory could not be deleted completely.")
            except Exception as e:
                print(f"  [ERROR] Failed to wipe {CHROMA_DIR}: {e}")
    else:
        print(f"  [INFO] ChromaDB storage directory {CHROMA_DIR} does not exist. (Clean)")


def verify_sterile():
    """Verify all legacy/test data has been destroyed."""
    print("\n" + "=" * 70)
    print("STEP 3: STERILITY VERIFICATION")
    print("=" * 70)

    chroma_gone = not CHROMA_DIR.exists()
    catalog_gone = not CATALOG_FILE.exists()

    if chroma_gone and catalog_gone:
        print("  [SUCCESS] All legacy vector databases and catalog caches are ANNIHILATED.")
        print("  [SUCCESS] Environment is 100% STERILE and ready for Clean Core Ingestion.")
        print("=" * 70)
        return True
    else:
        print(f"  [WARNING] Environment check: ChromaDB exists: {not chroma_gone}, Catalog exists: {not catalog_gone}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    terminate_crawler_processes()
    annihilate_databases()
    verify_sterile()
