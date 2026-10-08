"""Application launcher script for SAP CDS View Recommender & Generator."""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    app_path = BASE_DIR / "app" / "main.py"
    print("=" * 60)
    print("Launching SAP CDS View Recommender & Generator (Streamlit)")
    print(f"Target UI Script: {app_path}")
    print("=" * 60)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.headless=false"]
    subprocess.run(cmd)
