#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)

print("==> Installing requirements...")
run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "-q"])

print("==> Updating nodes...")
run(["bash", str(ROOT / "shell" / "update_nodes.sh")])

print("==> Starting server...")
run([sys.executable, str(ROOT / "scripts" / "server.py")])
