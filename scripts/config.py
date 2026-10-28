"""Private settings (endpoint, accounts, local paths) live in local.json, which is gitignored.
See local.example.json for the keys."""
import json
from pathlib import Path

LOCAL = Path(__file__).resolve().parent.parent / "local.json"
CFG = json.loads(LOCAL.read_text()) if LOCAL.exists() else {}


def get(key):
    if key not in CFG:
        raise SystemExit(f"missing '{key}' in {LOCAL} (copy local.example.json and fill it in)")
    return CFG[key]
