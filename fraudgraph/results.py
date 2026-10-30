import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["git_sha"] = _git_sha()
    body["created_at"] = datetime.now(timezone.utc).isoformat()
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True))
    return path


def read_results() -> dict[str, dict]:
    if not RESULTS_DIR.exists():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))}
