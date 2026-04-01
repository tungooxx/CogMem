import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml


def save_results(results: dict, logs_dir: str, name: str) -> str:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = f"{logs_dir}/{name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def save_config_snapshot(config, logs_dir: str) -> str:
    from dataclasses import asdict

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    path = f"{logs_dir}/config_snapshot.yaml"
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
    return path


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
