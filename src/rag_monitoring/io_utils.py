import json
from pathlib import Path
import pandas as pd
import yaml

def create_run_directory(output_root, run_name):
    root = Path(output_root)
    candidate = root / run_name
    index = 1
    while candidate.exists():
        candidate = root / f"{run_name}_{index:02d}"
        index += 1
    candidate.mkdir(parents=True)
    return candidate

def save_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def save_summary_csv(path, calibration_rows, test_rows):
    frames = []
    for split, rows in [("calibration", calibration_rows), ("test", test_rows)]:
        frame = pd.DataFrame(rows)
        frame.insert(0, "split", split)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
