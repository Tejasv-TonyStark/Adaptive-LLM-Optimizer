"""Offline human/judge agreement report and fail-closed learning gate.

Input: JSON list of {id, judge_model, answer_model, human_quality, judge_quality,
human_reviewed: true}. Labels must be independent of the judge. No paid calls.
"""
import argparse
import json
import math
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from Evaluation.Evaluator import JUDGE_MODEL, JUDGE_RUBRIC_VERSION

MIN_CASES = 20
MAX_MAE = 0.15
MAX_BIAS = 0.10

def calibration_report(rows):
    groups = {}
    seen = set()
    for row in rows:
        key = (row["judge_model"], row["answer_model"])
        identity = (*key, row["id"])
        if identity in seen:
            raise ValueError("Duplicate calibration case")
        seen.add(identity)
        if row.get("human_reviewed") is not True:
            raise ValueError("Independent human labels are required")
        scores = [row["human_quality"], row["judge_quality"]]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
            raise ValueError("Quality must be finite in [0, 1]")
        groups.setdefault(key, []).append(scores[1]-scores[0])
    results = []
    for (judge, answer), errors in sorted(groups.items()):
        mae = sum(abs(v) for v in errors)/len(errors)
        bias = sum(errors)/len(errors)
        results.append(dict(judge_model=judge, answer_model=answer, count=len(errors),
                            mae=mae, bias=bias, eligible=judge != answer and len(errors)>=MIN_CASES
                            and mae<=MAX_MAE and abs(bias)<=MAX_BIAS))
    return dict(rubric_version=JUDGE_RUBRIC_VERSION,
                created_at=datetime.now(timezone.utc).isoformat(), groups=results)

def judge_learning_allowed(answer_model):
    path = os.getenv("JUDGE_CALIBRATION_PATH")
    if not path or answer_model == JUDGE_MODEL:
        return False
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        created = datetime.fromisoformat(report["created_at"])
        now = datetime.now(timezone.utc)
        if report["rubric_version"] != JUDGE_RUBRIC_VERSION or not now-timedelta(days=30) <= created <= now:
            return False
        return any(g["judge_model"] == JUDGE_MODEL and g["answer_model"] == answer_model
                   and g["eligible"] is True and g["count"] >= MIN_CASES
                   and 0 <= g["mae"] <= MAX_MAE and abs(g["bias"]) <= MAX_BIAS
                   for g in report["groups"])
    except (OSError, ValueError, KeyError, TypeError):
        return False

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/calibration.json"))
    args = parser.parse_args()
    report = calibration_report(json.loads(args.labels.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
