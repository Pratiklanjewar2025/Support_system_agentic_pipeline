"""Run the repository's existing deterministic suites in isolated processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SUITES = ["test_evaluation_metrics.py", "test_intervention_policy.py", "test_adaptive_execution.py",
          "test_evidence_grounded_diagnosis.py", "test_incident_response_cache.py",
          "test_public_benchmark.py", "test_ablation_studies.py", "test_reproducibility.py"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=SUITES)
    args = parser.parse_args()
    import uuid
    directory = ROOT/"results/ada_rag_ops_evaluation"/("LOCAL_TESTS_"+uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    # Some legacy test scripts overwrite their conventional output paths. Run a
    # copied source tree so existing experimental artifacts cannot be changed.
    import shutil
    isolated = directory/"workspace"
    shutil.copytree(ROOT, isolated, ignore=shutil.ignore_patterns(
        ".git", ".venv", ".vscode", "results", "__pycache__", ".env", ".env.example",
        "research_paper_results", "research_paper_results.zip", "report_screenshots", "*.db-shm", "*.db-wal"))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}
    results = []
    for suite in args.suites:
        try:
            # Existing suites configure their own deterministic mocks. Never label these as live inference.
            completed = subprocess.run([sys.executable, "-B", suite], cwd=isolated, env=env, capture_output=True, timeout=300)
            (directory/(suite+".log")).write_bytes(completed.stdout+completed.stderr)
            result = {"suite": suite, "returncode": completed.returncode, "status": "PASS" if completed.returncode == 0 else "FAIL"}
        except subprocess.TimeoutExpired as error:
            (directory/(suite+".log")).write_bytes((error.stdout or b"")+(error.stderr or b""))
            result = {"suite": suite, "status": "FAILED", "reason": "300 second test timeout"}
        results.append(result)
        print(json.dumps(result), flush=True)
    (directory/"test_report.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "mode": "local_tests_not_experiment", "suites": results}, indent=2), encoding="utf-8")
    print("Test logs:", directory, flush=True)
    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
