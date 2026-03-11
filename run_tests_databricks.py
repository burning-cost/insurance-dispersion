"""
Run insurance-dispersion tests on Databricks serverless compute.

Usage:
    python run_tests_databricks.py
"""

import base64
import os
import sys
import time
from pathlib import Path

env_path = Path.home() / ".config/burning-cost/databricks.env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute as _compute
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

PROJECT_ROOT = Path(__file__).parent
WORKSPACE_PATH = "/Workspace/insurance-dispersion"

# ---------------------------------------------------------------------------
# Upload project files
# ---------------------------------------------------------------------------

def upload_file(local_path: Path, remote_path: str) -> None:
    remote_dir = "/".join(remote_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=remote_dir)
    except Exception:
        pass
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=ImportFormat.AUTO,
        overwrite=True,
    )


SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache"}

print("Uploading project files to Databricks workspace...")
for fpath in sorted(PROJECT_ROOT.rglob("*")):
    if not fpath.is_file():
        continue
    if fpath.suffix not in (".py", ".toml", ".md", ".txt"):
        continue
    rel = fpath.relative_to(PROJECT_ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        continue
    # Skip the test runner itself to avoid confusion
    if fpath.name == "run_tests_databricks.py":
        continue
    remote = f"{WORKSPACE_PATH}/{rel}".replace("\\", "/")
    upload_file(fpath, remote)
    print(f"  Uploaded: {rel}")

print("Upload complete.")

# ---------------------------------------------------------------------------
# Create test runner notebook
# ---------------------------------------------------------------------------

NOTEBOOK_CONTENT = """\
# Databricks notebook source
# MAGIC %pip install formulaic scipy pandas numpy pytest --quiet

# COMMAND ----------

import subprocess, sys, os, shutil

# Copy project to /tmp to avoid Workspace FS issues with editable installs
shutil.copytree("/Workspace/insurance-dispersion", "/tmp/insurance-dispersion", dirs_exist_ok=True)

# Install the package
r_install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/tmp/insurance-dispersion", "--quiet", "--no-deps"],
    capture_output=True, text=True,
)

# Run tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/tmp/insurance-dispersion/tests/",
     "-v", "--tb=short",
     "--rootdir=/tmp/insurance-dispersion",
    ],
    capture_output=True, text=True,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
         "PYTHONPATH": "/tmp/insurance-dispersion/src"},
)
test_out = result.stdout + ("\\nSTDERR:\\n" + result.stderr if result.stderr else "")

summary = (
    f"=== INSTALL (rc={r_install.returncode}) ===\\n{r_install.stderr[-300:] if r_install.stderr else 'ok'}\\n\\n"
    f"=== TESTS (exit={result.returncode}) ===\\n"
    + (test_out[-7000:] if len(test_out) > 7000 else test_out)
)

dbutils.notebook.exit(summary)
"""

test_nb_path = f"{WORKSPACE_PATH}/run_tests"
encoded_nb = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=test_nb_path,
    content=encoded_nb,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Test notebook uploaded to {test_nb_path}")

# ---------------------------------------------------------------------------
# Submit job with serverless compute (client="2")
# ---------------------------------------------------------------------------

print("Submitting test job (serverless)...")
run_resp = w.jobs.submit(
    run_name="insurance-dispersion-tests",
    tasks=[
        jobs.SubmitTask(
            task_key="run_tests",
            notebook_task=jobs.NotebookTask(
                notebook_path=test_nb_path,
                base_parameters={},
            ),
            environment_key="serverless",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="serverless",
            spec=_compute.Environment(client="2"),
        )
    ],
)

run_id = run_resp.run_id
print(f"Job run ID: {run_id}")

# ---------------------------------------------------------------------------
# Poll and collect results
# ---------------------------------------------------------------------------

print("Waiting for job to complete...")
while True:
    state = w.jobs.get_run(run_id=run_id)
    lc = state.state.life_cycle_state
    print(f"  {time.strftime('%H:%M:%S')} {lc.value}")
    if lc in (
        jobs.RunLifeCycleState.TERMINATED,
        jobs.RunLifeCycleState.SKIPPED,
        jobs.RunLifeCycleState.INTERNAL_ERROR,
    ):
        break
    time.sleep(15)

result_state = state.state.result_state
print(f"\nResult: {result_state.value if result_state else 'UNKNOWN'}")

for task in (state.tasks or []):
    try:
        out = w.jobs.get_run_output(run_id=task.run_id)
        if out.notebook_output and out.notebook_output.result:
            print("\n" + out.notebook_output.result)
        if out.error:
            print(f"Error: {out.error}")
        if out.error_trace:
            print(f"Trace:\n{out.error_trace[-2000:]}")
    except Exception as e:
        print(f"Could not get output: {e}")

ok = result_state == jobs.RunResultState.SUCCESS
print("\nSUCCESS" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
