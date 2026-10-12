#!/usr/bin/env python3
"""
Creates a run directory for one task. A run directory is self-contained: it holds its own copy of the system files,
the scorer, the data sets, the initial champion, the task, one instruction file per agent, and the shared-state folders
the agents read and write. The template directory is not modified, so several runs can be started from it without
interfering.

Usage:
    python3 launch.py                    # name from task + timestamp
    python3 launch.py my-run
    python3 launch.py my-run --task task-nmf
    python3 launch.py my-run --output-dir /tmp/runs

Each task directory must contain TASK.md, and a LAUNCH.md must exist either in
that directory or in one of its parents. LAUNCH.md holds the settings the
orchestrator reads.

Then:
    cd <run-dir>
    # open runbook.md in the orchestrator session
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TEMPLATE_DIR = Path(__file__).resolve().parent

# The four agent roles and their role files. Each agent's instructions consist of
# HEARTBEAT.md followed by its role file. The Theorist proposes objectives, the
# Analyst derives update rules, the Evaluator runs them, and the Observer audits.
ROLE_FILES = {
    "theorist": "ROLE-THEORIST.md",
    "analyst": "ROLE-ANALYST.md",
    "evaluator": "ROLE-EVALUATOR.md",
    "observer": "ROLE-OBSERVER.md",
}


def find_task_profile(task_dir: Path) -> Path:
    """
    Find the LAUNCH.md that governs a task.

    Searches the task directory, then its parents, so several tasks may share
    one LAUNCH.md while any of them can override it.

    Raises:
        FileNotFoundError: if no LAUNCH.md is found.
    """
    current = task_dir.resolve()
    template_root = TEMPLATE_DIR.resolve()
    while True:
        candidate = current / "LAUNCH.md"
        if candidate.exists():
            return candidate
        if current == template_root or template_root not in current.parents:
            raise FileNotFoundError(
                f"No LAUNCH.md found walking up from {task_dir} to {template_root}. "
                "Every task must ship a LAUNCH.md (there is no generic fallback)."
            )
        current = current.parent


def build_run_directory(run_root: Path, task_dir: Path, profile_path: Path) -> None:
    """
    Assemble a fresh run directory.

    Args:
        run_root:     Directory to create. Must not already exist.
        task_dir:     Task directory containing TASK.md.
        profile_path: The LAUNCH.md found by find_task_profile.
    """
    run_root.mkdir(parents=True, exist_ok=False)

    shutil.copytree(TEMPLATE_DIR / "system", run_root / "system")
    shutil.copytree(TEMPLATE_DIR / "scorer", run_root / "scorer")

    # The champion is copied, not referenced. It is the only code that a run
    # changes: promotion overwrites champion/algorithm.py and champion.json in the
    # run directory, and the template must stay unchanged.
    shutil.copytree(TEMPLATE_DIR / "champion", run_root / "champion")

    # Copied to a fixed name, so agents need not know the task directory name.
    shutil.copytree(task_dir, run_root / "task")

    # Data sets. scorer/datasets.py resolves names against <run>/data/.
    data_dir = TEMPLATE_DIR / "data"
    if data_dir.exists():
        shutil.copytree(data_dir, run_root / "data")

    shutil.copy(TEMPLATE_DIR / "runbook.md", run_root / "runbook.md")
    shutil.copy(profile_path, run_root / "task-profile.md")

    # One instruction file per agent at the top of the run directory: the
    # shared heartbeat with that role appended.
    heartbeat = (TEMPLATE_DIR / "system" / "templates" / "HEARTBEAT.md").read_text(
        encoding="utf-8"
    )
    for role, role_filename in ROLE_FILES.items():
        role_body = (
            TEMPLATE_DIR / "system" / "templates" / role_filename
        ).read_text(encoding="utf-8")

        instructions = heartbeat.replace("<injected by launch.py>", role)
        instructions += "\n\n---\n\n# Your Role\n\n" + role_body
        (run_root / f"HEARTBEAT-{role}.md").write_text(instructions, encoding="utf-8")

    # Create the folders the agents read and write.
    sys.path.insert(0, str(run_root))
    from system.helper import FolderState

    FolderState(run_root).initialise()


def main() -> None:
    """Parse arguments, resolve the task and profile, build the run directory."""
    parser = argparse.ArgumentParser(
        description="Create a fresh MultiScientist-NMF run directory."
    )
    parser.add_argument("name", nargs="?", default=None, help="run name")
    parser.add_argument(
        "--task",
        default="task-nmf",
        help="task directory inside the template",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where to create the run (default: template's parent)",
    )
    arguments = parser.parse_args()

    task_dir = TEMPLATE_DIR / arguments.task
    if not (task_dir / "TASK.md").exists():
        raise SystemExit(f"No TASK.md in {task_dir}")

    profile_path = find_task_profile(task_dir)

    name = arguments.name or (
        f"{arguments.task}-{datetime.now(timezone.utc).strftime('%m%d_%H%M')}"
    )
    out_dir = Path(arguments.output_dir) if arguments.output_dir else TEMPLATE_DIR.parent
    run_root = out_dir / name

    build_run_directory(run_root, task_dir, profile_path)

    print(f"Run created: {run_root}")
    print(f"  task    : {arguments.task}")
    print(f"  profile : {profile_path.relative_to(TEMPLATE_DIR)}")
    print(f"  agents  : {', '.join(ROLE_FILES)}")
    print("            HEARTBEAT-<role>.md, one per agent")
    print()
    print(f"  cd {run_root}")
    print("  # then open runbook.md in the orchestrator session")


if __name__ == "__main__":
    main()
