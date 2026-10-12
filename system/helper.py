"""
Shared state for a run. Agents run one at a time and do not communicate
directly. Each agent reads what the others did from the files below, so a run
can be audited afterwards.

Files in a run directory:

    logs/proposals.jsonl       one line per objective the Theorist proposed
    logs/derivations.jsonl     one line per derivation, including refusals
    logs/experiments.jsonl     one line per algorithm evaluated
    logs/dead-ends.jsonl       one line per rejected proposal, with the cause
    logs/sessions.jsonl        one line per agent cycle
    champion/champion.json     the current best, overwritten on promotion
    task/TASK.md               the task, copied in by launch.py
    task-profile.md            settings, copied from LAUNCH.md

The logs are append-only. A proposal is complete when a derivation or a dead-end
record with the same `proposal_id` exists. A derivation is complete when an
experiment or a dead-end record with that id exists. No record is deleted, so a
run can be resumed at any point and its full history is kept.

The champion files are the exception. They are overwritten on promotion, because
they hold the current best algorithm. The history of all evaluations is in
experiments.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def proposal_id(objective: str) -> str:
    """
    Stable identifier for a proposed objective: a content hash of the objective
    as stated, normalised for whitespace.

    Two proposals with the same text therefore get the same identifier. This is
    intended, because the second proposal is a duplicate of the first.
    """
    text = " ".join(objective.split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveProposal:
    """
    An objective function the Theorist proposes.

    Attributes:
        proposal_id:  Content hash of the objective.
        objective:    The objective in full, as a minimisation problem with its
                      constraints.
        changed:      Which part of the champion this replaces, and why it should
                      help on this data.
        theory:       The origin of each term, and the term that balances it.
                      For example, a penalty term without an opposing term can
                      drive a factor matrix to zero.
        coefficients: The trade-off coefficients this objective introduces, with
                      the values to evaluate at.
        falsified_by: The result that would show the reasoning was wrong, stated
                      before the run.
        novelty:      One of "novel", "published", "adjacent".
        queries:      The literature searches run to establish novelty.
        citations:    The closest work found.
        proposed_by:  Agent name.
    """

    proposal_id: str
    objective: str
    changed: str
    theory: str
    coefficients: dict[str, float]
    falsified_by: str
    novelty: str
    queries: list[str]
    citations: list[str]
    proposed_by: str
    proposed_at: str = field(default_factory=_now)


@dataclass
class DerivationRecord:
    """
    The Analyst's derivation of update rules from a proposed objective.

    A refusal is recorded in the same way. For some objectives the gradient
    cannot be separated into non-negative parts. Identifying these objectives is a
    result about the framework, not a failed cycle.

    Attributes:
        proposal_id:  Which proposal this derives.
        lagrangian:   The Lagrangian, including constraint terms.
        gradients:    Partial derivative per block, with K substituted for every
                      Phi'(X)Phi(X) pair.
        kkt:          The complementary slackness conditions.
        pseudocode:   The specification the Evaluator implements: blocks,
                      context, the two parts of each update, guards,
                      intermediates, and the objective. Every symbol used must
                      appear in one of them.
        supports:     The properties that the derivation establishes, and the
                      argument for each. A proof by an auxiliary function is a
                      valid argument for monotonicity. Similarity to a rule that
                      is known to converge is not a proof, and is recorded as
                      such.
        refused:      True if no valid separation exists.
        reason:       Why, when refused.
        closes:       The class of objectives that a refusal rules out, if the
                      refusal applies to more than this proposal. Recording the
                      class prevents a later Theorist from proposing another
                      member of it.
        agent:        Which Analyst produced it.
    """

    proposal_id: str
    lagrangian: str
    gradients: dict[str, str]
    kkt: str
    pseudocode: dict[str, Any]
    supports: dict[str, str]
    agent: str
    refused: bool = False
    reason: str = ""
    closes: str = ""
    derived_at: str = field(default_factory=_now)


@dataclass
class ExperimentRecord:
    """
    The result of running one derived algorithm on the declared protocol.

    Attributes:
        experiment_id: Unique id for this evaluation.
        proposal_id:   Which proposal.
        accuracy:      Per data set, the mean and standard deviation of the
                       clustering accuracy over restarts. Empty if screened out.
        paired:        Per data set, the comparison against the champion run
                       from the same initialisations. Empty if screened out
                       before the paired protocol.
        sigma:         Per data set, the kernel width selected on the hold-out.
        n_iter:        Per data set, the mean number of iterations to stop.
        checks:        Per data set, how many restarts satisfied each
                       theoretical criterion. Reported per data set, because a
                       rule can satisfy a criterion on one data set and fail it
                       on another.
        n_restarts:    Size of the ensemble.
        seed:          Base seed, so the result is reproducible.
        agent:         Which Evaluator produced it.
    """

    experiment_id: str
    proposal_id: str
    accuracy: dict[str, dict[str, float]]
    paired: dict[str, dict[str, float]]
    sigma: dict[str, float]
    n_iter: dict[str, float]
    checks: dict[str, dict[str, int]]
    n_restarts: int
    seed: int
    agent: str
    recorded_at: str = field(default_factory=_now)


@dataclass
class DeadEndRecord:
    """
    A rejected proposal and the reason for its rejection.

    Attributes:
        proposal_id: Which proposal.
        cause:       A theoretical criterion the run failed — "monotone",
                     "nonneg" or "finite" — or one of "published",
                     "not_derivable", "no_improvement", "redundant",
                     "underspecified". Only "published" and "not_derivable"
                     rule out the idea itself. The other causes describe
                     outcomes that a later proposal may improve on.
        detail:      The supporting evidence: a citation, a refusal reason, the
                     check that failed, or the numbers.
        closes:      Copied from the derivation when a refusal rules out a
                     class of objectives. The Theorist reads these records before
                     proposing.
        agent:       Which agent recorded it.
    """

    proposal_id: str
    cause: str
    detail: str
    agent: str
    closes: str = ""
    recorded_at: str = field(default_factory=_now)


@dataclass
class SessionRecord:
    """One agent cycle, for the run's audit trail."""

    agent: str
    action: str
    detail: str
    at: str = field(default_factory=_now)


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------


class FolderState:
    """
    Read and write a run directory.

    Logs are append-only. The champion is overwritten on promotion.

    Args:
        run_root: The run directory.
    """

    def __init__(self, run_root: Path | str) -> None:
        self.root = Path(run_root)
        self.logs = self.root / "logs"

    # -- setup --------------------------------------------------------------

    def initialise(self) -> None:
        """Create the log directory and its empty logs. Calling it again has no effect."""
        self.logs.mkdir(parents=True, exist_ok=True)
        for name in (
            "proposals.jsonl",
            "derivations.jsonl",
            "experiments.jsonl",
            "dead-ends.jsonl",
            "sessions.jsonl",
        ):
            (self.logs / name).touch(exist_ok=True)
        (self.root / "champion").mkdir(parents=True, exist_ok=True)

    # -- the task -----------------------------------------------------------

    def read_task(self) -> tuple[dict[str, Any], str]:
        """
        Read TASK.md.

        The frontmatter contains the data sets, the protocol, the trade-off
        parameters and the promotion rule. The body describes the task. To change
        the protocol, edit TASK.md, not the code.

        Returns:
            (metadata, body) — frontmatter as a dict, Markdown body as text.

        Raises:
            FileNotFoundError: if task/TASK.md is missing.
        """
        path = self.root / "task" / "TASK.md"
        if not path.exists():
            raise FileNotFoundError(
                f"No TASK.md at {path}. launch.py copies the chosen task into "
                "<run>/task/ — if it is missing, the run was not built by launch.py."
            )
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return {}, text
        _, frontmatter, body = text.split("---", 2)
        return _parse_frontmatter(frontmatter), body.strip()

    # -- proposals ----------------------------------------------------------

    def append_proposal(self, proposal: ObjectiveProposal) -> None:
        """Append one proposal to the proposal log."""
        self._append(self.logs / "proposals.jsonl", proposal)

    def read_proposals(self) -> list[ObjectiveProposal]:
        """Every proposal made so far, in order."""
        return [ObjectiveProposal(**record)
                for record in self._read(self.logs / "proposals.jsonl")]

    def pending_proposal(self) -> ObjectiveProposal | None:
        """
        The oldest proposal without a derivation or a dead-end record, or None.

        The Analyst uses this to decide whether it has work. A refusal counts as
        a derivation. A proposal that the Theorist recorded as published is not
        pending, because it is not derived.
        """
        derived = {record.proposal_id for record in self.read_derivations()}
        closed = {record.proposal_id for record in self.read_dead_ends()}
        for proposal in self.read_proposals():
            if proposal.proposal_id not in derived and proposal.proposal_id not in closed:
                return proposal
        return None

    # -- derivations --------------------------------------------------------

    def append_derivation(self, record: DerivationRecord) -> None:
        """Append one derivation, or one refusal, to the derivation log."""
        self._append(self.logs / "derivations.jsonl", record)

    def read_derivations(self) -> list[DerivationRecord]:
        """Every derivation attempted so far."""
        return [DerivationRecord(**record)
                for record in self._read(self.logs / "derivations.jsonl")]

    def pending_derivation(self) -> DerivationRecord | None:
        """
        The oldest derivation without an experiment or a dead-end record, or None.

        The Evaluator uses this to decide whether it has work. A refusal is
        returned like any other derivation. The Evaluator then records it in the
        dead-end log instead of running it.
        """
        evaluated = {record.proposal_id for record in self.read_experiments()}
        closed = {record.proposal_id for record in self.read_dead_ends()}
        for derivation in self.read_derivations():
            if derivation.proposal_id not in evaluated and derivation.proposal_id not in closed:
                return derivation
        return None

    # -- experiments --------------------------------------------------------

    def append_experiment(self, record: ExperimentRecord) -> None:
        """Append one evaluated algorithm to the experiment log."""
        self._append(self.logs / "experiments.jsonl", record)

    def read_experiments(self) -> list[ExperimentRecord]:
        """Every algorithm evaluated so far."""
        return [ExperimentRecord(**record)
                for record in self._read(self.logs / "experiments.jsonl")]

    def experiments_for(self, proposal_id: str) -> list[ExperimentRecord]:
        """All evaluations of one proposal — more than one if it was re-run."""
        return [record for record in self.read_experiments()
                if record.proposal_id == proposal_id]

    # -- dead ends ----------------------------------------------------------

    def append_dead_end(self, record: DeadEndRecord) -> None:
        """Append one rejection, with its cause."""
        self._append(self.logs / "dead-ends.jsonl", record)

    def read_dead_ends(self) -> list[DeadEndRecord]:
        """
        Every rejection so far.

        The Theorist reads this before proposing. Records with a `closes` field
        rule out a whole class of objectives.
        """
        return [DeadEndRecord(**record)
                for record in self._read(self.logs / "dead-ends.jsonl")]

    # -- champion -----------------------------------------------------------

    def read_champion(self) -> dict[str, Any]:
        """
        The current best algorithm: its objective, its derivation, and the module
        that holds its code.

        The Theorist reads this record to decide what to change, so it contains
        the derivation. It contains no accuracy values, because stored values
        would become outdated if the protocol changed. Accuracy is recorded in the
        experiment log.

        Raises:
            FileNotFoundError: if the champion record is missing.
        """
        path = self.root / "champion" / "champion.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No champion at {path}. launch.py copies the initial champion into "
                "<run>/champion/ — if it is missing, the run was not built by launch.py."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def write_champion(self, record: dict[str, Any]) -> None:
        """
        Replace the champion record. The record is overwritten, not appended to,
        because it describes the current best algorithm. The history is in the
        experiment log.
        """
        path = self.root / "champion" / "champion.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def promote_champion(self, record: dict[str, Any], algorithm_source: str) -> None:
        """
        Replace the champion with a proposal that passed the promotion rule: its
        code in champion/algorithm.py and its record in champion/champion.json.

        `promotes` in scorer/run_algorithm.py makes the decision. This method only
        writes it. Both files are checked and written in full before either is
        replaced.

        Raises:
            ValueError: if the record names another module or carries accuracy,
                or the code is empty.
            SyntaxError: if the code does not compile.
        """
        if record.get("module") != "champion/algorithm.py":
            raise ValueError("Champion module must be champion/algorithm.py")
        if "accuracy" in record:
            raise ValueError("Champion accuracy belongs in the experiment log")
        if not isinstance(algorithm_source, str) or not algorithm_source.strip():
            raise ValueError("Champion algorithm source must be nonempty")
        compile(algorithm_source, "champion/algorithm.py", "exec")
        record_text = json.dumps(record, indent=2)
        folder = self.root / "champion"
        folder.mkdir(parents=True, exist_ok=True)
        algorithm_path = folder / "algorithm.py"
        metadata_path = folder / "champion.json"
        # Write both files before replacing either, so an error leaves the current
        # champion unchanged.
        staged = []
        try:
            for target, content in ((algorithm_path, algorithm_source),
                                    (metadata_path, record_text)):
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=folder, delete=False
                ) as handle:
                    handle.write(content)
                    staged.append((Path(handle.name), target))
            for temporary, target in staged:
                os.replace(temporary, target)
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)

    # -- sessions -----------------------------------------------------------

    def append_session(self, record: SessionRecord) -> None:
        """Log one agent cycle."""
        self._append(self.logs / "sessions.jsonl", record)

    # -- internals ----------------------------------------------------------

    def _append(self, path: Path, record: Any) -> None:
        """Append one dataclass record as a JSON line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")

    def _read(self, path: Path) -> Iterator[dict[str, Any]]:
        """Read a JSON-line log, skipping blank lines. Missing files read as empty."""
        if not path.exists():
            return iter(())
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """
    Parse the subset of YAML used in TASK.md: scalars, inline lists, and nested
    blocks.

    Implemented here instead of using PyYAML, so the only dependencies are numpy
    and scipy. An inline mapping that contains a list is not supported, so TASK.md
    uses nested blocks instead.
    """
    meta: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, meta)]

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue

        parent[key] = _coerce(value)
    return meta


def _coerce(value: str) -> Any:
    """Convert one YAML scalar, inline list or inline mapping to a Python value."""
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        mapping: dict[str, Any] = {}
        for part in inner.split(","):
            key, _, item = part.partition(":")
            mapping[key.strip()] = _coerce(item.strip())
        return mapping
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_coerce(item.strip()) for item in inner.split(",")]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")
