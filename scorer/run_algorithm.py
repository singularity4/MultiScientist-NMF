"""
Evaluates one proposed algorithm under the protocol declared in TASK.md. It
reports the clustering accuracy on the hold-out half and whether the theoretical
criteria hold.

An algorithm consists of two functions, which the Evaluator writes from the
Analyst's derivation:

    update(state, ctx) -> state        one iteration
    objective(state, ctx) -> float     the scalar being minimised

`state` holds the factor matrices named in the derivation, so an objective with
an additional factor matrix needs no change in this module. `ctx` holds the
kernel matrix K, the graph affinity matrix A, the degree matrix D and the
trade-off parameters. Neither function loads data, computes accuracy or writes
files.

This module contains everything the comparison depends on: the initialisation,
the iteration loop, the restarts, the kernel-width selection, the accuracy
metric and the checks. Neither the Analyst nor the Evaluator modifies it, so all
proposals and the champion are evaluated in the same way.

The notation follows the framework: n data points, k clusters, m dimensions,
sigma the width of the Gaussian kernel, alpha and mu the trade-off parameters,
and any further coefficients that a proposal introduces. No other variable in
this module uses these names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import reload
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats

from scorer import datasets
from scorer.knsc import accuracy, gausskernel, labels

Update = Callable[[dict[str, np.ndarray], dict[str, Any]], dict[str, np.ndarray]]
Objective = Callable[[dict[str, np.ndarray], dict[str, Any]], float]

# The theoretical criteria, which a correct derivation guarantees. Non-negativity
# is preserved by the multiplicative form, monotonicity follows from the
# convergence proof, and finiteness means that no value becomes infinite or NaN.
# No other property is checked.
CHECKS = ("finite", "nonneg", "monotone")

# Tolerances for the checks. The monotonicity tolerance is relative, because the
# magnitude of the objective differs by orders of magnitude between data sets. The
# non-negativity tolerance is absolute, because a correct multiplicative update
# produces no negative entries.
MONOTONE_TOL = 1e-8
NONNEG_TOL = 1e-9

# Seed of the hold-out split. Every algorithm selects its kernel width on the same
# half and is evaluated on the same other half, so both sides of a comparison use
# the same data, and a repeated run gives the same result.
HOLDOUT_SEED = 0


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """
    One run of one algorithm, on one data set, from one initialisation.

    Attributes:
        accuracy:  Clustering accuracy as a percentage.
        n_iter:    Iterations taken, which is max_iter unless kappa stopped it.
        checks:    Verdict per check; see `_check`.
        objective: Final value of the declared objective.
    """

    accuracy: float
    n_iter: int
    checks: dict[str, Any]
    objective: float


@dataclass
class DatasetResult:
    """
    An algorithm evaluated on one data set over many initialisations.

    Attributes:
        dataset:           Data set name.
        sigma:             Kernel width selected on the hold-out.
        accuracy_mean:     Mean clustering accuracy over restarts.
        accuracy_sd:       Standard deviation over restarts.
        n_restarts:        How many initialisations were used.
        n_iter_mean:       Mean iterations to stop.
        checks:            Per check, the number of restarts that passed.
        paired:            Comparison against the champion on the same
                           initialisations, or None where the run was not
                           paired. See `evaluate_algorithm`.
        accuracy_per_seed: Accuracy at each restart, in seed order. The paired
                           test compares these values with the champion's, seed
                           by seed, so they are stored in addition to the summary.
    """

    dataset: str
    sigma: float
    accuracy_mean: float
    accuracy_sd: float
    n_restarts: int
    n_iter_mean: float
    checks: dict[str, int]
    paired: dict[str, float] | None = None
    accuracy_per_seed: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The iteration loop
# ---------------------------------------------------------------------------


def _initialise(
    blocks: dict[str, tuple[int, int]], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Random non-negative matrices, uniform on [0, 1], as the framework specifies."""
    return {name: rng.random(shape) for name, shape in blocks.items()}


def _converged(objective_trace: list[float], kappa: float) -> bool:
    """
    Stopping rule: E_prev - E <= kappa * max(1, E_prev), as specified.
    """
    if len(objective_trace) < 2:
        return False
    previous, current = objective_trace[-2], objective_trace[-1]
    return (previous - current) <= kappa * max(1.0, abs(previous))


def _check(state: dict[str, np.ndarray], objective_trace: np.ndarray) -> dict[str, Any]:
    """
    Whether a completed run satisfied the theoretical criteria.

    Non-negativity follows from the multiplicative form when both parts of the
    gradient separation are non-negative. Monotonic decrease follows from a
    convergence proof. Finiteness means that no value became infinite or NaN. A
    rule that fails any of these criteria is not promoted, regardless of its
    accuracy.

    Args:
        state:           Factor matrices at the last iteration.
        objective_trace: The objective at every iteration, including its initial
                         value.
    """
    verdict: dict[str, Any] = {}

    verdict["finite"] = bool(
        all(np.all(np.isfinite(factor)) for factor in state.values())
    )
    verdict["nonneg"] = bool(
        all(factor.min() >= -NONNEG_TOL for factor in state.values())
    )
    verdict["monotone"] = bool(
        np.all(
            np.diff(objective_trace)
            <= MONOTONE_TOL * np.maximum(1.0, np.abs(objective_trace[:-1]))
        )
    )
    verdict["objective"] = float(objective_trace[-1])
    return verdict


def run_once(
    update: Update,
    objective: Objective,
    ctx: dict[str, Any],
    blocks: dict[str, tuple[int, int]],
    labels_true: np.ndarray,
    seed: int,
    max_iter: int = 300,
    kappa: float | None = 1e-3,
) -> RunResult:
    """
    Run one algorithm from one initialisation.

    The objective is evaluated at every iteration, so monotonicity can be checked
    without knowing the form of the objective. The same seed gives the same
    initialisation for any algorithm with the same factor matrices. This makes the
    paired comparison possible.

    Args:
        blocks: {name: shape} for every factor matrix the derivation names.
        kappa:  Stopping tolerance, or None to run to max_iter.
    """
    rng = np.random.default_rng(seed)
    state = _initialise(blocks, rng)

    objective_trace = [float(objective(state, ctx))]
    for _ in range(max_iter):
        state = update(state, ctx)
        objective_trace.append(float(objective(state, ctx)))
        if kappa is not None and _converged(objective_trace, kappa):
            break

    trace = np.asarray(objective_trace)
    if "H" not in state:
        raise KeyError(
            "The update returned no block named H. Cluster memberships are read "
            f"from H; the blocks returned were {sorted(state)}."
        )
    return RunResult(
        accuracy=float(accuracy(labels_true, labels(state["H"]))),
        n_iter=len(objective_trace) - 1,
        checks=_check(state, trace),
        objective=float(trace[-1]),
    )


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


def _sigma_values(sigma_grid: list[float]) -> np.ndarray:
    """
    Expand a [start, stop, step] grid as TASK.md declares it.

    Values are rounded to the precision of the step. Without rounding,
    floating-point accumulation gives values such as 2.000000000000001, which
    would then be reported as the selected width.
    """
    start, stop, step = sigma_grid
    values = np.arange(start, stop + step / 2, step)
    decimals = max(0, -int(np.floor(np.log10(step))) + 2)
    return np.round(values, decimals)


def _blocks_for(
    n_clusters: int, n_points: int, extra: dict[str, tuple[int, int]] | None
) -> dict[str, tuple[int, int]]:
    """
    The factor matrices: H is (k x n) and F is (n x k). A proposal with
    additional factor matrices declares them in its derivation, and they are
    added here.
    """
    blocks = {"H": (n_clusters, n_points), "F": (n_points, n_clusters)}
    if extra:
        blocks.update(extra)
    return blocks


def _context(kernel: np.ndarray, params: dict[str, float]) -> dict[str, Any]:
    """
    The fixed quantities that every algorithm receives.

    The affinity matrix A is the fully connected graph with Gaussian weights, so
    it equals the kernel matrix. D is its degree matrix, and Dm = D^(-1/2) is the
    scaling that turns the ratio cut into the normalized cut. A proposal with a
    graph term uses A and D.
    """
    degrees = kernel.sum(axis=1)
    return {
        "K": kernel,
        "A": kernel,
        "D": np.diag(degrees),
        "Dm": np.diag(1.0 / np.sqrt(degrees)),
        **params,
    }


def _holdout_split(labels_true: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the sample indices into two equally sized halves with equally
    distributed cluster membership.

    The split is stratified by cluster. With a plain random permutation, a small
    cluster could fall almost entirely into one half, and the kernel width
    selected on one half would not suit the other.

    Returns:
        (selection, validation) index arrays.
    """
    rng = np.random.default_rng(seed)
    selection: list[int] = []
    validation: list[int] = []
    for label in np.unique(labels_true):
        indices = np.flatnonzero(labels_true == label)
        indices = indices[rng.permutation(len(indices))]
        half = len(indices) // 2
        selection.extend(indices[:half])
        validation.extend(indices[half:])
    return np.array(sorted(selection)), np.array(sorted(validation))


def _select_sigma(
    update: Update,
    objective: Objective,
    params: dict[str, float],
    X: np.ndarray,
    labels_true: np.ndarray,
    selection: np.ndarray,
    n_clusters: int,
    sigma_grid: list[float],
    n_restarts: int,
    max_iter: int,
    kappa: float | None,
    seed: int,
    extra_blocks: dict[str, tuple[int, int]] | None = None,
) -> float:
    """
    Choose the kernel width on the selection half, by best mean accuracy.

    The kernel is built on the selection half only, so the validation half does
    not influence the choice. The grid is searched with fewer restarts than the
    full protocol, to reduce the cost: the grid multiplies the work by the number
    of widths, and the ranking of the widths stabilises after a few restarts. The
    selected width is then evaluated with the full number of restarts on the
    other half.
    """
    X_selection = X[:, selection]
    labels_selection = labels_true[selection]
    blocks = _blocks_for(n_clusters, X_selection.shape[1], extra_blocks)
    n_grid_restarts = max(4, n_restarts // 8)

    best_sigma = None
    best_accuracy = -np.inf
    for sigma in _sigma_values(sigma_grid):
        ctx = _context(gausskernel(X_selection, X_selection, float(sigma)), params)
        mean_accuracy = np.mean(
            [
                run_once(
                    update, objective, ctx, blocks, labels_selection,
                    seed=seed + restart, max_iter=max_iter, kappa=kappa,
                ).accuracy
                for restart in range(n_grid_restarts)
            ]
        )
        if mean_accuracy > best_accuracy:
            best_sigma = float(sigma)
            best_accuracy = float(mean_accuracy)
    return best_sigma


def _tally(runs: list[RunResult]) -> dict[str, int]:
    """Count how many restarts passed each boolean check."""
    tally: dict[str, int] = {}
    for run in runs:
        for check, verdict in run.checks.items():
            if isinstance(verdict, bool):
                tally[check] = tally.get(check, 0) + int(verdict)
    return tally


def _screen(
    update: Update,
    objective: Objective,
    params: dict[str, float],
    dataset: str,
    sigma_grid: list[float],
    n_restarts: int,
    seed: int,
    extra_blocks: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """
    A short first test on one small data set, at three widths across its grid.

    Finiteness and non-negativity follow from the algebra of the update and must
    hold at all three widths. Monotonicity can fail at one width and hold at
    another, and the full protocol uses the width selected on the hold-out half.
    Monotonicity is therefore required at one of the three widths only. A screen
    at a single width could reject a rule that converges at the width actually
    used.

    The screen uses the selection half only. The validation half is first used in
    the full protocol.

    Returns:
        passed, the criteria that failed, the tally at the width that did best
        on monotonicity, and the widths tried.
    """
    X, labels_true, n_clusters = datasets.load(dataset)
    selection, _ = _holdout_split(labels_true, HOLDOUT_SEED)
    X_selection = X[:, selection]
    labels_selection = labels_true[selection]
    blocks = _blocks_for(n_clusters, X_selection.shape[1], extra_blocks)

    grid = _sigma_values(sigma_grid)
    widths = sorted({float(grid[0]), float(grid[len(grid) // 2]), float(grid[-1])})

    always = ("finite", "nonneg")
    best_tally: dict[str, int] = {}
    best_sigma = widths[0]
    best_accuracy = 0.0
    monotone_anywhere = False
    failed_somewhere: set[str] = set()

    for sigma in widths:
        ctx = _context(gausskernel(X_selection, X_selection, sigma), params)
        runs = [
            run_once(update, objective, ctx, blocks, labels_selection, seed=seed + restart)
            for restart in range(n_restarts)
        ]
        tally = _tally(runs)
        failed_somewhere.update(check for check in always if tally.get(check, 0) < n_restarts)
        if tally.get("monotone", 0) == n_restarts:
            monotone_anywhere = True
        if tally.get("monotone", 0) >= best_tally.get("monotone", -1):
            best_tally = tally
            best_sigma = sigma
            best_accuracy = float(np.mean([run.accuracy for run in runs]))

    # The failed criteria are collected over all widths. The returned tally belongs
    # to the width with the best monotonicity result, and finiteness or
    # non-negativity can fail at a different width.
    failed = [check for check in always if check in failed_somewhere]
    if not monotone_anywhere:
        failed.append("monotone")
    return {
        "passed": not failed,
        "failed": failed,
        "checks": best_tally,
        "sigma": best_sigma,
        "sigma_tried": widths,
        "accuracy_mean": best_accuracy,
    }


def _evaluate_dataset(
    update: Update,
    objective: Objective,
    params: dict[str, float],
    dataset: str,
    sigma_grid: list[float],
    n_restarts: int,
    max_iter: int,
    kappa: float | None,
    seed: int,
    extra_blocks: dict[str, tuple[int, int]] | None,
) -> DatasetResult:
    """
    Evaluate one algorithm on one data set by hold-out: the kernel width is
    selected on one half, and the accuracy is reported on the other.
    """
    X, labels_true, n_clusters = datasets.load(dataset)
    selection, validation = _holdout_split(labels_true, HOLDOUT_SEED)

    sigma = _select_sigma(
        update, objective, params, X, labels_true, selection, n_clusters,
        sigma_grid, n_restarts, max_iter, kappa, seed, extra_blocks,
    )

    X_validation = X[:, validation]
    labels_validation = labels_true[validation]
    blocks = _blocks_for(n_clusters, X_validation.shape[1], extra_blocks)
    ctx = _context(gausskernel(X_validation, X_validation, sigma), params)

    runs = [
        run_once(
            update, objective, ctx, blocks, labels_validation,
            seed=seed + restart, max_iter=max_iter, kappa=kappa,
        )
        for restart in range(n_restarts)
    ]
    accuracies = np.array([run.accuracy for run in runs])

    return DatasetResult(
        dataset=dataset,
        sigma=sigma,
        accuracy_mean=float(accuracies.mean()),
        accuracy_sd=float(accuracies.std()),
        n_restarts=n_restarts,
        n_iter_mean=float(np.mean([run.n_iter for run in runs])),
        checks=_tally(runs),
        accuracy_per_seed=accuracies.tolist(),
    )


def _pair_against_champion(
    proposal_result: DatasetResult,
    dataset: str,
    sigma_grid: list[float],
    protocol: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """
    Run the champion on the same data and the same initialisations, and test the
    per-seed differences.

    The champion selects its own kernel width on the same selection half. Running
    it at the proposal's width would compare a tuned algorithm with an untuned
    one. Both algorithms start from the same initialisations, so the variance due
    to initialisation cancels in the per-seed differences. This is why the
    comparison is paired, and why it needs a second full run.
    """
    champion = load_champion()
    X, labels_true, n_clusters = datasets.load(dataset)
    selection, validation = _holdout_split(labels_true, HOLDOUT_SEED)

    champion_sigma = _select_sigma(
        champion.UPDATE, champion.OBJECTIVE, champion.PARAMS,
        X, labels_true, selection, n_clusters, sigma_grid,
        protocol["n_restarts"], protocol["max_iter"], protocol["kappa"], seed,
    )

    X_validation = X[:, validation]
    labels_validation = labels_true[validation]
    blocks = _blocks_for(n_clusters, X_validation.shape[1], None)
    ctx = _context(
        gausskernel(X_validation, X_validation, champion_sigma), champion.PARAMS
    )

    champion_accuracies = np.array(
        [
            run_once(
                champion.UPDATE, champion.OBJECTIVE, ctx, blocks, labels_validation,
                seed=seed + restart, max_iter=protocol["max_iter"],
                kappa=protocol["kappa"],
            ).accuracy
            for restart in range(protocol["n_restarts"])
        ]
    )

    differences = np.array(proposal_result.accuracy_per_seed) - champion_accuracies
    _, p_value = stats.ttest_rel(proposal_result.accuracy_per_seed, champion_accuracies)
    return {
        "champion_sigma": champion_sigma,
        "champion_mean": float(champion_accuracies.mean()),
        "delta_mean": float(differences.mean()),
        "delta_sd": float(differences.std()),
        "p_value": float(p_value),
    }


def _merge_coefficients(
    params: dict[str, float], coefficients: dict[str, float] | None
) -> dict[str, float]:
    """
    Add the proposal's own coefficients to the trade-off parameters.

    A new term in the objective comes with its own coefficient. TASK.md declares
    only the protocol, so the Theorist sets the value, and the Evaluator passes it
    on unchanged.

    Raises:
        ValueError: if a coefficient has the same name as a parameter declared in
            TASK.md. These parameters are shared with the champion, and changing
            one would invalidate the comparison.
    """
    if not coefficients:
        return dict(params)
    clash = set(coefficients) & set(params)
    if clash:
        raise ValueError(
            f"The proposal redefines task parameters {sorted(clash)}. Those are "
            "fixed by TASK.md and are shared with the champion; a proposal may "
            "introduce coefficients but not change the protocol."
        )
    return {**params, **coefficients}


def evaluate_algorithm(
    update: Update,
    objective: Objective,
    task_meta: dict[str, Any],
    seed: int = 0,
    extra_blocks: dict[str, tuple[int, int]] | None = None,
    coefficients: dict[str, float] | None = None,
    paired: bool = True,
) -> dict[str, Any]:
    """
    Evaluate one algorithm on all data sets under the protocol declared in TASK.md.

    A short screen on the small data set named in TASK.md runs first, and the
    evaluation stops if a criterion fails. The screen saves time, because the
    paired comparison doubles the cost of a full run.

    Args:
        task_meta:    Frontmatter of TASK.md — data sets, protocol, parameters.
        seed:         Base seed. Restarts run at seed, seed + 1, and so on.
        extra_blocks: Factor matrices beyond H and F, where the derivation names
                      them.
        coefficients: The coefficients that the proposal introduces, with the
                      values set by the Theorist. They may not have the same name
                      as a parameter declared in TASK.md. The champion is run
                      with the parameters in its own PARAMS.
        paired:       Run the champion on the same initialisations and test the
                      per-seed differences.

    Returns:
        The fields an ExperimentRecord takes, and `_results`, which carries the
        per-data-set results for `promotes` and is not logged.
    """
    protocol = task_meta["protocol"]
    params = _merge_coefficients(task_meta["parameters"], coefficients)
    screen_dataset = protocol["screen_dataset"]

    screen = _screen(
        update, objective, params,
        dataset=screen_dataset,
        sigma_grid=task_meta["datasets"][screen_dataset]["sigma_grid"],
        n_restarts=protocol["screen_restarts"],
        seed=seed,
        extra_blocks=extra_blocks,
    )
    if not screen["passed"]:
        return {
            "accuracy": {},
            "paired": {},
            "sigma": {screen_dataset: screen["sigma"]},
            "sigma_tried": screen["sigma_tried"],
            "n_iter": {},
            "checks": {screen_dataset: screen["checks"]},
            "n_restarts": protocol["screen_restarts"],
            "screened": "failed",
            "failed": screen["failed"],
        }

    results: dict[str, DatasetResult] = {}
    for dataset, spec in task_meta["datasets"].items():
        result = _evaluate_dataset(
            update, objective, params, dataset,
            sigma_grid=spec["sigma_grid"],
            n_restarts=protocol["n_restarts"],
            max_iter=protocol["max_iter"],
            kappa=protocol["kappa"],
            seed=seed,
            extra_blocks=extra_blocks,
        )
        if paired:
            result.paired = _pair_against_champion(
                result, dataset, spec["sigma_grid"], protocol, seed
            )
        results[dataset] = result

    return {
        "accuracy": {
            dataset: {"mean": dataset_result.accuracy_mean, "sd": dataset_result.accuracy_sd}
            for dataset, dataset_result in results.items()
        },
        "paired": {
            dataset: dataset_result.paired
            for dataset, dataset_result in results.items() if dataset_result.paired
        },
        "sigma": {
            dataset: dataset_result.sigma for dataset, dataset_result in results.items()
        },
        "n_iter": {
            dataset: dataset_result.n_iter_mean for dataset, dataset_result in results.items()
        },
        "checks": {
            dataset: dataset_result.checks for dataset, dataset_result in results.items()
        },
        "n_restarts": protocol["n_restarts"],
        "screened": "passed",
        "sigma_tried": screen["sigma_tried"],
        "_results": results,
    }


# ---------------------------------------------------------------------------
# The champion
# ---------------------------------------------------------------------------


def _champion_path() -> Path:
    """
    Locate the champion module.

    `launch.py` copies the champion into every run directory, so it is always
    found next to the scorer folder.
    """
    return Path(__file__).resolve().parents[1] / "champion"


def load_champion():
    """
    Import the current champion's algorithm module.

    The module is reloaded on every call, because promotion overwrites it and a
    single import would keep the old champion. The module defines UPDATE,
    OBJECTIVE and PARAMS, whatever the names of its functions.

    Raises:
        ImportError: if the champion is missing from the run directory.
    """
    import sys

    path = _champion_path()
    if not (path / "algorithm.py").exists():
        raise ImportError(
            f"The champion's algorithm.py was not found at {path}. launch.py "
            "copies the champion into each run; if it is missing, the run was "
            "not built by launch.py."
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import algorithm  # type: ignore

    return reload(algorithm)


# ---------------------------------------------------------------------------
# The promotion rule
# ---------------------------------------------------------------------------


def promotes(result: dict[str, Any], significance: float = 0.05) -> tuple[bool, str]:
    """
    Apply the promotion rule of TASK.md: every theoretical criterion satisfied on
    every restart of every data set, and a paired difference against the champion
    that is positive and significant.

    Args:
        significance: Threshold on the p-value of the paired test. It is not
                      called alpha, because alpha is the weight of the
                      reconstruction term.

    Returns:
        (promotes, reason). The reason names the first condition that failed.
    """
    if result.get("screened") == "failed":
        dataset = next(iter(result["checks"]))
        return False, f"screen failed on {dataset}: {', '.join(result['failed'])}"

    results = result["_results"]
    for dataset, dataset_result in results.items():
        for check in CHECKS:
            passed = dataset_result.checks.get(check, 0)
            if passed < dataset_result.n_restarts:
                return False, (
                    f"{check} failed on {dataset} "
                    f"({passed}/{dataset_result.n_restarts})"
                )

    compared = [
        dataset_result for dataset_result in results.values()
        if dataset_result.paired is not None
    ]
    if not compared:
        return False, "no paired comparison against the champion"

    for dataset_result in compared:
        paired = dataset_result.paired
        if paired["delta_mean"] <= 0:
            return False, (
                f"no improvement on {dataset_result.dataset} "
                f"({paired['delta_mean']:+.2f})"
            )
        if paired["p_value"] >= significance:
            return False, (
                f"improvement on {dataset_result.dataset} not significant "
                f"(p={paired['p_value']:.2g})"
            )

    return True, "every criterion held; paired improvement on every data set"
