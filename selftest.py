#!/usr/bin/env python3
"""
Tests the code before a run.

Three checks test the environment. Five checks test that the protocol is
implemented as specified and that the scorer rejects invalid rules.

The most important check is `champion is invariant to reparameterisation`. It
detects a misplaced constant factor in the update rules.

    python3 selftest.py          # -v for tracebacks
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Every check uses this data set, so no check depends on data that is not
# included in the repository.
TEST_DATASET = "synthetic"

failures: list[str] = []


def check(name, test):
    try:
        print(f"  ok    {name} — {test()}")
    except Exception as error:
        failures.append(name)
        print(f"  FAIL  {name}: {type(error).__name__}: {error}")
        if "-v" in sys.argv:
            traceback.print_exc()


def task_meta():
    from system.helper import _parse_frontmatter

    text = (HERE / "task-nmf" / "TASK.md").read_text(encoding="utf-8")
    return _parse_frontmatter(text.split("---", 2)[1])


def fixture(sigma=5.0):
    """
    The test data, the champion, and the context the scorer would build.

    It uses `_context` from the scorer instead of building its own context, so the
    checks run on exactly the context that the scorer produces.
    """
    from scorer import datasets
    from scorer.knsc import gausskernel
    from scorer.run_algorithm import _blocks_for, _context, load_champion

    champion = load_champion()
    X, labels_true, n_clusters = datasets.load(TEST_DATASET)
    ctx = _context(gausskernel(X, X, sigma), champion.PARAMS)
    return champion, X, labels_true, _blocks_for(n_clusters, X.shape[1], None), ctx


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def scorer_imports():
    from scorer import datasets, run_algorithm  # noqa: F401
    from scorer.run_algorithm import load_champion

    champion = load_champion()
    for name in ("UPDATE", "OBJECTIVE", "PARAMS"):
        assert hasattr(champion, name), f"the champion exposes no {name}"
    return "scorer and champion import"


def task_parses():
    meta = task_meta()
    for block in ("datasets", "protocol", "parameters", "promotion"):
        assert block in meta, f"TASK.md has no {block} block"

    grid = meta["datasets"][TEST_DATASET]["sigma_grid"]
    assert isinstance(grid, list) and len(grid) == 3, (
        f"sigma_grid parsed as {grid!r}, not a [start, stop, step] list. An "
        "inline mapping containing a list does not parse; use nested blocks."
    )
    return f"{len(meta['datasets'])} data sets, {meta['protocol']['n_restarts']} restarts"


def datasets_load():
    """
    Every data set declared in TASK.md must load, have the declared number of
    clusters, and give the same data when loaded twice. Otherwise a data error
    would appear only in the Evaluator's first full run.
    """
    from scorer import datasets

    meta = task_meta()
    for name, spec in meta["datasets"].items():
        X, labels_true, n_clusters = datasets.load(name)
        assert X.shape[1] == len(labels_true), f"{name}: samples must be in columns"
        assert n_clusters == spec["k"], (
            f"{name}: TASK.md declares k={spec['k']}, the data has {n_clusters}"
        )
        repeated, _, _ = datasets.load(name)
        assert np.allclose(X, repeated), f"{name}: loading twice gives different data"
    return ", ".join(meta["datasets"])


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


def protocol_follows_the_specification():
    """
    Three points where the implementation could differ from the specification
    without raising an error.

    The kernel is exp(-d^2 / sigma^2), not exp(-d^2 / (2 sigma^2)), because the
    width grids in TASK.md are defined for this form. The two hold-out halves have
    the same cluster proportions, so the width selected on one half also suits the
    other. The affinity matrix is the fully connected Gaussian graph. A zero
    affinity matrix would make any proposed graph term ineffective.
    """
    from scorer import datasets
    from scorer.knsc import gausskernel
    from scorer.run_algorithm import HOLDOUT_SEED, _holdout_split

    two_points = np.array([[0.0, 1.0], [0.0, 0.0]])
    kernel_entry = gausskernel(two_points, two_points, 1.0)[0, 1]
    assert abs(kernel_entry - np.exp(-1.0)) < 1e-12, (
        f"K = {kernel_entry}, expected exp(-1) for unit distance and unit width"
    )

    X, labels_true, _ = datasets.load(TEST_DATASET)
    selection, validation = _holdout_split(labels_true, HOLDOUT_SEED)
    assert not set(selection) & set(validation), "the two halves overlap"
    assert len(selection) + len(validation) == X.shape[1], "the split loses samples"
    for label in np.unique(labels_true):
        in_selection = int((labels_true[selection] == label).sum())
        in_validation = int((labels_true[validation] == label).sum())
        assert abs(in_selection - in_validation) <= 1, (
            f"cluster {label} is unbalanced: {in_selection} against {in_validation}"
        )
    assert np.array_equal(
        selection, _holdout_split(labels_true, HOLDOUT_SEED)[0]
    ), "the split is not reproducible"

    _, _, _, _, ctx = fixture()
    assert ctx["A"].sum() > 0, "the affinity graph is empty"
    assert np.allclose(np.diag(ctx["D"]), ctx["A"].sum(1)), (
        "D is not the degree matrix of A"
    )
    return "kernel, stratified hold-out, affinity graph"


def champion_is_invariant_to_reparameterisation():
    """
    The factor two in the kernel width and the factor two on the orthogonality
    terms can both be absorbed into the parameters. This gives an algebraically
    identical algorithm,

        champion(sigma, mu)  ==  absorbed(sig = sigma / sqrt(2), mu = 2 mu),

    so both forms must give the same factors from the same initialisation. A
    misplaced factor in the update rules breaks this identity.
    """
    from scorer import datasets
    from scorer.knsc import accuracy, labels
    from scorer.run_algorithm import load_champion

    champion = load_champion()
    X, labels_true, n_clusters = datasets.load(TEST_DATASET)
    n_points = X.shape[1]

    sig, alpha, mu_absorbed = 5.0, 10.0, 100.0
    squared_norms = np.sum(X ** 2, axis=0)
    kernel = np.exp(
        -(squared_norms[:, None] - 2 * (X.T @ X) + squared_norms[None, :])
        / (2 * sig ** 2)
    )

    def absorbed_rcut(H, F, n_iter=300):
        """The same rules with both factors of two absorbed."""
        K = kernel
        for _ in range(n_iter):
            H = H * (alpha * (F.T @ K) + mu_absorbed * H) / (
                alpha * (F.T @ K @ F @ H) + mu_absorbed * (H @ H.T) @ H
            )
            F = F * (K @ H.T) / (K @ F @ H @ H.T)
        return H

    ctx = {"K": kernel, "alpha": alpha, "mu": mu_absorbed / 2}
    largest_difference = 0.0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        H_start = rng.random((n_clusters, n_points))
        F_start = rng.random((n_points, n_clusters))

        H_absorbed = absorbed_rcut(H_start.copy(), F_start.copy())

        state = {"H": H_start.copy(), "F": F_start.copy()}
        for _ in range(300):
            state = champion.update_rcut(state, ctx)

        largest_difference = max(
            largest_difference, float(np.max(np.abs(state["H"] - H_absorbed)))
        )
        assert accuracy(labels_true, labels(state["H"])) == accuracy(
            labels_true, labels(H_absorbed)
        ), f"the two forms disagree on the clustering at seed {seed}"

    assert largest_difference < 1e-10, (
        f"the two forms differ by {largest_difference:.3e}, more than accumulated "
        "rounding"
    )
    return f"agrees to {largest_difference:.1e} over 4 seeds"


def champion_objective_decreases():
    """
    The champion module computes the objective through the kernel matrix. If this
    expression is wrong, the objective does not decrease monotonically under
    update rules that are proved to converge, and every later proposal would be
    compared against a faulty champion.
    """
    from scorer.run_algorithm import run_once

    champion, _, labels_true, blocks, ctx = fixture()
    for seed in range(8):
        run = run_once(
            champion.UPDATE, champion.OBJECTIVE, ctx, blocks, labels_true, seed=seed
        )
        assert run.checks["monotone"], f"the objective increased on seed {seed}"
    return "monotone on 8/8 seeds"


def declarations_reach_the_algorithm():
    """
    The trade-off parameters come from TASK.md and the coefficients from the
    proposal. Both must reach the update rule unchanged, and a proposal must not
    be able to redefine a parameter declared in TASK.md.
    """
    from scorer.knsc import EPS
    from scorer.run_algorithm import _merge_coefficients, evaluate_algorithm

    seen: dict[str, float] = {}

    def update(state, ctx):
        seen.update({name: ctx.get(name) for name in ("alpha", "mu", "nu")})
        H, F = state["H"], state["F"]
        H = H * (F.T @ ctx["K"]) / np.maximum(F.T @ ctx["K"] @ F @ H, EPS)
        return {"H": H, "F": F}

    def objective(state, ctx):
        return float(np.trace(ctx["K"]))

    meta = task_meta()
    meta = {**meta, "protocol": {**meta["protocol"], "screen_restarts": 2}}
    evaluate_algorithm(update, objective, meta, seed=0, coefficients={"nu": 0.25})

    for name in ("alpha", "mu"):
        assert seen[name] == meta["parameters"][name], seen
    assert seen["nu"] == 0.25, seen

    try:
        _merge_coefficients({"alpha": 10.0}, {"alpha": 5.0})
    except ValueError:
        pass
    else:
        raise AssertionError("a proposal redefined a declared parameter")
    return f"alpha={seen['alpha']}, mu={seen['mu']}, nu={seen['nu']}"


def a_failed_criterion_refuses_promotion():
    """
    Both parts of the promotion rule, tested on a rule that improves accuracy but
    fails monotonicity.

    An L1 penalty on F has a valid gradient separation and keeps F non-negative.
    The penalty term does not scale with F, so the convergence proof of the
    champion does not apply. The monotonicity check must detect the failure, and
    the promotion rule must reject the rule regardless of its accuracy.
    """
    from scorer.knsc import EPS
    from scorer.run_algorithm import DatasetResult, promotes, run_once

    _, _, labels_true, blocks, base_ctx = fixture()
    ctx = {**base_ctx, "lambda_F": 1.0}

    def update(state, ctx):
        H, F = state["H"], state["F"]
        K, alpha, mu, lambda_F = ctx["K"], ctx["alpha"], ctx["mu"], ctx["lambda_F"]
        H = H * (alpha * (F.T @ K) + 2 * mu * H) / np.maximum(
            alpha * (F.T @ K @ F @ H) + 2 * mu * (H @ H.T) @ H, EPS
        )
        F = F * (K @ H.T) / np.maximum(
            K @ F @ H @ H.T + lambda_F * np.ones_like(F), EPS
        )
        return {"H": H, "F": F}

    def objective(state, ctx):
        H, F = state["H"], state["F"]
        K, alpha, mu, lambda_F = ctx["K"], ctx["alpha"], ctx["mu"], ctx["lambda_F"]
        fidelity = (
            np.trace(K) - 2 * np.trace(H.T @ F.T @ K) + np.trace(H.T @ F.T @ K @ F @ H)
        )
        orthogonality = np.linalg.norm(H @ H.T - np.eye(H.shape[0]), "fro") ** 2
        return float(alpha * fidelity + mu * orthogonality + lambda_F * np.abs(F).sum())

    violations = sum(
        not run_once(update, objective, ctx, blocks, labels_true, seed=seed)
        .checks["monotone"]
        for seed in range(8)
    )
    assert violations > 0, "a rule with no convergence argument satisfied monotonicity"

    # A result with this failure and a large, significant improvement in accuracy.
    # It is constructed rather than computed, because the test concerns the
    # promotion rule, not the numbers.
    result = DatasetResult(
        dataset=TEST_DATASET, sigma=5.0, accuracy_mean=64.4, accuracy_sd=5.0,
        n_restarts=32, n_iter_mean=24.0,
        checks={"finite": 32, "nonneg": 32, "monotone": 3},
        paired={"champion_mean": 58.5, "delta_mean": 5.9, "delta_sd": 6.1,
                "p_value": 3e-8},
    )
    promoted, reason = promotes(
        {"screened": "passed", "_results": {TEST_DATASET: result}, "n_restarts": 32}
    )
    assert not promoted, "a proposal failing monotonicity promoted"
    assert "monotone" in reason, f"the reason does not name the criterion: {reason}"
    return f"caught on {violations}/8 seeds; refused with '{reason}'"


print("MultiScientist-NMF selftest\n")
check("scorer imports", scorer_imports)
check("TASK.md parses", task_parses)
check("data sets load", datasets_load)
check("protocol follows the specification", protocol_follows_the_specification)
check("champion is invariant to reparameterisation",
      champion_is_invariant_to_reparameterisation)
check("champion objective decreases", champion_objective_decreases)
check("declarations reach the algorithm", declarations_reach_the_algorithm)
check("a failed criterion refuses promotion", a_failed_criterion_refuses_promotion)

print()
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("All checks passed. `python3 launch.py <name>` to create a run.")
