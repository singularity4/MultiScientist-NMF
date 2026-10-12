"""
The current champion in the form the scorer expects: an `update` and an
`objective` function for each algorithm, both with arguments (state, ctx).

This file is overwritten when a proposal is promoted. It is the only code in a
run that changes, so it is kept in champion/ next to the champion record, and
not in scorer/, which is never modified. `champion/champion.json` describes the
objective and its derivation. Accuracy is recorded in the experiment log, not in
the champion record. The Theorist reads both files to decide what to change.

The initial champion consists of KNSC-Rcut and KNSC-Ncut: kernel-based
non-negative spectral clustering with the ratio cut and with the normalized cut.
The two differ only by the scaling D^(-1/2), which turns the ratio cut into the
normalized cut.

The update rules and the objective follow the published paper. The objective is
expressed through the kernel matrix K = Phi'(X) Phi(X), so it can be evaluated
without computing the feature map Phi(X).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scorer.knsc import EPS

PARAMS = {"alpha": 10.0, "mu": 100.0}

# The orthogonality terms have the factor 2*mu in the update rules and mu in the
# objective. A rule written with mu instead of 2*mu uses half the stated
# orthogonality weight, and would minimise a different objective from the one the
# scorer evaluates.


def update_rcut(state: dict[str, np.ndarray], ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    One iteration of KNSC-Rcut.

    Each rule multiplies a factor matrix elementwise by the ratio of the negative
    to the positive part of its gradient, as obtained from the KKT conditions.
    """
    H, F = state["H"], state["F"]
    K, alpha, mu = ctx["K"], ctx["alpha"], ctx["mu"]

    negative_part = alpha * (F.T @ K) + 2 * mu * H
    positive_part = alpha * (F.T @ K @ F @ H) + 2 * mu * (H @ H.T) @ H
    H = H * negative_part / np.maximum(positive_part, EPS)

    negative_part = K @ H.T
    positive_part = K @ F @ H @ H.T
    F = F * negative_part / np.maximum(positive_part, EPS)
    return {"H": H, "F": F}


def objective_rcut(state: dict[str, np.ndarray], ctx: dict[str, Any]) -> float:
    """
    The objective, evaluated through the kernel:

        alpha ||Phi(X) - Phi(X) F H||_F^2 + mu ||H H' - I||_F^2

    The fidelity term expands to Tr(K) - 2 Tr(H'F'K) + Tr(H'F'KFH), so the
    feature map is never needed explicitly.
    """
    H, F = state["H"], state["F"]
    K, alpha, mu = ctx["K"], ctx["alpha"], ctx["mu"]
    fidelity = np.trace(K) - 2 * np.trace(H.T @ F.T @ K) + np.trace(H.T @ F.T @ K @ F @ H)
    orthogonality = np.linalg.norm(H @ H.T - np.eye(H.shape[0]), "fro") ** 2
    return float(alpha * fidelity + mu * orthogonality)


def update_ncut(state: dict[str, np.ndarray], ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    One iteration of KNSC-Ncut.

    Differs from Rcut only by the scaling D^(-1/2). D is fixed, so its inverse
    square root is taken from ctx rather than recomputed inside the loop.
    """
    Z, F = state["H"], state["F"]
    K, alpha, mu = ctx["K"], ctx["alpha"], ctx["mu"]
    Dm = ctx["Dm"]
    KD = K @ Dm

    negative_part = alpha * (F.T @ KD) + 2 * mu * Z
    positive_part = alpha * (F.T @ K @ F @ Z) + 2 * mu * (Z @ Z.T) @ Z
    Z = Z * negative_part / np.maximum(positive_part, EPS)

    negative_part = KD @ Z.T
    positive_part = K @ F @ Z @ Z.T
    F = F * negative_part / np.maximum(positive_part, EPS)
    return {"H": Z, "F": F}


def objective_ncut(state: dict[str, np.ndarray], ctx: dict[str, Any]) -> float:
    """The Ncut objective: the data is scaled by D^(-1/2)."""
    Z, F = state["H"], state["F"]
    K, alpha, mu, Dm = ctx["K"], ctx["alpha"], ctx["mu"], ctx["Dm"]
    KD = Dm @ K @ Dm
    fidelity = np.trace(KD) - 2 * np.trace(Z.T @ F.T @ K @ Dm) + np.trace(Z.T @ F.T @ K @ F @ Z)
    orthogonality = np.linalg.norm(Z @ Z.T - np.eye(Z.shape[0]), "fro") ** 2
    return float(alpha * fidelity + mu * orthogonality)


# The scorer reads UPDATE, OBJECTIVE and PARAMS, whatever the functions are
# called. A promoted champion overwrites this file and binds them to its own
# functions. The normalized-cut form above is not bound. It is the second
# published algorithm and is kept to document the effect of the D^(-1/2) scaling.
UPDATE = update_rcut
OBJECTIVE = objective_rcut
