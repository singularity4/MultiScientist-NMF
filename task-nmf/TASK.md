---
framework: nonlinear orthogonal NMF, kernel form

datasets:
  synthetic:
    n: 200
    dim: 2
    k: 4
    sigma_grid: [0.5, 8.0, 0.5]
  moons:
    n: 200
    dim: 2
    k: 2
    sigma_grid: [0.2, 2.0, 0.1]

protocol:
  metric: clustering accuracy, Hungarian matching against ground-truth labels
  n_restarts: 256
  seed: 0
  max_iter: 300
  stop: "E_prev - E <= kappa * max(1, E_prev)"
  kappa: 1.0e-3
  screen_dataset: synthetic
  screen_restarts: 8

parameters:
  alpha: 10
  mu: 100

promotion:
  rule: "every theoretical criterion holds AND accuracy beats the champion"
  comparison: paired seeds
  # Champion and proposal are run from identical initialisations, seed by seed,
  # and the per-seed differences are tested. Paired rather than a comparison of
  # means, because initialisation accounts for most of the variance here.
  test: "paired difference, positive and significant at p < 0.05"
---

# Task

Find better algorithms in the nonlinear orthogonal NMF framework.

## The framework

The factorisation is carried out after a nonlinear map Φ to a feature space. The
basis is restricted to the column space of Φ(X), so Φ(X) never has to be
computed, and every quantity can be written through the kernel matrix
K = Φᵀ(X)Φ(X). Every proposal must be expressible through K.

## The champion

`champion/` contains KNSC-Rcut and KNSC-Ncut with their objectives, derivations
and code. KNSC-Rcut is the active champion.

Read the derivations as well as the results. They show which modelling choices
were made, and each of these choices could have been made differently.

## What is being asked

Propose an objective function that satisfies the theoretical criteria and gives
higher clustering accuracy than the champion on the data sets declared above.

The task does not restrict which part of the objective may be changed. Only the
framework and the evaluation are fixed.

## Possible outcomes

The theoretical criteria are:

- **Non-negativity:** the factor matrices stay non-negative. The multiplicative
  form of the updates guarantees this.
- **Monotonicity:** the objective does not increase from one iteration to the
  next. A convergence proof guarantees this.
- **Finiteness:** no value becomes infinite or NaN.

A rule that fails any of them is not promoted.

**The criteria hold and accuracy is significantly higher.** The proposal is
promoted.

**A criterion fails and accuracy is higher.** A factor matrix became negative,
the objective increased, or a value became infinite or NaN. The proposal is not
promoted. Its results remain in the experiment log, and the dead-end log names
the failed criterion.

**The criteria hold and accuracy is not significantly higher.** The objective is
valid and converges, but does not improve accuracy on these data sets. It is
recorded in the dead-end log, so that it is not proposed again.

**Not derivable.** The gradient cannot be split into non-negative parts, so no
multiplicative update exists. This is a result about the framework, not a
failure of the run.
