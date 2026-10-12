# Settings for the task

`runbook.md` refers to the settings below by name. Each section describes what
the orchestrator does at that step.

Agents run one at a time and exchange information only through files in the run
directory.

## Setting: launch_command

```
python3 launch.py <run-name> --task task-nmf
```

This command creates a run directory. The directory contains copies of the
system files, the scorer, the data sets, the initial champion and the task, one
instruction file per agent, and a copy of this file named `task-profile.md`.

## Setting: seeding_policy

Start one Theorist. In the first cycle the proposal log is empty, and the
Theorist proposes one objective function based on the champion. The task file
specifies the data sets, the metric and the protocol, but not what to propose.
The Theorist chooses the direction of the search.

## Setting: cycle_order

The agents run in the order Theorist, Analyst, Evaluator, Observer. Each agent
first checks whether it has work and exits if it has none. Not every agent
therefore acts in every cycle:

- In the first cycle, the Theorist proposes an objective, the Analyst derives
  the update rules, the Evaluator runs them, and the Observer audits the result.
- In later cycles, the Theorist proposes again once every earlier proposal has
  been derived and evaluated. It uses the experiment log and the dead-end log.
- If the Theorist finds that its proposal is already published, the proposal is
  recorded in the dead-end log and is not derived. The Analyst then has no work
  and exits.
- If the Analyst finds that no valid separation of the gradient exists, it
  records a refusal. The Evaluator then records the refusal in the dead-end log
  and does not run anything.

The Observer only reads the logs.

## Setting: evaluate_dispatch

Start one Evaluator. It reads the pending derivation and implements the
pseudocode. It then runs a short screen on one small data set. If the screen
passes, it runs the full protocol and adds one record to
`logs/experiments.jsonl`.

The screen reduces the cost of evaluation. The paired comparison doubles the
cost of a full run, because the champion is run again from the same
initialisations as the proposal. A proposal that fails a theoretical criterion
on the screen is stopped before the full run.

## Setting: champion_promotion

After the Evaluator finishes, the orchestrator checks that an experiment record
exists for the current proposal. Records of earlier proposals do not count:

```python
assert state.experiments_for(derivation.proposal_id), (
    "evaluator recorded no experiment for the current proposal"
)
```

If the Analyst refused the proposal, the Evaluator records the refusal in
`logs/dead-ends.jsonl` and runs no experiment. In that case, check for a
dead-end record with `derivation.proposal_id` instead. These checks only confirm
that the Evaluator finished. They do not decide whether the proposal is
promoted.

The Evaluator decides promotion by calling `promotes(result)`. A proposal
replaces the champion if every theoretical criterion holds on every restart of
every data set, and its accuracy is significantly higher than the champion's in
a paired test. In that case the Evaluator replaces `champion/algorithm.py` and
`champion/champion.json`. Otherwise it adds a record to the dead-end log that
names the first condition that failed.

A proposal that improves accuracy but fails a criterion is not promoted. Its
results remain in the experiment log, and the dead-end log names the criterion
it failed.

## Setting: stagnation_response

The run has no fixed length. Because the search space is open, the Theorist can
always propose another objective.

Stop the run when the Theorist declines to propose because the dead-end log
already excludes every direction it can identify, or when the operator stops
the run. Neither case is an error.

If the Theorist proposes nothing and the proposal log is empty, the run has
stalled. This is an error. Report it and stop without retrying. Likely causes
are a malformed `TASK.md`, a missing data set or a missing champion record.

## Setting: exit_condition

The run is finished when the Theorist has declined to propose and every earlier
proposal has been derived and evaluated.

```python
import json
sessions = [json.loads(line) for line in (RUN_ROOT / "logs" / "sessions.jsonl").open()]
declined = any(session["action"] == "declined" for session in sessions)
done = declined and state.pending_proposal() is None and state.pending_derivation() is None
```

When the Theorist has no proposal, it adds a session record with the action
`"declined"`. Without this record, the orchestrator cannot distinguish a
finished run from one that is still waiting.

## Setting: final_report

The Observer reports what was proposed, derived, refused and evaluated, the
result of every check, and whether each agent stayed within its role.

The run has no separate summary of findings. The result of the run is the final
champion together with the dead-end log, which records the directions that were
ruled out.
