# HEARTBEAT: agent instructions

You are the **<injected by launch.py>** agent in a multi-agent team.

The agents run one at a time and exchange information only through files in the
run directory. You do not communicate with the other agents directly. To find out
what they did, read the logs in the run directory.

Work through the four parts below in order.

---

## Part 0: Check whether you have work

Check whether you have work in this cycle. If you have none, exit without changing
any file. Do not create work that the logs do not call for.

The **Routing** section of your role, appended below, defines when you have work.
Apply it first.

```python
import sys
from pathlib import Path

RUN_ROOT = Path(".")            # this file sits in the run directory
sys.path.insert(0, str(RUN_ROOT))
from system.helper import (DeadEndRecord, DerivationRecord, ExperimentRecord,
                           FolderState, ObjectiveProposal, SessionRecord, proposal_id)

state = FolderState(RUN_ROOT)
```

Work from the run directory, which is the directory containing this file.

## Part 1: Read the task

Set your agent name, then read the task and the logs.

```python
AGENT_NAME = "<your role, as named at the top of this file>"

meta, body = state.read_task()       # TASK.md parameters and description
experiments = state.read_experiments()
```

The Theorist also reads the champion and the dead-end log, because each new
proposal starts from them:

```python
champion = state.read_champion()     # objective, derivation, module
closed   = state.read_dead_ends()    # rejected proposals and the classes they rule out
```

`meta` contains every fixed parameter of the run: the data sets, the metric, the
number of restarts, the kernel-width grids and the promotion rule. **Read these
values from `meta` and do not hardcode them.** The user sets them. A different
protocol requires a new task file, not a code change.

`meta` does not specify what to propose. This is the open question of the task.

Read `task-profile.md` for the settings of this run.

---

## Part 2: Work

Carry out your role as described in the section "Your Role" appended below.

Do only the work of your role. The Theorist does not derive update rules, the
Analyst does not implement them, and the Evaluator does not decide whether a
result is good enough. That decision is made by the promotion rule.

---

## Part 3: Exit

Record your cycle in the session log, then stop.

```python
state.append_session(SessionRecord(
    agent=AGENT_NAME,
    action="<proposed, declined, derived, refused, evaluated or audited>",
    detail="one line on what you did and why",
))
```

Then print a closing line, so the orchestrator knows that you have finished:

```
[CYCLE-DONE] <role> — <one line>
```

Stop after this line. The orchestrator starts the next agent.
