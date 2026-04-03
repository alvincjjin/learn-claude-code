# Thread vs Subprocess in Python

Quick reference using s01-s12 agent framework examples.

## Thread vs Subprocess

| | Thread | Subprocess |
|---|---|---|
| Runs in | Same Python process | Separate OS process |
| GIL | Affected (one at a time) | Bypassed |
| Use for | Teammate agents, background tasks | Shell commands (bash, git) |
| Examples | s08-s11 | s01-s12 |

---

## Thread: `threading.Thread`

**Run multiple teammates in parallel** (s09-s11):

```python
thread = threading.Thread(
    target=self._teammate_loop,
    args=(name, role, prompt),
    daemon=True,
)
thread.start()
```

**Non-blocking background task** (s08):

```python
thread = threading.Thread(target=self._execute, args=(task_id, cmd), daemon=True)
thread.start()
return f"Task {task_id} started"  # Returns immediately!
```

---

## Lock: `threading.Lock`

**Problem: Race condition** - two threads modify same data at same time:

```python
# BAD: Both threads see task unclaimed, both claim it
task = read_task(id)
task["owner"] = me
write_task(id, task)
```

**Solution: Lock** - only one thread enters at a time:

```python
_claim_lock = threading.Lock()

def claim_task(task_id, owner):
    with _claim_lock:               # Serializes access
        task = read_task(task_id)
        task["owner"] = owner
        write_task(task_id, task)
```

**Locks in agents**:

| Lock | Purpose |
|---|---|
| `_claim_lock` | Task claiming (s11) |
| `_tracker_lock` | Request tracking (s10-s11) |
| `_lock` | Notification queue (s08) |

---

## Subprocess: `subprocess.run`

**Run shell commands**:

```python
r = subprocess.run(
    "git status",
    shell=True,
    cwd=WORKDIR,
    capture_output=True,
    text=True,
    timeout=120,
)
return r.stdout + r.stderr
```

---

## Thread + Subprocess Combined (s08)

```python
def run(command):
    thread = threading.Thread(target=self._execute, args=(task_id, command))
    thread.start()
    return "started"

def _execute(task_id, command):
    subprocess.run(command, shell=True, ...)  # Bypasses GIL
    # Push notification to queue
```

1. Start thread → returns immediately
2. Thread runs subprocess → separate process, no GIL blocking
3. Done → notify agent

---

## Summary

| Need | Solution |
|---|---|
| Run teammates in parallel | `threading.Thread` |
| Don't block on long commands | Thread + subprocess |
| Protect shared data | `threading.Lock` |
| Run shell/git commands | `subprocess.run` |
