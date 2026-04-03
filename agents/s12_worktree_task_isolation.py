#!/usr/bin/env python3
"""
s12_worktree_task_isolation.py - Worktree + Task Isolation

Directory-level isolation for parallel task execution.
Tasks are the control plane and worktrees are the execution plane.

    .tasks/task_12.json
      {
        "id": 12,
        "subject": "Implement auth refactor",
        "status": "in_progress",
        "worktree": "auth-refactor"
      }

    .worktrees/index.json
      {
        "worktrees": [
          {
            "name": "auth-refactor",
            "path": ".../.worktrees/auth-refactor",
            "branch": "wt/auth-refactor",
            "task_id": 12,
            "status": "active"
          }
        ]
      }

Key insight: "Isolate by directory, coordinate by task ID."

This harness provides:
- Task management: create, list, update, bind/unbind worktrees
- Worktree management: create, list, run commands, keep, remove
- Event tracking: lifecycle events for observability
- Directory isolation: each worktree is a separate git worktree
- Parallel execution: multiple tasks can run in separate worktrees simultaneously
"""

# Standard library imports
import json
import os
import re
import subprocess
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file, overriding any existing ones
# This allows customization of ANTHROPIC_BASE_URL and other settings
load_dotenv(override=True)

# Remove any existing auth token if using a custom base URL
# This prevents conflicts between different API endpoints
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# WORKDIR is the current working directory where the agent operates
# All file operations are scoped to this directory for safety
WORKDIR = Path.cwd()

# Create Anthropic client with optional custom base URL
# Falls back to default Anthropic API if ANTHROPIC_BASE_URL is not set
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# MODEL_ID is the AI model to use for agentic tasks
# Set via environment variable (e.g., export MODEL_ID="claude-3-opus-20240229")
MODEL = os.environ["MODEL_ID"]


def detect_repo_root(cwd: Path) -> Path | None:
    """
    Return git repo root if cwd is inside a repo, else None.
    
    This function walks up the directory tree to find the git repository root.
    It uses 'git rev-parse --show-toplevel' which is a standard git command
    that returns the absolute path of the top-level directory of the repository.
    
    Args:
        cwd: The current working directory to check
        
    Returns:
        Path to the git repository root, or None if not in a git repo
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return None
        root = Path(r.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None


# REPO_ROOT is the git repository root, used for worktree operations
# It defaults to WORKDIR if not inside a git repository
REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR

# SYSTEM is the system prompt that defines the agent's role and capabilities
# It tells the agent to use task + worktree tools for multi-task work
# and emphasizes directory isolation for parallel or risky changes
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task + worktree tools for multi-task work. "
    "For parallel or risky changes: create tasks, allocate worktree lanes, "
    "run commands in those lanes, then choose keep/remove for closeout. "
    "Use worktree_events when you need lifecycle visibility."
)


# -- EventBus: append-only lifecycle events for observability --
class EventBus:
    """
    EventBus provides append-only logging for lifecycle events.
    
    Events are stored in a JSONL (JSON Lines) file for easy streaming
    and parsing. Each event includes a timestamp, event type, and
    optional task/worktree information.
    
    Event types:
    - worktree.create.before/after/failed
    - worktree.remove.before/after/failed
    - worktree.keep
    - task.completed
    
    Usage:
        events = EventBus(Path(".worktrees/events.jsonl"))
        events.emit("worktree.create.after", task={"id": 1}, worktree={"name": "foo"})
    """
    
    def __init__(self, event_log_path: Path):
        """
        Initialize EventBus with a path for the event log file.
        
        Creates the parent directory if needed and initializes an empty
        log file if it doesn't exist.
        
        Args:
            event_log_path: Path to the JSONL event log file
        """
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def emit(
        self,
        event: str,
        task: dict | None = None,
        worktree: dict | None = None,
        error: str | None = None,
    ):
        """
        Emit a lifecycle event to the event log.
        
        Appends a JSON object to the event log file. Each event contains:
        - event: string identifier for the event type
        - ts: Unix timestamp
        - task: optional task information
        - worktree: optional worktree information
        - error: optional error message if event represents a failure
        
        Args:
            event: The event type (e.g., "worktree.create.after")
            task: Optional dict with task information (e.g., {"id": 12})
            worktree: Optional dict with worktree information
            error: Optional error message if this is a failure event
        """
        payload = {
            "event": event,
            "ts": time.time(),
            "task": task or {},
            "worktree": worktree or {},
        }
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def list_recent(self, limit: int = 20) -> str:
        """
        Retrieve recent events from the event log.
        
        Reads the event log file and returns the most recent N events
        as a formatted JSON string.
        
        Args:
            limit: Maximum number of events to return (default 20, max 200)
            
        Returns:
            JSON string containing the recent events
        """
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:]
        items = []
        for line in recent:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2)


# -- TaskManager: persistent task board with optional worktree binding --
class TaskManager:
    """
    TaskManager provides a persistent task board stored as JSON files.
    
    Tasks are individual work items that can be tracked and optionally
    bound to a git worktree for isolated execution. Each task has:
    - id: unique identifier
    - subject: short description
    - description: detailed description
    - status: one of "pending", "in_progress", "completed"
    - owner: agent or person responsible
    - worktree: name of bound worktree (if any)
    - blockedBy: list of task IDs this depends on
    
    Tasks are stored in .tasks/task_{id}.json files.
    
    Usage:
        tasks = TaskManager(Path(".tasks"))
        tasks.create("Fix authentication bug", "Investigate login flow...")
        tasks.bind_worktree(1, "auth-fix")
    """
    
    def __init__(self, tasks_dir: Path):
        """
        Initialize TaskManager with a tasks directory.
        
        Creates the directory if needed and scans for existing tasks
        to determine the next available ID.
        
        Args:
            tasks_dir: Path to the .tasks directory
        """
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """Find the highest task ID currently in use."""
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                pass
        return max(ids) if ids else 0

    def _path(self, task_id: int) -> Path:
        """Get the file path for a task by ID."""
        return self.dir / f"task_{task_id}.json"

    def _load(self, task_id: int) -> dict:
        """
        Load a task from disk by ID.
        
        Args:
            task_id: The task ID to load
            
        Returns:
            Dict containing task data
            
        Raises:
            ValueError: If task file doesn't exist
        """
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        """
        Save a task to disk.
        
        Args:
            task: Dict containing task data with 'id' key
        """
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        """
        Create a new task on the task board.
        
        Allocates a new task ID and creates a task file with
        initial status "pending".
        
        Args:
            subject: Short description of the task
            description: Optional detailed description
            
        Returns:
            JSON string representation of the created task
        """
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": "",
            "worktree": "",
            "blockedBy": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        """
        Get a task by ID.
        
        Args:
            task_id: The task ID to retrieve
            
        Returns:
            JSON string representation of the task
        """
        return json.dumps(self._load(task_id), indent=2)

    def exists(self, task_id: int) -> bool:
        """Check if a task exists by ID."""
        return self._path(task_id).exists()

    def update(self, task_id: int, status: str = None, owner: str = None) -> str:
        """
        Update a task's status or owner.
        
        Args:
            task_id: The task ID to update
            status: New status ("pending", "in_progress", "completed")
            owner: New owner string
            
        Returns:
            JSON string representation of the updated task
            
        Raises:
            ValueError: If status is invalid
        """
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        if owner is not None:
            task["owner"] = owner
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        """
        Bind a task to a worktree name.
        
        Associates a task with a worktree for isolated execution.
        Automatically changes status to "in_progress" if currently "pending".
        
        Args:
            task_id: The task ID to bind
            worktree: Name of the worktree
            owner: Optional owner string
            
        Returns:
            JSON string representation of the updated task
        """
        task = self._load(task_id)
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        if task["status"] == "pending":
            task["status"] = "in_progress"
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def unbind_worktree(self, task_id: int) -> str:
        """
        Unbind a task from its worktree.
        
        Removes the worktree association without changing status.
        
        Args:
            task_id: The task ID to unbind
            
        Returns:
            JSON string representation of the updated task
        """
        task = self._load(task_id)
        task["worktree"] = ""
        task["updated_at"] = time.time()
        self._save(task)
        return json.dumps(task, indent=2)

    def list_all(self) -> str:
        """
        List all tasks with status, owner, and worktree binding.
        
        Returns a formatted text list of all tasks, similar to:
        [>] #12: Implement auth refactor owner=alice wt=auth-refactor
        [ ] #13: Fix bug in payment flow
        
        Returns:
            Formatted string listing all tasks
        """
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(t["status"], "[?]")
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            wt = f" wt={t['worktree']}" if t.get("worktree") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{wt}")
        return "\n".join(lines)


TASKS = TaskManager(REPO_ROOT / ".tasks")
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")


# -- WorktreeManager: create/list/run/remove git worktrees + lifecycle index --
class WorktreeManager:
    """
    WorktreeManager handles git worktree operations for isolated task execution.
    
    Git worktrees allow checking out multiple branches simultaneously in
    separate directories. This enables true parallel execution where
    multiple tasks can be worked on without interfering with each other.
    
    Worktrees are tracked in .worktrees/index.json with metadata including:
    - name: identifier for the worktree
    - path: absolute path to the worktree directory
    - branch: git branch name (prefixed with wt/)
    - task_id: optional bound task ID
    - status: one of "active", "kept", "removed"
    - created_at/kept_at/removed_at: timestamps
    
    Usage:
        wm = WorktreeManager(repo_root, tasks, events)
        wm.create("auth-refactor", task_id=12)
        wm.run("auth-refactor", "npm test")
        wm.keep("auth-refactor")  # or wm.remove("auth-refactor", complete_task=True)
    """
    
    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        """
        Initialize WorktreeManager.
        
        Args:
            repo_root: Path to the git repository root
            tasks: TaskManager instance for task binding
            events: EventBus instance for lifecycle events
        """
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))
        self.git_available = self._is_git_repo()

    def _is_git_repo(self) -> bool:
        """Check if the current directory is inside a git repository."""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _run_git(self, args: list[str]) -> str:
        """
        Run a git command in the repository root.
        
        Args:
            args: List of git command arguments (e.g., ["worktree", "add", "-b", "wt/feature"])
            
        Returns:
            Git command output as string
            
        Raises:
            RuntimeError: If git is not available or command fails
        """
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            msg = (r.stdout + r.stderr).strip()
            raise RuntimeError(msg or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"

    def _load_index(self) -> dict:
        """Load the worktree index from disk."""
        return json.loads(self.index_path.read_text())

    def _save_index(self, data: dict):
        """Save the worktree index to disk."""
        self.index_path.write_text(json.dumps(data, indent=2))

    def _find(self, name: str) -> dict | None:
        """Find a worktree by name in the index."""
        idx = self._load_index()
        for wt in idx.get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None

    def _validate_name(self, name: str):
        """
        Validate worktree name format.
        
        Names must be 1-40 characters and contain only letters,
        numbers, dots, underscores, and hyphens.
        
        Args:
            name: The worktree name to validate
            
        Raises:
            ValueError: If name is invalid
        """
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
            )

    def create(self, name: str, task_id: int = None, base_ref: str = "HEAD") -> str:
        """
        Create a new git worktree.
        
        Creates a new worktree directory with a separate branch.
        Optionally binds to a task for lifecycle tracking.
        
        Args:
            name: Name for the worktree (used in path and branch)
            task_id: Optional task ID to bind to
            base_ref: Git reference to base on (default: HEAD)
            
        Returns:
            JSON string with created worktree details
            
        Raises:
            ValueError: If name is invalid or already exists
        """
        self._validate_name(name)
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")

        path = self.dir / name
        branch = f"wt/{name}"
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else {},
            worktree={"name": name, "base_ref": base_ref},
        )
        try:
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])

            entry = {
                "name": name,
                "path": str(path),
                "branch": branch,
                "task_id": task_id,
                "status": "active",
                "created_at": time.time(),
            }

            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)

            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)

            self.events.emit(
                "worktree.create.after",
                task={"id": task_id} if task_id is not None else {},
                worktree={
                    "name": name,
                    "path": str(path),
                    "branch": branch,
                    "status": "active",
                },
            )
            return json.dumps(entry, indent=2)
        except Exception as e:
            self.events.emit(
                "worktree.create.failed",
                task={"id": task_id} if task_id is not None else {},
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),
            )
            raise

    def list_all(self) -> str:
        """
        List all tracked worktrees.
        
        Returns formatted output like:
        [active] auth-refactor -> /path/.worktrees/auth-refactor (wt/auth-refactor) task=12
        
        Returns:
            Formatted string listing all worktrees
        """
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "No worktrees in index."
        lines = []
        for wt in wts:
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt['name']} -> "
                f"{wt['path']} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)

    def status(self, name: str) -> str:
        """
        Show git status for a worktree.
        
        Runs 'git status --short --branch' in the worktree directory
        to show current changes and branch state.
        
        Args:
            name: Name of the worktree
            
        Returns:
            Git status output as string
        """
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (r.stdout + r.stderr).strip()
        return text or "Clean worktree"

    def run(self, name: str, command: str) -> str:
        """
        Run a shell command in a worktree directory.
        
        Executes the given command in the worktree's directory,
        allowing isolated execution of build/test commands.
        
        Args:
            name: Name of the worktree
            command: Shell command to execute
            
        Returns:
            Command output as string (truncated to 50k chars)
        """
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Error: Dangerous command blocked"

        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            return f"Error: Worktree path missing: {path}"

        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (300s)"

    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        """
        Remove a git worktree.
        
        Deletes the worktree directory and optionally marks the
        bound task as completed.
        
        Args:
            name: Name of the worktree to remove
            force: If True, use --force to remove even with uncommitted changes
            complete_task: If True, mark the bound task as completed
            
        Returns:
            Success message string
        """
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )
        try:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)

            if complete_task and wt.get("task_id") is not None:
                task_id = wt["task_id"]
                before = json.loads(self.tasks.get(task_id))
                self.tasks.update(task_id, status="completed")
                self.tasks.unbind_worktree(task_id)
                self.events.emit(
                    "task.completed",
                    task={
                        "id": task_id,
                        "subject": before.get("subject", ""),
                        "status": "completed",
                    },
                    worktree={"name": name},
                )

            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)

            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )
            return f"Removed worktree '{name}'"
        except Exception as e:
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise

    def keep(self, name: str) -> str:
        """
        Mark a worktree as kept without removing it.
        
        This is used when the worktree's changes should be preserved
        (e.g., merged back to main). The worktree remains in the index
        but with status "kept".
        
        Args:
            name: Name of the worktree to keep
            
        Returns:
            JSON string with worktree details
        """
        wt = self._find(name)
        if not wt:
            return f"Error: Unknown worktree '{name}'"

        idx = self._load_index()
        kept = None
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)

        self.events.emit(
            "worktree.keep",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={
                "name": name,
                "path": wt.get("path"),
                "status": "kept",
            },
        )
        return json.dumps(kept, indent=2) if kept else f"Error: Unknown worktree '{name}'"


WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)


# -- Base tools: core file operations with safety checks --
def safe_path(p: str) -> Path:
    """
    Resolve and validate a path to prevent directory traversal.
    
    Ensures the resolved path is within the working directory
    to prevent accessing files outside the workspace.
    
    Args:
        p: Relative path string
        
    Returns:
        Resolved Path object
        
    Raises:
        ValueError: If path escapes the working directory
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    Run a shell command in the current workspace.
    
    Blocks until command completes. Includes safety checks
    to block dangerous commands.
    
    Args:
        command: Shell command string
        
    Returns:
        Command output as string (truncated to 50k chars)
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """
    Read file contents.
    
    Reads the entire file and returns as string. Optionally
    limits the number of lines returned.
    
    Args:
        path: File path (relative to workspace)
        limit: Optional maximum number of lines to return
        
    Returns:
        File contents as string (truncated to 50k chars)
    """
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    Write content to file.
    
    Creates parent directories if needed. Overwrites existing files.
    
    Args:
        path: File path (relative to workspace)
        content: Content to write
        
    Returns:
        Success message with byte count
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in file.
    
    Finds the first occurrence of old_text and replaces it with new_text.
    Only replaces one instance (use run_write for multiple).
    
    Args:
        path: File path (relative to workspace)
        old_text: Text to find and replace
        new_text: Replacement text
        
    Returns:
        Success message
    """
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Tool handlers: map tool names to Python functions --
# Each tool name from TOOLS maps to a handler function that
# receives keyword arguments from the tool's input_schema
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner")),
    "task_bind_worktree": lambda **kw: TASKS.bind_worktree(kw["task_id"], kw["worktree"], kw.get("owner", "")),
    "worktree_create": lambda **kw: WORKTREES.create(kw["name"], kw.get("task_id"), kw.get("base_ref", "HEAD")),
    "worktree_list": lambda **kw: WORKTREES.list_all(),
    "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
    "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
    "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
    "worktree_remove": lambda **kw: WORKTREES.remove(kw["name"], kw.get("force", False), kw.get("complete_task", False)),
    "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
}

# -- Tool definitions: JSON schemas for Claude's tool_use blocks --
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the current workspace (blocking).",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "task_create",
        "description": "Create a new task on the shared task board.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["subject"],
        },
    },
    {
        "name": "task_list",
        "description": "List all tasks with status, owner, and worktree binding.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "task_get",
        "description": "Get task details by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "integer"}},
            "required": ["task_id"],
        },
    },
    {
        "name": "task_update",
        "description": "Update task status or owner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],
                },
                "owner": {"type": "string"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "task_bind_worktree",
        "description": "Bind a task to a worktree name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "integer"},
                "worktree": {"type": "string"},
                "owner": {"type": "string"},
            },
            "required": ["task_id", "worktree"],
        },
    },
    {
        "name": "worktree_create",
        "description": "Create a git worktree and optionally bind it to a task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "task_id": {"type": "integer"},
                "base_ref": {"type": "string"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_list",
        "description": "List worktrees tracked in .worktrees/index.json.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "worktree_status",
        "description": "Show git status for one worktree.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_run",
        "description": "Run a shell command in a named worktree directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "command": {"type": "string"},
            },
            "required": ["name", "command"],
        },
    },
    {
        "name": "worktree_remove",
        "description": "Remove a worktree and optionally mark its bound task completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "force": {"type": "boolean"},
                "complete_task": {"type": "boolean"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "worktree_keep",
        "description": "Mark a worktree as kept in lifecycle state without removing it.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "worktree_events",
        "description": "List recent worktree/task lifecycle events from .worktrees/events.jsonl.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
]


def agent_loop(messages: list):
    """
    Main agent loop for handling tool calls.
    
    Sends messages to the Claude API with available tools,
    handles tool_use responses by executing handlers,
    and continues until the model returns a text response.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
    """
    while True:
        # Send request to Claude API with system prompt, messages, and tools
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        # Add assistant response to message history
        messages.append({"role": "assistant", "content": response.content})
        
        # If no tool use requested, we're done - return the response
        if response.stop_reason != "tool_use":
            return

        # Handle tool use requests
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # Look up handler function for this tool
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    # Execute handler with input parameters
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    }
                )
        # Add tool results back to message history for next iteration
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    Main entry point for the interactive agent.
    
    Initializes the agent with available tools and runs
    an interactive loop reading commands from stdin.
    
    Usage:
        python s12_worktree_task_isolation.py
        
    Commands:
        - Any text: sent to the agent for processing
        - q, exit, or empty line: terminates the session
    """
    print(f"Repo root for s12: {REPO_ROOT}")
    if not WORKTREES.git_available:
        print("Note: Not in a git repo. worktree_* tools will return errors.")

    history = []
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
