#!/usr/bin/env python3
# Harness: persistent tasks -- goals that outlive any single conversation.
"""
s07_task_system.py - Tasks

Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task has a dependency graph (blockedBy/blocks).

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], "blocks":[], ...}

    Dependency resolution:
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- completing task 1 removes it from task 2's blockedBy

Key insight: "State that survives compression -- because it's outside the conversation."
"""

import json
import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
TASKS_DIR = WORKDIR / ".tasks"

SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."


# =============================================================================
# TaskManager: CRUD with dependency graph, persisted as JSON files
# =============================================================================
#
# PROBLEM: Context compression (s06) clears conversation history, losing task state.
# SOLUTION: Store tasks as JSON files in .tasks/ directory - outside the conversation.
#
# Task data structure:
#   {
#     "id": 1,                        # Unique ID (auto-incremented)
#     "subject": "Fix login bug",      # Short title
#     "description": "Users can't...", # Optional details
#     "status": "pending",            # pending | in_progress | completed
#     "blockedBy": [2, 3],            # Task IDs that must complete first
#     "blocks": [],                   # Task IDs waiting on this one
#     "owner": ""                     # Reserved for future multi-agent use
#   }
#
# Why JSON files?
#   - Simple, human-readable, easy to debug
#   - Survives context compression (stored on disk)
#   - Can be version-controlled if needed
#
# Thread safety: This is single-agent, no locks needed. For multi-agent,
# would need file locking or a database.
#
class TaskManager:
    """
    Manages persistent tasks with dependency tracking.
    
    All tasks are stored as individual JSON files in TASKS_DIR.
    Each task can block or be blocked by other tasks.
    
    Example workflow:
        1. Create task: task_create(subject="A")
        2. Create dependent: task_create(subject="B")
        3. Link them: task_update(task_id=2, addBlockedBy=[1])
        4. Complete task 1: task_update(task_id=1, status="completed")
           -> This automatically removes task 1 from task 2's blockedBy list
    """
    
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        # Find max existing ID to auto-increment next task ID
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """Find the highest existing task ID by scanning task_*.json files."""
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        """Load a single task from disk by ID."""
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        """Write a task to disk."""
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        """Create a new task with auto-incremented ID. Returns JSON string of created task."""
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],  # No dependencies by default
            "blocks": [],     # No tasks waiting on this by default
            "owner": "",      # Reserved for future use
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        """Get full details of a task by ID. Returns JSON string."""
        return json.dumps(self._load(task_id), indent=2)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        """
        Update task properties including dependencies.
        
        Args:
            task_id: ID of task to update
            status: New status (pending/in_progress/completed)
            add_blocked_by: List of task IDs that must complete before this one
            add_blocks: List of task IDs that depend on this one
            
        Bidirectional dependency management:
            When add_blocks=[X], we automatically add this task to X's blockedBy.
        """
        task = self._load(task_id)
        
        # Update status
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            
            # When a task is completed, clear it from all other tasks' blockedBy
            if status == "completed":
                self._clear_dependency(task_id)
        
        # Add tasks that must complete before this one
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        
        # Add tasks that depend on this one
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            
            # Bidirectional sync: also update the blocked tasks' blockedBy lists
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass  # Task might not exist yet
        
        self._save(task)
        return json.dumps(task, indent=2)

    def _clear_dependency(self, completed_id: int):
        """Remove completed task from all other tasks' blockedBy lists."""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        """List all tasks with status and dependency info."""
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        
        if not tasks:
            return "No tasks."
        
        lines = []
        for t in tasks:
            # Status markers: [ ] pending, [>] in_progress, [x] completed
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        
        return "\n".join(lines)


# Global task manager instance - shared across all agent interactions
TASKS = TaskManager(TASKS_DIR)


# =============================================================================
# Base tool implementations (same as other agents)
# =============================================================================
def safe_path(p: str) -> Path:
    """Validate path stays within workspace."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """Execute shell command with safety checks."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    """Read file contents with optional line limit."""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    """Write content to file, creating directories as needed."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file."""
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Tool definitions for the agent
# =============================================================================
#
# Tool handlers map tool names to Python functions.
# Uses kw["key"] for required params, kw.get("key") for optional.
#
# Task tools (unique to s07):
#   - task_create: Create new task
#   - task_update: Update status or dependencies
#   - task_list: Show all tasks
#   - task_get: Show single task details
#
TOOL_HANDLERS = {
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("addBlocks")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
}


# Tool schemas define inputs the LLM can use to call each tool.
# input_schema follows Anthropic's JSON Schema format.
#
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


# =============================================================================
# Agent loop - handles tool calls and returns
# =============================================================================
#
# Standard agent loop pattern (same as other agents):
#   1. Call LLM with messages + tools
#   2. If no tool_use, return response
#   3. Execute tools, collect results
#   4. Append results as user message, loop back
#
def agent_loop(messages: list):
    """Main agent loop with task management tools."""
    while True:
        # Call LLM with current message history and available tools
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})
        
        # If LLM didn't call any tools, we're done
        if response.stop_reason != "tool_use":
            return
        
        # Execute tool calls and collect results
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                
                print(f"> {block.name}: {str(output)[:200]}")
                
                # Build tool_result for the LLM
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)
                })
        
        # Append tool results as user message to continue conversation
        messages.append({"role": "user", "content": results})


# =============================================================================
# Interactive REPL for testing
# =============================================================================
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        
        history.append({"role": "user", "content": query})
        agent_loop(history)
        
        # Print assistant's final response (non-tool text)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
