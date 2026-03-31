#!/usr/bin/env python3
# Harness: planning -- keeping the model on course without scripting the route.
"""
s03_todo_write.py - TodoWrite

The model tracks its own progress via a TodoManager. A nag reminder
forces it to keep updating when it forgets.

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

Key insight: "The agent can track its own progress -- and I can see it."
"""

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

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# -- TodoManager: structured state the LLM writes to --
# The model can call the todo tool to track multi-step tasks.
# This class validates and manages the todo list state.
#
# Todo item structure (what the LLM sends):
#   {
#       "id": "1",              # unique identifier for the task
#       "text": "Fix the bug",   # description of the task
#       "status": "pending"     # one of: "pending", "in_progress", "completed"
#   }
#
# The todo tool's input_schema expects an array of items:
#   "items": [
#       {"id": "1", "text": "Task 1", "status": "pending"},
#       {"id": "2", "text": "Task 2", "status": "in_progress"}
#   ]
#
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        """
        Validate and update the todo list.
        
        IMPORTANT: This receives the FULL list from the LLM each time.
        The LLM decides which status to change (pending→in_progress→completed)
        and sends the entire updated list. We don't incrementally update -
        we just validate and replace self.items with the new list.
        
        Example flow:
          LLM sends: [{"id": "1", "status": "completed"}, {"id": "2", "status": "in_progress"}]
          Previous: [{"id": "1", "status": "in_progress"}, {"id": "2", "status": "pending"}]
          Result:    [{"id": "1", "status": "completed"}, {"id": "2", "status": "in_progress"}]
        
        Validation rules:
        - Max 20 items (prevent context bloat)
        - Each item must have text
        - Status must be one of: pending, in_progress, completed
        - Only ONE item can be in_progress at a time (prevents parallel work)
        
        Args:
            items: Full list of dicts with id, text, status keys (from block.input["items"])
        """
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        """
        Convert the todo list to a readable string for the LLM.
        
        WHY SEND BACK TO LLM?
        After the LLM calls todo tool with new status updates, the LLM needs
        to SEE the current state to reason about next steps. The LLM generates
        updates but doesn't automatically remember them - we send them back
        as tool_result content so the LLM can read the todo list in context.
        
        Example flow:
          1. LLM calls: todo(items=[{id:"1", status:"in_progress"}])
          2. We update: self.items = [{id:"1", status:"in_progress"}]
          3. We render: "[>] #1: Task A\n(0/1 completed)"
          4. Send back as tool_result for next LLM turn
        
        Output format:
          [ ] #1: Task A         <- pending
          [>] #2: Task B         <- in_progress (current task)
          [x] #3: Task C         <- completed
          
          (1/3 completed)         <- progress summary
        """
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


# -- Tool implementations --
# Same as s02, but with safe_path and path validation for file operations

def safe_path(p: str) -> Path:
    """
    Resolve a relative path and ensure it stays within WORKDIR.
    Prevents path traversal attacks (e.g., ../../../etc/passwd).
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """Execute a shell command with safety checks and 120s timeout."""
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
    """Write content to file, creating parent directories if needed."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace first occurrence of old_text with new_text in a file."""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # The todo tool calls TODO.update() with items from block.input["items"]
    # block.input structure: {"items": [{"id": "1", "text": "Task", "status": "pending"}, ...]}
    "todo":       lambda **kw: TODO.update(kw["items"]),
}

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "todo", "description": "Update task list. Track progress on multi-step tasks.",
     # The todo tool takes an array of items, each with id, text, and status
     # status must be one of: "pending", "in_progress", "completed"
     "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["id", "text", "status"]}}}, "required": ["items"]}},
]


# -- Agent loop with nag reminder injection --
# This version adds a "nag reminder" that reminds the model to update todos
# if it hasn't called the todo tool for 3+ rounds.
#
# Flow per round:
#   1. Call LLM with tools (including todo) and current messages
#   2. LLM responds - either done or calls tools
#   3. Execute each tool call, track if todo was used
#   4. If todo was used: send back rendered todo list (via tool_result)
#   5. Track rounds since last todo call
#   6. If 3+ rounds without calling todo: inject reminder as separate message
#   7. Continue loop with tool results + optional reminder
#
# Why separate reminder message?
#   - Anthropic requires tool_result blocks immediately after tool_use blocks
#   - Inserting text before tool_results would violate this rule
#   - So reminder goes in its own user message after the tool results
#
def agent_loop(messages: list):
    """
    Agent loop with todo tracking and nag reminder.
    
    Key differences from s02:
    - Tracks rounds_since_todo counter
    - Checks if LLM called the todo tool in each round
    - Injects reminder after 3 rounds without todo call
    
    When LLM calls todo tool:
      - block.name == "todo"
      - block.input["items"] = [{"id": "1", "text": "...", "status": "pending"}, ...]
      - handler executes TODO.update(items)
      - tool_result content = rendered todo list (see render())
    
    When todo NOT called for 3 rounds:
      - rounds_since_todo increments each round
      - At 3+, inject <reminder> as separate user message
      - This nudges the LLM to call todo tool
    """
    rounds_since_todo = 0
    while True:
        # Step 1: Call LLM with conversation history and available tools
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        
        # Step 2: If LLM didn't call a tool, we're done with this turn
        if response.stop_reason != "tool_use":
            return
        
        # Step 3: Execute each tool the LLM called
        results = []
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
                # Look up handler (bash, read_file, write_file, edit_file, or todo)
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    # block.input contains the arguments the LLM passed
                    # For todo: {"items": [{"id": "1", "text": "...", "status": "pending"}, ...]}
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                
                # Build tool_result - this is what the LLM sees in next turn
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                
                # Track if LLM called the todo tool
                if block.name == "todo":
                    used_todo = True
        
        # Step 4: Track rounds since last todo call
        # If todo was called this round, reset counter to 0
        # Otherwise increment - after 3, we'll inject a reminder
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        
        # Step 5: Append tool results as user message to continue loop
        messages.append({"role": "user", "content": results})
        
        # Step 6: Nag reminder injection
        # If model hasn't called todo in 3+ rounds, inject a reminder
        # IMPORTANT: sent as separate message (not inside results) because
        # tool_result blocks must immediately follow tool_use blocks
        if rounds_since_todo >= 3:
            messages.append({"role": "user", "content": [{"type": "text", "text": "<reminder>Update your todos.</reminder>"}]})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
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
