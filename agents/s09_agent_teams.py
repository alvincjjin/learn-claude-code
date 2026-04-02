#!/usr/bin/env python3
# Harness: team mailboxes -- multiple models, coordinated through files.
"""
s09_agent_teams.py - Agent Teams

Persistent named agents with file-based JSONL inboxes. Each teammate runs
its own agent loop in a separate thread. Communication via append-only inboxes.

    Subagent (s04):  spawn -> execute -> return summary -> destroyed
    Teammate (s09):  spawn -> work -> idle -> work -> ... -> shutdown

    .team/config.json                   .team/inbox/
    +----------------------------+      +------------------+
    | {"team_name": "default",   |      | alice.jsonl      |
    |  "members": [              |      | bob.jsonl        |
    |    {"name":"alice",        |      | lead.jsonl       |
    |     "role":"coder",        |      +------------------+
    |     "status":"idle"}       |
    |  ]}                        |      send_message("alice", "fix bug"):
    +----------------------------+        open("alice.jsonl", "a").write(msg)

                                    read_inbox("alice"):
    spawn_teammate("alice","coder",...)   msgs = [json.loads(l) for l in ...]
         |                                open("alice.jsonl", "w").close()
         v                                return msgs  # drain
    Thread: alice runs, completes, exits. config: "idle" (thread is DEAD)

    5 message types (all declared, not all handled here):
    +-------------------------+-----------------------------------+
    | message                 | Normal text message               |
    | broadcast               | Sent to all teammates             |
    | shutdown_request        | Request graceful shutdown (s10)   |
    | shutdown_response       | Approve/reject shutdown (s10)     |
    | plan_approval_response  | Approve/reject plan (s10)         |
    +-------------------------+-----------------------------------+

Key insight: "Teammates that can talk to each other."

Difference from s04 (subagents):
    s04: Spawns a subagent that runs once and returns. Like a function call.
    s09: Spawns a persistent teammate that stays alive, checks inbox periodically,
         can receive messages, go idle, and wake up again. Like a long-lived worker.
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

# Valid message types for inter-agent communication
VALID_MSG_TYPES = {
    "message",                  # Normal text message
    "broadcast",                # Sent to all teammates
    "shutdown_request",         # Request graceful shutdown (s10)
    "shutdown_response",        # Approve/reject shutdown (s10)
    "plan_approval_response",  # Approve/reject plan (s10)
}


# =============================================================================
# MessageBus: JSONL inbox per teammate
# =============================================================================
#
# PROBLEM: Threads need to communicate but can't share memory safely.
# SOLUTION: File-based mailboxes using JSONL (JSON Lines) format.
#
# How it works:
#   - Each teammate has an inbox file: .team/inbox/{name}.jsonl
#   - send_message() appends a JSON line to the recipient's inbox
#   - read_inbox() reads all lines, then truncates file (drains it)
#
# Why JSONL?
#   - Append-only = atomic writes (no file corruption)
#   - Simple, human-readable format
#   - Easy to debug (cat inbox.jsonl)
#
# Thread safety:
#   - Single writer at a time (one message per append)
#   - read_inbox drains after reading (no duplicates)
#   - For high concurrency, would need file locking
#
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """
        Send a message to a teammate's inbox.
        
        Args:
            sender: Name of sender
            to: Recipient's inbox name
            content: Message text
            msg_type: One of VALID_MSG_TYPES
            extra: Additional fields to include
            
        Appends a JSON line to {to}.jsonl (append-only, atomic).
        """
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """
        Read and drain a teammate's inbox.
        
        Args:
            name: Inbox owner (teammate name)
            
        Returns:
            List of message dicts (all messages in inbox)
            
        How it works:
            1. Read ALL lines from {name}.jsonl
            2. Empty the file (write_text("")) - messages delivered exactly once
            3. Next call only gets NEW messages added since last read
        """
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")  # Drain the inbox
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """
        Send a message to all teammates except sender.
        
        Args:
            sender: Name of sender (excluded from broadcast)
            content: Message text
            teammates: List of all teammate names
        """
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# Global message bus instance
BUS = MessageBus(INBOX_DIR)


# =============================================================================
# TeammateManager: persistent named agents with config.json
# =============================================================================
#
# PROBLEM: Need persistent agents that survive beyond single tool calls.
# SOLUTION: Long-running threads with config stored in .team/config.json
#
# Teammate lifecycle:
#   1. spawn() creates config entry and starts daemon thread
#   2. Thread runs _teammate_loop() - executes tools until task done
#   3. When done, thread EXITS and status = "idle" (written to config)
#   4. "idle" means: thread is GONE, config record remains as availability flag
#   5. To wake up: call spawn_teammate() again with new prompt
#
# IMPORTANT: "idle" != thread waiting. Thread is dead. It's just a config flag
#             saying "alice is available to be respawned with a new task."
#
# Config persistence:
#   - .team/config.json stores team name and member list
#   - Each member has: name, role, status
#   - Status: "working" (thread running), "idle" (thread dead, can respawn), "shutdown" (thread dead, permanently stopped)
#
# Thread safety:
#   - config is read at start, written on spawn/status change
#   - daemon=True means process exits when main thread exits
#   - Python GIL: only one thread executes Python bytecode at a time
#   - I/O releases GIL -> concurrent I/O works
#   - subprocess runs in separate OS process -> bypasses GIL entirely
#
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        """Load team config from disk, or return default empty config."""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        """Save team config to disk."""
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        """Find a member by name in config."""
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        Spawn a persistent teammate.
        
        Args:
            name: Teammate identifier
            role: Role (e.g., "coder", "reviewer", "tester")
            prompt: Initial task prompt
            
        Returns:
            Status string
        """
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        
        # Start daemon thread for this teammate
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """
        Main loop for a persistent teammate.
        
        Each iteration:
            1. Check inbox for new messages
            2. Add messages to conversation history
            3. Call LLM with tools
            4. Execute tools
            5. Repeat until no tool calls (task complete)
            6. Set status to "idle"
        
        Runs up to 50 iterations, then exits gracefully.
        """
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()
        
        for _ in range(50):
            # Check inbox for new messages
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})
            
            try:
                response = client.messages.create(
                    model=MODEL,
                    system=sys_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=8000,
                )
            except Exception:
                break
            
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            
            # Execute tool calls
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = self._exec(name, block.name, block.input)
                    print(f"  [{name}] {block.name}: {str(output)[:120]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(output),
                    })
            messages.append({"role": "user", "content": results})
        
        # Task complete, set status to idle (not destroyed)
        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        Execute a tool for a teammate.
        
        Uses dict-based handler pattern for cleaner code.
        Includes base tools and communication tools.
        """
        handlers = {
            "bash": lambda: _run_bash(args["command"]),
            "read_file": lambda: _run_read(args["path"]),
            "write_file": lambda: _run_write(args["path"], args["content"]),
            "edit_file": lambda: _run_edit(args["path"], args["old_text"], args["new_text"]),
            "send_message": lambda: BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message")),
            "read_inbox": lambda: json.dumps(BUS.read_inbox(sender), indent=2),
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown tool: {tool_name}"
        try:
            return handler()
        except KeyError as e:
            return f"Missing required argument: {e}"

    def _teammate_tools(self) -> list:
        """
        Tools available to teammates.
        
        Same as lead's tools minus spawn_teammate (teammates can't spawn).
        """
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
        ]

    def list_all(self) -> str:
        """List all teammates with name, role, status."""
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """Get list of all teammate names."""
        return [m["name"] for m in self.config["members"]]


# Global teammate manager instance
TEAM = TeammateManager(TEAM_DIR)


# =============================================================================
# Base tool implementations (unchanged from s02)
# =============================================================================
def _safe_path(p: str) -> Path:
    """Validate path stays within workspace."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """Execute shell command with safety checks."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    """Read file contents with optional line limit."""
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """Write content to file, creating directories as needed."""
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in file."""
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Lead tool dispatch (9 tools)
# =============================================================================
#
# Tools for the team lead:
#   - Base tools: bash, read_file, write_file, edit_file
#   - Team tools: spawn_teammate, list_teammates, send_message, read_inbox, broadcast
#
TOOL_HANDLERS = {
    "bash":            lambda **kw: _run_bash(kw["command"]),
    "read_file":       lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":      lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":       lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}


# Tool schemas for lead's LLM
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "spawn_teammate", "description": "Spawn a persistent teammate that runs in its own thread.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates with name, role, status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate's inbox.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
]


# =============================================================================
# Agent loop for team lead
# =============================================================================
#
# Key difference from other agents:
#   - Before each LLM call, check lead's inbox for messages from teammates
#   - CRITICAL: Anthropic API requires conversation to end with user message only
#   - Cannot add assistant prefill (violates API rules)
#
def agent_loop(messages: list):
    """
    Main agent loop for team lead.
    
    Flow per turn:
        1. Check lead's inbox for messages
        2. Inject any messages as user content (no assistant prefill!)
        3. Call LLM with tools
        4. Execute tools
        5. Loop back
    """
    while True:
        # Check lead's inbox for messages from teammates
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
            # CRITICAL: Do NOT add assistant prefill - Anthropic requires ending with user message
        
        # Call LLM with current message history and available tools
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        
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
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
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
            query = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        
        # Print assistant's final response (non-tool text)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
