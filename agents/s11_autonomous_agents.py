#!/usr/bin/env python3
# Harness: autonomy -- models that find work without being told.
"""
s11_autonomous_agents.py - Autonomous Agents

Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression. Builds on s10's protocols.

    Teammate lifecycle:
    +-------+
    | spawn |
    +---+---+
        |
        v
    +-------+  tool_use    +-------+
    | WORK  | <----------- |  LLM  |
    +---+---+              +-------+
        |
        | stop_reason != tool_use
        v
    +--------+
    | IDLE   | poll every 5s for up to 60s
    +---+----+
        |
        +---> check inbox -> message? -> resume WORK
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> resume WORK
        |
        +---> timeout (60s) -> shutdown

    Identity re-injection after compression:
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"

Key insight: "The agent finds work itself."
"""

import json
import os
import subprocess
import threading
import time
import uuid
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
TASKS_DIR = WORKDIR / ".tasks"

# =============================================================================
# POLL_INTERVAL: How often idle teammate checks for work (in seconds)
# IDLE_TIMEOUT: Max time to stay idle before shutting down (in seconds)
# =============================================================================
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."

# =============================================================================
# Message types for inter-agent communication:
#   - message: general message
#   - broadcast: message sent to all teammates
#   - shutdown_request: lead requests teammate to shut down
#   - shutdown_response: teammate responds to shutdown request
#   - plan_approval_response: lead responds to teammate's plan
# =============================================================================
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# =============================================================================
# Request trackers - IN-MEMORY ONLY, lost on restart.
# Use _tracker_lock for thread-safe access.
# Format: {request_id: {"target": name, "from": name, "status": "pending|approved|rejected"}}
# =============================================================================
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()


# =============================================================================
# MessageBus: JSONL inbox per teammate
# 
# Each teammate has an inbox file (e.g., .team/inbox/coder.jsonl)
# - send() appends JSON (one line per message) to recipient's inbox
# - read_inbox() reads all lines, then empties file (drains the inbox)
# 
# Uses JSONL format for simplicity and atomic appends.
# =============================================================================
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # Send a message to a teammate's inbox.
    # 
    # msg_type must be in VALID_MSG_TYPES to prevent invalid message types.
    # extra dict allows adding fields like request_id for correlation.
    # =============================================================================
    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
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

    # =============================================================================
    # Read and drain inbox - reads all messages, then clears the file.
    # 
    # Draining is important because we don't want to re-process old messages.
    # Returns list of message dicts.
    # =============================================================================
    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        return messages

    # =============================================================================
    # Broadcast: send message to all teammates except sender.
    # Useful for team-wide announcements.
    # =============================================================================
    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# =============================================================================
# Task board scanning
# 
# Task board: .tasks/task_{id}.json files with format:
#   {"id": 1, "subject": "...", "description": "...", "status": "pending|in_progress|completed", "owner": "name", "blockedBy": []}
# 
# scan_unclaimed_tasks() finds tasks that:
#   - status = "pending"
#   - no owner assigned
#   - not blocked by other tasks
# 
# Auto-claiming enables teammates to find work without being assigned.
# =============================================================================
def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    return unclaimed


# =============================================================================
# claim_task(): Atomically claim a task by setting owner and status.
# 
# Uses _claim_lock to prevent race conditions when multiple teammates
# try to claim the same task simultaneously.
# =============================================================================
def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


# =============================================================================
# Identity re-injection after context compression
# 
# When context is compressed (s06), the LLM loses track of who it is.
# Solution: prepend an identity block that reminds the agent:
#   "You are 'coder', role: backend, team: my-team"
# 
# This block is inserted when:
#   1. Starting a new task after idle (messages <= 3 means fresh context)
#   2. After auto-claiming a task (need to remind who we are)
# 
# The <identity> tags help the model recognize this as system context.
# =============================================================================
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


# =============================================================================
# Autonomous TeammateManager
# 
# Key difference from s09/s10: autonomous teammates that find work themselves.
# 
# Lifecycle phases:
#   1. WORK PHASE: standard agent loop (LLM calls tools until stop_reason != tool_use)
#   2. IDLE PHASE: poll for inbox messages and unclaimed tasks every POLL_INTERVAL seconds
#      - If inbox has message -> resume WORK
#      - If unclaimed task exists -> claim it and resume WORK  
#      - If timeout (IDLE_TIMEOUT) -> shutdown
# 
# The "idle" tool signals the teammate has no more work and should enter IDLE PHASE.
# Auto-claiming enables autonomous work discovery without lead assignment.
# =============================================================================
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    # =============================================================================
    # _load_config(): Load team config from disk (persisted to .team/config.json)
    # Config includes team_name and members list with status.
    # =============================================================================
    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    # =============================================================================
    # _set_status(): Update member status and persist to disk.
    # 
    # Status values:
    #   - working: actively processing tasks
    #   - idle: polling for work (can be woken)
    #   - shutdown: permanently stopped (cannot respawn without explicit spawn)
    # =============================================================================
    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    # =============================================================================
    # spawn(): Start a new autonomous teammate thread.
    # 
    # If member already exists and status is idle/shutdown, respawn with new prompt.
    # Otherwise return error if already working.
    # =============================================================================
    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)  # Look up existing member in config
        
        if member:
            # Member exists - check if can be respawned
            if member["status"] not in ("idle", "shutdown"):
                # Already working - can't spawn again
                return f"Error: '{name}' is currently {member['status']}"
            # Respawn: update existing member with new role/prompt
            member["status"] = "working"  # Mark as working (thread will set to idle when done)
            member["role"] = role
        else:
            # New member - create config entry
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        
        # Persist config to disk (so status survives restart)
        self._save_config()
        
        # Create thread running _loop() - this is where the teammate works
        thread = threading.Thread(
            target=self._loop,           # Function to run in thread
            args=(name, role, prompt),   # Arguments passed to _loop
            daemon=True,                  # Thread dies when main program exits
        )
        self.threads[name] = thread      # Store reference (can check if alive later)
        thread.start()                   # Start the thread - _loop() begins executing
        return f"Spawned '{name}' (role: {role})"

    # =============================================================================
    # _loop(): Main teammate loop - alternates between WORK and IDLE phases.
    # 
    # WORK PHASE:
    #   - Check inbox for messages (including shutdown_request)
    #   - Run LLM with tools until stop_reason != tool_use or idle tool called
    #   - Handle tool results and append to messages
    # 
    # IDLE PHASE (entered when idle tool called or no work):
    #   - Poll inbox every POLL_INTERVAL seconds
    #   - Poll task board for unclaimed tasks
    #   - If inbox message or unclaimed task -> resume WORK
    #   - If IDLE_TIMEOUT exceeded -> shutdown
    # 
    # Identity re-injection: When starting new work after idle (messages <= 3),
    # prepend identity block to remind agent who it is.
    # =============================================================================
    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            # =============================================================================
            # WORK PHASE: Standard agent loop
            # 
            # Check inbox first (may contain messages or shutdown request)
            # Run LLM until it stops requesting tools or calls idle tool
            # =============================================================================
            # -- WORK PHASE: standard agent loop --
            for _ in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
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
                    self._set_status(name, "idle")
                    return
                messages.append({"role": "assistant", "content": response.content})
                if response.stop_reason != "tool_use":
                    break
                results = []
                idle_requested = False
                for block in response.content:
                    if block.type == "tool_use":
                        if block.name == "idle":
                            idle_requested = True
                            output = "Entering idle phase. Will poll for new tasks."
                        else:
                            output = self._exec(name, block.name, block.input)
                        print(f"  [{name}] {block.name}: {str(output)[:120]}")
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(output),
                        })
                messages.append({"role": "user", "content": results})
                if idle_requested:
                    break

            # =============================================================================
            # IDLE PHASE: Poll for work
            # 
            # Poll interval: POLL_INTERVAL seconds
            # Poll timeout: IDLE_TIMEOUT seconds (after which teammate shuts down)
            # 
            # Priority: inbox messages > unclaimed tasks
            # If either found, resume WORK phase
            # =============================================================================
            # -- IDLE PHASE: poll for inbox messages and unclaimed tasks --
            self._set_status(name, "idle")
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for _ in range(polls):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    claim_task(task["id"], name)
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    # Identity re-injection: if fresh context, prepend identity block
                    if len(messages) <= 3:
                        messages.insert(0, make_identity_block(name, role, team_name))
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    break

            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    # =============================================================================
    # _exec(): Tool execution handler for teammate's tools.
    # 
    # Base tools (unchanged from s02):
    #   - bash, read_file, write_file, edit_file: file operations
    #   - send_message, read_inbox: inter-agent communication
    # 
    # Protocol tools (from s10):
    #   - shutdown_response: respond to lead's shutdown request
    #   - plan_approval: submit plan for lead approval
    # 
    # Autonomous-specific tools:
    #   - idle: signal no more work, enter IDLE PHASE
    #   - claim_task: manually claim a task from the board
    # =============================================================================
    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # these base tools are unchanged from s02
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": args["approve"]},
            )
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
        return f"Unknown tool: {tool_name}"

    # =============================================================================
    # _teammate_tools(): Tools available to autonomous teammates.
    # 
    # Base tools (unchanged from s02):
    #   - bash: run shell commands
    #   - read_file, write_file, edit_file: file operations
    # 
    # Communication tools:
    #   - send_message: send message to teammate
    #   - read_inbox: read and drain inbox
    # 
    # Protocol tools (s10):
    #   - shutdown_response: respond to shutdown request
    #   - plan_approval: submit plan for approval
    # 
    # Autonomous-specific tools:
    #   - idle: signal no more work, enter IDLE PHASE
    #   - claim_task: manually claim a task (also auto-claimed in idle phase)
    # =============================================================================
    def _teammate_tools(self) -> list:
        # these base tools are unchanged from s02
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
            {"name": "shutdown_response", "description": "Respond to a shutdown request.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            {"name": "plan_approval", "description": "Submit a plan for lead approval.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
            {"name": "idle", "description": "Signal that you have no more work. Enters idle polling phase.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "claim_task", "description": "Claim a task from the task board by ID.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
        ]

    # =============================================================================
    # list_all(): Return formatted list of all team members and their statuses.
    # Useful for lead to see who's working/idle/shutdown.
    # =============================================================================
    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    # =============================================================================
    # member_names(): Return list of all teammate names.
    # Used for broadcast and other team-wide operations.
    # =============================================================================
    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


# =============================================================================
# Base tool implementations (unchanged from s02)
# 
# _safe_path(): Ensure path is within WORKDIR (security check)
# _run_bash(): Execute shell command with timeout and danger check
# _run_read(), _run_write(), _run_edit(): File operations with error handling
# =============================================================================
# -- Base tool implementations (these base tools are unchanged from s02) --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
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
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
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
# Lead-specific protocol handlers
# 
# handle_shutdown_request(): Send shutdown request to a teammate.
#   - Generates unique request_id
#   - Stores in shutdown_requests dict (IN-MEMORY, lost on restart)
#   - Sends message via MessageBus
# 
# handle_plan_review(): Respond to teammate's plan submission.
#   - Look up plan by request_id
#   - Update status to approved/rejected
#   - Send response via MessageBus
# 
# _check_shutdown_status(): Query status of a shutdown request.
# =============================================================================
# -- Lead-specific protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# =============================================================================
# Lead tool dispatch (14 tools)
# 
# Maps tool names to handler functions. Handlers use:
#   - kw["param"] for required parameters (will raise KeyError if missing)
#   - kw.get("param", default) for optional parameters
# 
# Tools include:
#   - Base: bash, read_file, write_file, edit_file
#   - Team: spawn_teammate, list_teammates, send_message, read_inbox, broadcast
#   - Protocol: shutdown_request, shutdown_response, plan_approval
#   - Autonomous: idle, claim_task
# =============================================================================
# -- Lead tool dispatch (14 tools) --
TOOL_HANDLERS = {
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":              lambda **kw: "Lead does not idle.",
    "claim_task":        lambda **kw: claim_task(kw["task_id"], "lead"),
}

# =============================================================================
# Lead tool definitions
# 
# input_schema defines what parameters each tool accepts.
# Required parameters must be listed in the "required" array.
# =============================================================================
# these base tools are unchanged from s02
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "spawn_teammate", "description": "Spawn an autonomous teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check shutdown request status.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state (for lead -- rarely used).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


# =============================================================================
# agent_loop(): Main lead agent loop.
# 
# Flow:
#   1. Check inbox for messages from teammates
#   2. Call LLM with current messages and tools
#   3. If stop_reason != tool_use, return (conversation done)
#   4. Otherwise execute tools and append results
#   5. Repeat from step 1
# 
# Unlike teammate loop, lead does NOT idle - it stays active.
# Inbox messages are processed every turn.
# =============================================================================
def agent_loop(messages: list):
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
            messages.append({
                "role": "assistant",
                "content": "Noted inbox messages.",
            })
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
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
        messages.append({"role": "user", "content": results})


# =============================================================================
# REPL for interactive testing
# 
# Commands:
#   /team    - List all teammates and their statuses
#   /inbox   - Show lead's inbox messages
#   /tasks   - List all tasks on the task board
#   q/exit   - Quit the REPL
# 
# Any other input is sent to the lead agent loop as a user message.
# 
# Task board format (.tasks/task_{id}.json):
#   {
#     "id": 1,
#     "subject": "Task title",
#     "description": "Task description",
#     "status": "pending|in_progress|completed",
#     "owner": "teammate_name",
#     "blockedBy": []
#   }
# =============================================================================
if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms11 >> \033[0m")
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
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
