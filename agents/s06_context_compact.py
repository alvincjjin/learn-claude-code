#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions.
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000  # Token count threshold to trigger auto_compact
TRANSCRIPT_DIR = WORKDIR / ".transcripts"  # Where to save full conversation logs
KEEP_RECENT = 3  # How many recent tool_results to keep (not compress)


def estimate_tokens(messages: list) -> int:
    """
    Rough token estimation.
    
    Uses character count / 4 as a rough approximation.
    Not exact but good enough for triggering compression.
    
    Note: This is a simplified estimation. Real tokenization varies
    by model and content type (code uses more tokens than text).
    """
    return len(str(messages)) // 4


# -- Layer 1: micro_compact - replace old tool results with placeholders --
# This runs on EVERY turn, silently compressing old tool outputs.
#
# Problem: Tool outputs can be huge (reading large files, bash output).
# Solution: Keep only last 3 results, replace older ones with short placeholders.
#
# Example:
#   Before:  [{"type": "tool_result", "content": "500 lines of file content..."}]
#   After:   [{"type": "tool_result", "content": "[Previous: used read_file]"}]
#
# Why > 100 chars? Only compress if there's actually savings.
# Short outputs like "Error: file not found" don't waste much context.
#
def micro_compact(messages: list) -> list:
    """
    Silently compress old tool results on every turn.
    
    Algorithm:
        1. Find all tool_result entries in message history
        2. Build a map from tool_use_id -> tool_name (by looking at prior assistant messages)
        3. Keep last KEEP_RECENT (3) tool results as-is
        4. Replace older tool_results (>100 chars) with "[Previous: used {tool_name}]"
    
    This is "micro" because it's lightweight and runs every turn.
    """
    # Step 1: Collect all tool_result entries from user messages
    # Each entry is (message_index, part_index, tool_result_dict)
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part_idx, part in enumerate(msg["content"]):
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tool_results.append((msg_idx, part_idx, part))
    
    # Step 2: Nothing to compress if we have KEEP_RECENT or fewer
    if len(tool_results) <= KEEP_RECENT:
        return messages
    
    # Step 3: Build a map from tool_use_id to tool_name
    # We need this because tool_result only has tool_use_id, not the tool name
    # Example: tool_result.tool_use_id="toolu_abc123" -> tool_name="read_file"
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        tool_name_map[block.id] = block.name
    
    # Step 4: Replace old tool results with placeholders
    # to_clear = all tool_results EXCEPT the last 3 (most recent)
    to_clear = tool_results[:-KEEP_RECENT]
    for _, _, result in to_clear:
        # Only compress if content is longer than 100 chars (worth compressing)
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_id = result.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
# This runs when token count exceeds THRESHOLD (50000 tokens).
#
# Problem: Even with micro_compact, context grows over time.
# Solution: Save full transcript to disk, ask LLM to summarize, replace all messages.
#
# Flow:
#   1. Save full conversation to .transcripts/transcript_<timestamp>.jsonl
#   2. Send conversation to LLM with summarize prompt
#   3. LLM returns summary with key info (accomplishments, current state, decisions)
#   4. Replace ALL messages with just 2: user message with summary + assistant acknowledgment
#
# Why save transcript? In case we need to reference old details later.
# The summary points to the transcript file for full details.
#
def auto_compact(messages: list) -> list:
    """
    Full conversation compression when context gets too large.
    
    Steps:
        1. Save full transcript to .transcripts/ directory
        2. Call LLM to summarize the conversation
        3. Replace all messages with compressed summary
    
    Returns:
        A new 2-message conversation: [user message with summary, assistant acknowledgment]
    """
    # Step 1: Save full transcript to disk
    # Each message becomes a JSON line for potential later reference
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    
    # Step 2: Ask LLM to summarize the conversation
    # We limit to 80k chars to avoid sending huge context to the summarizer
    conversation_text = json.dumps(messages, default=str)[:80000]
    response = client.messages.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )
    summary = response.content[0].text
    
    # Step 3: Replace all messages with just the summary
    # Only return the user message - Anthropic API requires ending with user message only
    # (no assistant prefill allowed)
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]


# -- Tool implementations --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
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
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
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
    "compact":    lambda **kw: "Manual compression requested.",
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
    {"name": "compact", "description": "Trigger manual conversation compression.",
     "input_schema": {"type": "object", "properties": {"focus": {"type": "string", "description": "What to preserve in the summary"}}}},
]


def agent_loop(messages: list):
    """
    Agent loop with three-layer compression pipeline.
    
    Three layers work together:
        - Layer 1 (micro_compact): Runs every turn, silently compresses old tool outputs
        - Layer 2 (auto_compact): Runs when tokens > 50000, summarizes conversation
        - Layer 3 (manual compact): Runs when model explicitly calls compact tool
    
    Compression flow per turn:
        1. micro_compact() - replace old tool results with placeholders
        2. Check token count - if > 50000, trigger auto_compact
        3. Call LLM with compressed messages
        4. Execute tools
        5. If model called compact tool, trigger manual compression
    """
    while True:
        # Layer 1: micro_compact runs before EVERY LLM call
        # This silently compresses old tool outputs (keeps last 3, replaces older)
        micro_compact(messages)
        
        # Layer 2: auto_compact triggers when context gets too large
        # If estimated tokens > THRESHOLD (50000), summarize and compress
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        
        # Call LLM with (now compressed) message history
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        
        # If LLM didn't call any tools, we're done
        if response.stop_reason != "tool_use":
            return
        
        # Execute tool calls
        results = []
        manual_compact = False  # Track if model called compact tool
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "compact":
                    # Layer 3: Manual compact - model explicitly requested compression
                    manual_compact = True
                    output = "Compressing..."
                else:
                    # Regular tool - execute it
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                    except Exception as e:
                        output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        
        # Append tool results as user message to continue loop
        messages.append({"role": "user", "content": results})
        
        # Layer 3: Manual compact triggered by model calling compact tool
        # This is the same as auto_compact but explicitly requested by the model
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
