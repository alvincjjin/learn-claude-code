#!/usr/bin/env python3
# Harness: tool dispatch -- expanding what the model can reach.
# This version adds file operations (read, write, edit) to the bash tool.
# The key insight: the agent loop itself doesn't change - only the tools definition.
"""
s02_tool_use.py - Tools

The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.

    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+

Key insight: "The loop didn't change at all. I just added tools."
"""

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# If using custom base URL, remove auth token to allow proxy connections
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# Use pathlib for safer path handling (especially important for file operations)
WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# Updated system prompt to reflect available tools
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


def safe_path(p: str) -> Path:
    """
    Resolve a relative path and ensure it stays within WORKDIR.
    
    This is a critical security function - it prevents path traversal attacks
    where the model might try to access files outside the workspace (e.g., ../../../etc/passwd).
    """
    # Join with WORKDIR and resolve to absolute path
    path = (WORKDIR / p).resolve()
    # Verify the path is still within WORKDIR (security check)
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """Execute a shell command with safety checks and timeout."""
    # Block dangerous commands that could harm the system
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # Run command in WORKDIR, capture both stdout and stderr
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: int = None) -> str:
    """
    Read file contents with optional line limit.
    
    Args:
        path: Relative path to file (validated by safe_path)
        limit: Optional max number of lines to return
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        # Apply limit if specified and file has more lines
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    Write content to a file, creating parent directories if needed.
    
    Args:
        path: Relative path to file (validated by safe_path)
        content: The text content to write
    """
    try:
        fp = safe_path(path)
        # Create parent directories (e.g., for new files in subdirectories)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in a file (first occurrence only).
    
    This is a simple but effective edit function - it finds the first occurrence
    of old_text and replaces it with new_text.
    
    Args:
        path: Relative path to file
        old_text: Exact string to find and replace
        new_text: Replacement string
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        # Verify the text exists before editing
        if old_text not in content:
            return f"Error: Text not found in {path}"
        # Replace only the first occurrence (replaceOldText, newText, 1)
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- The dispatch map: {tool_name: handler} --
# This maps tool names (from the LLM) to handler functions.
# Using **kwargs allows flexible argument passing from the tool_use block.
#
# How block.input flows to handler functions:
#
# block.input is a dict of arguments matching the tool's input_schema.
# Example for read_file tool:
#   block = {
#       "type": "tool_use",
#       "id": "toolu_abc123",
#       "name": "read_file",
#       "input": {
#           "path": "agents/s01_agent_loop.py",   # from input_schema properties
#           "limit": 50                            # optional, from input_schema
#       }
#   }
#
# We unpack block.input with ** into the handler:
#   handler(**block.input)  ->  run_read(path="agents/s01_agent_loop.py", limit=50)
#
# The lambda uses **kw to accept any named arguments, then extracts them:
#   lambda **kw: run_read(kw["path"], kw.get("limit"))
#                                  ^^^^^^^         ^^^^^^^^^^
#                                  required param   optional param
#
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    # Use kw.get() for optional params (not in "required" array) - returns None if missing
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

# Tool definitions for the LLM
# Each tool has a name, description, and JSON schema for its input parameters.
# The LLM uses these schemas to understand what each tool does and how to call it.
#
# input_schema is a JSON Schema that tells the LLM:
#   - What parameters the tool accepts (properties)
#   - Which parameters are required (required array)
#   - What type each parameter is (string, integer, boolean, etc.)
#
# Example - read_file tool:
#   "input_schema": {
#       "type": "object",                           # JSON object
#       "properties": {
#           "path": {"type": "string"},             # parameter "path" is a string
#           "limit": {"type": "integer"}            # parameter "limit" is optional (not in "required")
#       },
#       "required": ["path"]                         # "path" MUST be provided
#   }
#
# The LLM will generate block.input like:
#   {"path": "agents/s01_agent_loop.py", "limit": 50}
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]


def agent_loop(messages: list):
    """
    The agent loop - identical to s01, but now dispatches to multiple tools.
    
    The key change: instead of just run_bash, we now look up the handler
    in TOOL_HANDLERS based on the tool name from the LLM.
    """
    while True:
        # Call LLM with the expanded tool list
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        
        # Check if model wants to use a tool or is done
        if response.stop_reason != "tool_use":
            return
        
        # Execute each tool call using the dispatch map
        # response.content is a list of blocks parsed from the LLM response.
        # Each tool_use block has attributes that the LLM generated:
        #   - block.name: tool name (e.g., "read_file") - parsed from what the LLM decided to call
        #   - block.input: dict of arguments (e.g., {"path": "...", "limit": 50}) - parsed from LLM's input
        #   - block.id: unique ID (e.g., "toolu_abc123") - for matching with tool_result
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # Look up the handler function by tool name
                handler = TOOL_HANDLERS.get(block.name)
                
                # Call the handler with the arguments from the LLM
                # handler(**block.input) unpacks the dict into keyword arguments
                # e.g., run_read(path="...", limit=50)
                # If handler not found, return an error message
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"> {block.name}: {output[:200]}")
                
                # Build tool_result message for the feedback loop
                # tool_use_id must match block.id so the LLM knows which result goes with which call
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        
        # Append tool results to continue the loop
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    Interactive REPL - identical to s01, but now has access to file tools.
    
    The model can now read, write, and edit files in addition to running bash commands.
    """
    history = []  # Conversation history
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        
        # Add user message and run agent loop
        history.append({"role": "user", "content": query})
        agent_loop(history)
        
        # Display final response from the model
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
