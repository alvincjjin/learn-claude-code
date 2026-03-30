#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
# This is a minimal implementation showing the core agent loop pattern.
# The loop: model thinks -> calls tools -> executes tools -> feeds results back -> repeat
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess

from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file, override existing ones
load_dotenv(override=True)

# If using a custom base URL (e.g., for local models), remove the auth token
# This allows connecting to proxy servers or local endpoints
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# Create Anthropic client with optional custom base URL
# Falls back to default Anthropic API if ANTHROPIC_BASE_URL is not set
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# System prompt sets the agent's identity and behavior
# Tells the model it's a coding agent in the current working directory
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# Define the tools available to the model
# This is a JSON schema describing the tool's name, description, and input parameters
# The model decides when to call these tools based on the conversation context
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def run_bash(command: str) -> str:
    """
    Execute a shell command and return its output.
    
    Safety: Blocks dangerous commands that could harm the system.
    Timeout: Commands are limited to 120 seconds to prevent hanging.
    Output: Combined stdout and stderr, truncated to 50k chars.
    """
    # Block potentially destructive commands for safety
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # Run command in current working directory with output capture
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        # Combine stdout and stderr, truncate if too long
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    """
    The main agent loop - handles the conversation between user and model.
    
    Flow:
    1. Send current message history to the LLM with available tools
    2. If model responds with tool use: execute tools and loop back with results
    3. If model responds without tool use: return (conversation complete)
    
    The key insight: tool_results are fed back as user messages, 
    creating a feedback loop that lets the model continue reasoning.

    # Block Structure (response.content is a list of blocks):
    # 
    # 1. Text block:
    #    {"type": "text", "text": "Hello world"}
    #
    # 2. Tool Use block (model calling a tool):
    #    {
    #        "type": "tool_use",
    #        "id": "toolu_abc123",        # unique ID for this tool call
    #        "name": "bash",               # tool name (must match one in TOOLS)
    #        "input": {"command": "ls"}    # arguments from tool's input_schema
    #    }
    #
    # 3. Tool Result (what we send back in the next user message):
    #    {
    #        "type": "tool_result",
    #        "tool_use_id": "toolu_abc123", # matches block.id above
    #        "content": "file1.py\nfile2.py"  # tool's output
    #    }
    #
    # The loop: assistant calls tools -> we execute and return tool_results -> repeat
    """
    while True:
        # Step 1: Call the LLM with conversation history and available tools
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        
        # Step 2: Add model's response to message history
        messages.append({"role": "assistant", "content": response.content})
        
        # Step 3: Check if model wants to use a tool or is done
        # stop_reason="tool_use" means the model called a tool
        # Any other stop_reason (e.g., "end_turn", "max_tokens") means we're done
        if response.stop_reason != "tool_use":
            return
        
        # Step 4: Execute each tool the model called
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # Print command in yellow for visibility
                print(f"\033[33m$ {block.input['command']}\033[0m")
                # Run the command and capture output
                output = run_bash(block.input["command"])
                # Print first 200 chars of output
                print(output[:200])
                # Build tool_result message with the tool's output
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        
        # Step 5: Append tool results as a user message to continue the loop
        # This is the key to the feedback loop - model sees its tool results
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """
    Interactive REPL for testing the agent loop.
    
    - Maintains conversation history across multiple turns
    - Exits on 'q', 'exit', or empty input
    - Displays the final model response after each agent completion
    """
    history = []  # Stores the full conversation: user msgs, assistant msgs, tool results
    
    while True:
        try:
            # Prompt user for input (cyan color for visibility)
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        
        # Handle exit commands
        if query.strip().lower() in ("q", "exit", ""):
            break
        
        # Add user message to history and run the agent loop
        history.append({"role": "user", "content": query})
        agent_loop(history)
        
        # After agent loop completes, display the final response
        # The response could be a list of content blocks (text, tool_use, etc.)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
