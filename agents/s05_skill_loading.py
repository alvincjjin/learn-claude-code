#!/usr/bin/env python3
# Harness: on-demand knowledge -- domain expertise, loaded when the model asks.
"""
s05_skill_loading.py - Skills

Two-layer skill injection that avoids bloating the system prompt:

    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)
    Layer 2 (on demand): full skill body in tool_result

    skills/
      pdf/
        SKILL.md          <-- frontmatter (name, description) + body
      code-review/
        SKILL.md

    System prompt:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - pdf: Process PDF files...        |  <-- Layer 1: metadata only
    |   - code-review: Review code...      |
    +--------------------------------------+

    When model calls load_skill("pdf"):
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   Full PDF processing instructions   |  <-- Layer 2: full body
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                             |
    +--------------------------------------+

Key insight: "Don't put everything in the system prompt. Load on demand."
"""

import os
import re
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
SKILLS_DIR = WORKDIR / "skills"


# -- SkillLoader: scan skills/<name>/SKILL.md with YAML frontmatter --
# This class loads skills from the filesystem on startup.
#
# Expected directory structure:
#   skills/
#     pdf/
#       SKILL.md         <- skill name comes from folder name "pdf"
#     code-review/
#       SKILL.md         <- skill name = "code-review"
#
# SKILL.md format (YAML frontmatter + body):
#   ---
#   name: pdf
#   description: Process PDF files with OCR
#   tags: pdf, ocr, document
#   ---
#   Full skill instructions here...
#   Step 1: ...
#   Step 2: ...
#
# Two-layer approach:
#   - Layer 1 (get_descriptions): Only metadata for system prompt
#   - Layer 2 (get_content): Full body loaded on demand via tool
#
class SkillLoader:
    def __init__(self, skills_dir: Path):
        """
        Initialize and load all skills from the skills directory.
        
        Args:
            skills_dir: Path to the skills/ directory containing skill folders
        """
        self.skills_dir = skills_dir
        self.skills = {}  # {"name": {"meta": {...}, "body": "...", "path": "..."}}
        self._load_all()

    def _load_all(self):
        """
        Recursively scan skills_dir for all SKILL.md files.
        
        Uses rglob to find SKILL.md in any subdirectory.
        Each file's parent folder becomes the skill name.
        """
        if not self.skills_dir.exists():
            return
        # rglob finds SKILL.md in skills/ and all subdirectories
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            # Use folder name as skill name (e.g., skills/pdf/SKILL.md -> "pdf")
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        """
        Parse YAML frontmatter from skill file.
        
        Expected format:
            ---
            name: pdf
            description: Process PDF files
            tags: pdf, ocr
            ---
            Full skill body here...
        
        Returns:
            (meta dict, body string)
        """
        # Match text between first --- and second ---
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text  # No frontmatter, return full text as body
        meta = {}
        # Parse each line with "key: value" format
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """
        Layer 1: Get short descriptions for system prompt.
        
        Only returns metadata (name, description, tags) - not the full body.
        This keeps the system prompt small (~100 tokens/skill).
        
        Example output:
            - pdf: Process PDF files [pdf, ocr]
            - code-review: Review code for issues [lint, security]
        """
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """
        Layer 2: Get full skill body returned in tool_result.
        
        Called when model calls load_skill(name) tool.
        Returns wrapped in <skill> tags for easy parsing.
        
        Example output:
            <skill name="pdf">
            Full PDF processing instructions...
            Step 1: ...
            Step 2: ...
            </skill>
        """
        skill = self.skills.get(name)
        if not skill:
            available = ", ".join(self.skills.keys())
            return f"Error: Unknown skill '{name}'. Available: {available}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(SKILLS_DIR)

# Layer 1: skill metadata injected into system prompt
# This tells the LLM what skills exist WITHOUT loading the full content.
# Only metadata (~100 tokens per skill) goes in system prompt.
# Full skill body is loaded on-demand via load_skill tool.
#
# System prompt structure:
#   You are a coding agent at /path/to/workdir.
#   Use load_skill to access specialized knowledge before tackling unfamiliar topics.
#
#   Skills available:
#     - pdf: Process PDF files [pdf, ocr]
#     - code-review: Review code for issues [lint, security]
#
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


# -- Tool implementations --
# Same as s02-s04, but adds load_skill for on-demand skill loading

def safe_path(p: str) -> Path:
    """Resolve relative path and ensure it stays within WORKDIR (security check)."""
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


# -- Tool dispatch map --
# Maps tool names (from LLM) to handler functions.
# Added load_skill which calls SKILL_LOADER.get_content(name).
#
# How it works:
#   1. LLM calls load_skill tool
#   2. block.input = {"name": "pdf"}  <- skill name from LLM
#   3. handler calls SKILL_LOADER.get_content("pdf")
#   4. Returns full skill body wrapped in <skill> tags
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # load_skill: Takes skill name, returns full skill body from SKILL_LOADER
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}

# Tool definitions for the LLM
# Each tool has name, description, and input_schema (JSON Schema).
# The load_skill tool takes a "name" parameter (skill name to load).
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}},
]


def agent_loop(messages: list):
    """
    Agent loop with skill loading capability.
    
    Key difference from s02-s04:
    - Added load_skill tool to TOOLS
    - When LLM calls load_skill, returns full skill body in tool_result
    
    How skill loading works:
        1. LLM decides it needs specialized knowledge
        2. Calls load_skill with name parameter (e.g., "pdf")
        3. SKILL_LOADER.get_content(name) loads skill body from file
        4. Returns wrapped in <skill> tags as tool_result
        5. LLM can now read and use the full skill instructions
    
    Two-layer approach:
        - Layer 1 (system prompt): Only skill names + descriptions (~100 tokens)
        - Layer 2 (tool_result): Full skill body loaded on demand (can be thousands of tokens)
    
    HOW THE LLM DECIDES TO LOAD A SKILL:
    
    The LLM uses context from the user's request + system prompt to decide:
        - System prompt says: "Use load_skill to access specialized knowledge"
        - System prompt lists available skills with descriptions
        
    Example scenario:
        User: "Extract text from invoice.pdf using OCR"
              │
              ▼
        LLM thinks: "The user wants OCR/PDF processing. Do I have this knowledge?"
              │
              ▼
        Checks system prompt:
            "Skills available:
              - pdf: Process PDF files [pdf, ocr]
              - code-review: Review code for issues"
              │
              ▼
        LLM reasons: "I see 'pdf' skill has 'ocr' tag - I should load it!"
              │
              ▼
        Calls load_skill tool with: {"name": "pdf"}
              │
              ▼
        Tool result returns full skill body:
            <skill name="pdf">
            Step 1: Use pypdf to extract text...
            Step 2: If image, use pytesseract for OCR...
            </skill>
              │
              ▼
        LLM reads skill and executes: runs bash/read/write tools
    
    The decision is driven by:
        1. The task at hand (user's request)
        2. Skill descriptions in system prompt
        3. Tags which help the LLM match skills to tasks
    """
    while True:
        # Call LLM with available tools (including load_skill)
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        
        # If LLM didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return
        
        # Execute tool calls
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # Look up handler (bash, read_file, write_file, edit_file, or load_skill)
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}: {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        
        # Append tool results as user message to continue loop
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
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
