"""Filesystem skills: read_file, write_file, edit_file, delete_path, move_path.

Grouped in one module per the design spec (fewer directories than the
one-class-per-folder pattern used by the older built-ins) — same registry
surface either way.

Every skill resolves its path argument with `_resolve_path` (expand `~`,
absolute) and reports that resolved path back, so a confirmation prompt shows
the user exactly what will happen. `.absolute()` is used rather than
`.resolve()` deliberately: it does not dereference symlinks, so the path
shown is the one the user actually typed, not wherever a symlink points.
"""

import shutil
from pathlib import Path

from nero.security import envelope
from nero.skills.base import Skill, SkillMeta

DEFAULT_MAX_READ_BYTES = 100_000


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().absolute()


class ReadFileSkill(Skill):
    meta = SkillMeta(
        name="read_file",
        description=(
            "Read a text file from the user's filesystem. Use this when the user "
            "asks you to read, open, show, or look at a file. Refuses binary files "
            "and files larger than max_bytes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
                "max_bytes": {
                    "type": "integer",
                    "description": f"Maximum bytes to read (default {DEFAULT_MAX_READ_BYTES}).",
                },
            },
            "required": ["path"],
        },
        requires_network=False,
        permission_tier="read_only",
        ingests_external_content=True,
    )

    async def execute(self, **kwargs) -> str:
        path = _resolve_path(str(kwargs.get("path") or ""))
        max_bytes = int(kwargs.get("max_bytes") or DEFAULT_MAX_READ_BYTES)
        if not path.is_file():
            return f"Error: {path} is not a file."
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return f"Error reading {path}: {exc}"
        if b"\x00" in raw[:8192]:
            return f"Error: {path} looks like a binary file; refusing to read it as text."
        if len(raw) > max_bytes:
            return (
                f"Error: {path} is {len(raw)} bytes, over the {max_bytes}-byte limit. "
                "Ask for a higher max_bytes if you really need the whole file."
            )
        text = raw.decode("utf-8", errors="replace")
        return envelope(f"file:{path}", text)


class WriteFileSkill(Skill):
    meta = SkillMeta(
        name="write_file",
        description=(
            "Write text content to a file, creating it (and any parent directories) "
            "if needed, or overwriting it if it already exists. Use this when the "
            "user asks you to create, save, or write a file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            "required": ["path", "content"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    async def execute(self, **kwargs) -> str:
        path = _resolve_path(str(kwargs.get("path") or ""))
        content = str(kwargs.get("content") or "")
        existed = path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"Error writing {path}: {exc}"
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {path} ({len(content.encode('utf-8'))} bytes)."


class EditFileSkill(Skill):
    meta = SkillMeta(
        name="edit_file",
        description=(
            "Replace an exact, unique snippet of text in a file with new text. "
            "old_string must match exactly once in the file — if it's missing or "
            "appears more than once, the edit is refused. Use this for targeted "
            "changes to an existing file rather than rewriting the whole thing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_string", "new_string"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    async def execute(self, **kwargs) -> str:
        path = _resolve_path(str(kwargs.get("path") or ""))
        old_string = str(kwargs.get("old_string") or "")
        new_string = str(kwargs.get("new_string") or "")
        if not path.is_file():
            return f"Error: {path} is not a file."
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"Error reading {path}: {exc}"
        count = text.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}."
        if count > 1:
            return f"Error: old_string appears {count} times in {path}; it must be unique."
        try:
            path.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        except OSError as exc:
            return f"Error writing {path}: {exc}"
        return f"Edited {path}."


class DeletePathSkill(Skill):
    meta = SkillMeta(
        name="delete_path",
        description=(
            "Delete a file or directory. Deleting a non-empty directory requires "
            "recursive=true. Use this when the user explicitly asks to delete or "
            "remove a file or folder."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to delete."},
                "recursive": {
                    "type": "boolean",
                    "description": "Required to delete a non-empty directory (default false).",
                },
            },
            "required": ["path"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    async def execute(self, **kwargs) -> str:
        path = _resolve_path(str(kwargs.get("path") or ""))
        recursive = bool(kwargs.get("recursive") or False)
        if path.is_symlink():
            # Delete the link itself — never follow it into whatever it points at.
            path.unlink()
            return f"Deleted symlink {path}."
        if not path.exists():
            return f"Error: {path} does not exist."
        try:
            if path.is_dir():
                if any(path.iterdir()) and not recursive:
                    return (
                        f"Error: {path} is a non-empty directory. "
                        "Pass recursive=true to delete it and its contents."
                    )
                shutil.rmtree(path)
                return f"Deleted directory {path} and its contents."
            path.unlink()
            return f"Deleted {path}."
        except OSError as exc:
            return f"Error deleting {path}: {exc}"


class MovePathSkill(Skill):
    meta = SkillMeta(
        name="move_path",
        description=(
            "Move or rename a file or directory. Refuses if the destination already "
            "exists, to avoid silently overwriting something. Use this when the user "
            "asks to move or rename a file or folder."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to move."},
                "destination": {"type": "string", "description": "Where to move it to."},
            },
            "required": ["source", "destination"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    async def execute(self, **kwargs) -> str:
        source = _resolve_path(str(kwargs.get("source") or ""))
        destination = _resolve_path(str(kwargs.get("destination") or ""))
        if not source.exists() and not source.is_symlink():
            return f"Error: {source} does not exist."
        if destination.exists() or destination.is_symlink():
            return f"Error: {destination} already exists; refusing to overwrite it."
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        except OSError as exc:
            return f"Error moving {source} to {destination}: {exc}"
        return f"Moved {source} to {destination}."


READ_FILE = ReadFileSkill()
WRITE_FILE = WriteFileSkill()
EDIT_FILE = EditFileSkill()
DELETE_PATH = DeletePathSkill()
MOVE_PATH = MovePathSkill()
