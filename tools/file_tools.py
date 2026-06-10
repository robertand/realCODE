import os
import glob as glob_module
import re


class ReadTool:
    name = "read"
    description = "Read the contents of a file or list a directory."

    schema = {
        "description": "Read a file or list a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file or directory (absolute or relative to current working directory)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed), default 1",
                    "default": None
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to read, default 2000",
                    "default": None
                }
            },
            "required": ["path"]
        }
    }

    def execute(self, path: str, offset: int = None, limit: int = None) -> str:
        # Convert to absolute path if relative
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        
        if not os.path.exists(path):
            return f"[error: path does not exist: {path}]"

        if os.path.isdir(path):
            entries = sorted(os.listdir(path))
            lines = []
            for e in entries:
                full = os.path.join(path, e)
                suffix = "/" if os.path.isdir(full) else "*" if os.access(full, os.X_OK) and not os.path.isdir(full) else ""
                lines.append(f"{e}{suffix}")
            return "\n".join(lines) if lines else "[empty directory]"

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.readlines()
        except Exception as e:
            return f"[error: cannot read file: {e}]"

        total = len(content)
        start = (offset - 1) if offset and offset > 0 else 0
        end = start + (limit or 2000)
        selected = content[start:end]

        result = []
        for i, line in enumerate(selected, start=start + 1):
            result.append(f"{i:6d}| {line.rstrip()}")

        footer = []
        if start > 0:
            footer.append(f"[showing lines {start+1}-{min(end, total)} of {total}]")
        else:
            footer.append(f"[showing {min(len(selected), total)} of {total} lines]")
        if end < total:
            footer.append(f"[use offset={end+1} to read more]")

        return "\n".join(result) + ("\n" + "\n".join(footer) if footer else "")


class WriteTool:
    name = "write"
    description = "Write content to a file (overwrites existing content)."

    schema = {
        "description": "Write content to a file (creates or overwrites)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative to current working directory)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    }

    def execute(self, path: str, content: str) -> str:
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[ok: wrote {len(content)} bytes to {path}]"
        except Exception as e:
            return f"[error: {e}]"


class EditTool:
    name = "edit"
    description = "Perform an exact string replacement in a file."

    schema = {
        "description": "Replace oldString with newString in a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative)"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_string": {
                    "type": "string",
                    "description": "The new text to insert"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default: false)",
                    "default": False
                }
            },
            "required": ["path", "old_string", "new_string"]
        }
    }

    def execute(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
            
        if not os.path.isfile(path):
            return f"[error: file does not exist: {path}]"

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"[error: cannot read: {e}]"

        if old_string not in content:
            return f"[error: old_string not found in file]"

        count = content.count(old_string)
        if not replace_all and count > 1:
            return f"[error: found {count} matches; use replace_all=true or provide more context]"

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"[ok: replaced {count if replace_all else 1} occurrence(s) in {path}]"
        except Exception as e:
            return f"[error: {e}]"


class GlobTool:
    name = "glob"
    description = "Find files by glob pattern (e.g. **/*.py, src/**/*.ts)."

    schema = {
        "description": "Find files matching a glob pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match"
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search from (default: current directory)",
                    "default": None
                }
            },
            "required": ["pattern"]
        }
    }

    def execute(self, pattern: str, path: str = None) -> str:
        search_root = path or os.getcwd()
        if not os.path.isabs(search_root):
            search_root = os.path.join(os.getcwd(), search_root)
            
        full_pattern = os.path.join(search_root, pattern)
        matches = sorted(glob_module.glob(full_pattern, recursive=True))
        if not matches:
            return f"[no files matched: {pattern}]"
        # Limit to 1000 results
        if len(matches) > 1000:
            matches = matches[:1000]
            matches.append(f"... and {len(matches) - 1000} more")
        return "\n".join(matches)


class GrepTool:
    name = "grep"
    description = "Search file contents using regular expressions."

    schema = {
        "description": "Search files for a regex pattern",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "include": {
                    "type": "string",
                    "description": "File glob to filter (e.g. *.py, *.{ts,tsx})",
                    "default": None
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                    "default": None
                }
            },
            "required": ["pattern"]
        }
    }

    def execute(self, pattern: str, include: str = None, path: str = None) -> str:
        search_root = path or os.getcwd()
        if not os.path.isabs(search_root):
            search_root = os.path.join(os.getcwd(), search_root)
            
        matches = []
        try:
            import subprocess
            cmd = ["rg", "-n", "--hidden", pattern, search_root]
            if include:
                cmd.extend(["-g", include])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout
            elif result.returncode == 1:
                return f"[no matches for: {pattern}]"
            else:
                return f"[rg error: {result.stderr}]"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback to Python os.walk
        for root, dirs, files in os.walk(search_root):
            if ".git" in dirs:
                dirs.remove(".git")
            for file in files:
                if include:
                    import fnmatch
                    if not fnmatch.fnmatch(file, include):
                        continue
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if re.search(pattern, line):
                                relpath = os.path.relpath(fpath, search_root)
                                matches.append(f"{relpath}:{i}: {line.rstrip()[:200]}")
                                if len(matches) >= 500:
                                    matches.append("... (truncated, too many matches)")
                                    return "\n".join(matches)
                except Exception:
                    pass

        if not matches:
            return f"[no matches for: {pattern}]"
        return "\n".join(matches[:500])