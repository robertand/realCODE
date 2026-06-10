import ast
import json
import os
import re
import subprocess
import time
from typing import List, Dict
from collections import Counter


class CodeAnalyzerTool:
    name = "analyze_code"
    description = "Analyze Python code for errors, complexity, and suggestions"

    schema = {
        "description": "Analyze Python code statically",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to analyze"},
                "file_path": {"type": "string", "description": "Or path to file"}
            }
        }
    }

    def execute(self, code: str = None, file_path: str = None) -> str:
        if file_path:
            with open(file_path, 'r') as f:
                code = f.read()

        if not code:
            return "[error: no code provided]"

        issues = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if len(node.body) > 50:
                        issues.append(f"Function '{node.name}' has {len(node.body)} lines (consider splitting)")
                elif isinstance(node, (ast.For, ast.While, ast.If)):
                    depth = self._get_nesting_depth(node, tree)
                    if depth > 3:
                        issues.append(f"Deep nesting detected (depth {depth}) at line {node.lineno}")

            imports = set()
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module.split('.')[0] if node.module else '')
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)

            unused = imports - used_names
            if unused:
                issues.append(f"Unused imports: {', '.join(unused)}")

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")

        return "\n".join(issues) if issues else "No major issues found"

    def _get_nesting_depth(self, node, tree, current_depth=0):
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While, ast.If, ast.With)):
                depth = self._get_nesting_depth(child, tree, current_depth + 1)
                max_depth = max(max_depth, depth)
        return max_depth


class MemoryTool:
    name = "memory"
    description = "Store and retrieve information between sessions"

    def __init__(self):
        self.memory_file = "agent_memory.json"
        self.memory = self._load_memory()

    def _load_memory(self):
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"facts": [], "preferences": {}, "code_patterns": []}

    def _save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def execute(self, action: str, key: str = None, value: str = None) -> str:
        if action == "remember":
            if key and value:
                self.memory["facts"].append({"key": key, "value": value, "timestamp": time.time()})
                self._save_memory()
                return f"Remembered: {key} = {value}"
        elif action == "recall":
            if key:
                matches = [f for f in self.memory["facts"] if key.lower() in f["key"].lower()]
                if matches:
                    return "\n".join([f"{m['key']}: {m['value']}" for m in matches[-5:]])
                return "No memory found"
        elif action == "forget":
            if key:
                self.memory["facts"] = [f for f in self.memory["facts"] if key.lower() not in f["key"].lower()]
                self._save_memory()
                return f"Forgotten: {key}"

        return "Usage: memory(remember/recall/forget, key, value)"


class PlanTool:
    name = "plan"
    description = "Create and track multi-step plans"

    def __init__(self):
        self.plans = {}

    def execute(self, action: str, plan_name: str = None, steps: List[str] = None,
                step_index: int = None) -> str:

        if action == "create":
            self.plans[plan_name] = {
                "steps": steps,
                "completed": [],
                "current": 0
            }
            result = f"Plan '{plan_name}' created with {len(steps)} steps:\n"
            result += "\n".join([f"  {i+1}. {s}" for i, s in enumerate(steps)])
            return result

        elif action == "next" and plan_name:
            plan = self.plans.get(plan_name)
            if plan and plan["current"] < len(plan["steps"]):
                next_step = plan["steps"][plan["current"]]
                return f"Next step: {next_step}"
            return "Plan completed!"

        elif action == "complete" and plan_name:
            plan = self.plans.get(plan_name)
            if plan:
                step = plan["steps"][plan["current"]]
                plan["completed"].append(step)
                plan["current"] += 1
                return f"Completed: {step}"

        return "Usage: plan(create/list/next/complete, plan_name, steps, step_index)"


class ReasoningTool:
    name = "reason"
    description = "Break down complex problems step by step"

    schema = {
        "description": "Think through a problem systematically",
        "parameters": {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The problem to analyze"},
                "depth": {"type": "integer", "description": "Analysis depth (1-5)", "default": 3}
            }
        }
    }

    def execute(self, problem: str, depth: int = 3) -> str:
        reasoning = []

        reasoning.append(f"Analyzing: {problem}\n")

        for i in range(depth):
            reasoning.append(f"\nStep {i+1}:")

            if i == 0:
                reasoning.append("  Break down the problem into sub-problems")
                sub_problems = self._identify_subproblems(problem)
                for sp in sub_problems:
                    reasoning.append(f"    - {sp}")

            elif i == 1:
                reasoning.append("  Identify dependencies and constraints")
                reasoning.append("    - Check for required inputs")
                reasoning.append("    - Verify preconditions")

            elif i == 2:
                reasoning.append("  Consider alternative approaches")
                reasoning.append("    - Direct implementation")
                reasoning.append("    - Use existing libraries")
                reasoning.append("    - Optimize for performance")

            elif depth > 3 and i == 3:
                reasoning.append("  Evaluate potential issues")
                reasoning.append("    - Edge cases")
                reasoning.append("    - Error handling")
                reasoning.append("    - Performance bottlenecks")

        return "\n".join(reasoning)

    def _identify_subproblems(self, problem: str) -> List[str]:
        if "file" in problem.lower():
            return ["Read/parse input", "Process data", "Generate output"]
        elif "api" in problem.lower():
            return ["Make HTTP request", "Parse response", "Handle errors"]
        else:
            return ["Understand requirements", "Design solution", "Implement", "Test"]


class ProjectAnalyzerTool:
    name = "project_analyze"
    description = "Analyze project structure, detect framework, entry points, dependencies"

    schema = {
        "description": "Deep analysis of the current project",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Project root directory", "default": "."}
            }
        }
    }

    def execute(self, path: str = ".") -> str:
        root = os.path.abspath(path)
        if not os.path.isdir(root):
            return f"[error: not a directory: {root}]"

        parts = []
        parts.append(f"Project: {root}\n")

        # Detect project type
        project_type = self._detect_project_type(root)
        parts.append(f"Type: {project_type}")

        # Count files by extension
        files_by_ext = Counter()
        all_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
            for f in filenames:
                if f.startswith('.'):
                    continue
                ext = os.path.splitext(f)[1] or '(no ext)'
                files_by_ext[ext] += 1
                all_files.append(os.path.relpath(os.path.join(dirpath, f), root))

        parts.append(f"\nFiles: {len(all_files)} total")
        parts.append("Extensions:")
        for ext, count in files_by_ext.most_common(10):
            parts.append(f"  {ext or '(no ext)'}: {count}")

        # Entry points
        entry_points = self._find_entry_points(root, project_type)
        if entry_points:
            parts.append("\nEntry points:")
            for ep in entry_points:
                parts.append(f"  {ep}")

        # Dependencies
        deps = self._find_dependencies(root)
        if deps:
            parts.append("\nDependencies:")
            for dep in deps[:20]:
                parts.append(f"  {dep}")

        # Key definitions (classes, functions in main files)
        key_defs = self._find_key_definitions(root)
        if key_defs:
            parts.append(f"\nKey definitions:")
            for name, fpath, kind in key_defs[:15]:
                parts.append(f"  {kind} {name} -> {fpath}")

        parts.append(f"\nDirectories:")
        dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith('.') and d != '__pycache__'])
        for d in dirs:
            contents = os.listdir(os.path.join(root, d))
            parts.append(f"  {d}/ ({len(contents)} items)")

        return "\n".join(parts)

    def _detect_project_type(self, root: str) -> str:
        indicators = {
            "Python package": ["setup.py", "setup.cfg", "pyproject.toml"],
            "Django": ["manage.py", "settings.py"],
            "FastAPI": ["main.py"],
            "Flask": ["app.py", "wsgi.py"],
            "Node.js": ["package.json", "node_modules"],
            "React": ["vite.config.js", "next.config.js", "src/App.jsx"],
            "Rust": ["Cargo.toml"],
            "Go": ["go.mod"],
            "Docker": ["Dockerfile", "docker-compose.yml"],
        }
        detected = []
        for ptype, markers in indicators.items():
            for marker in markers:
                marker_path = os.path.join(root, marker)
                if os.path.exists(marker_path):
                    detected.append(ptype)
                    break
        return ", ".join(detected) if detected else "Unknown"

    def _find_entry_points(self, root: str, project_type: str) -> List[str]:
        candidates = ["main.py", "app.py", "cli.py", "manage.py", "run.py", "index.py"]
        found = []
        for c in candidates:
            p = os.path.join(root, c)
            if os.path.isfile(p):
                found.append(f"{c}")
        if os.path.isdir(os.path.join(root, "src")):
            for c in candidates:
                p = os.path.join(root, "src", c)
                if os.path.isfile(p):
                    found.append(f"src/{c}")
        return found

    def _find_dependencies(self, root: str) -> List[str]:
        dep_files = ["requirements.txt", "Pipfile", "pyproject.toml", "setup.py"]
        deps = []
        for df in dep_files:
            p = os.path.join(root, df)
            if os.path.isfile(p):
                try:
                    with open(p) as f:
                        content = f.read()
                    deps.append(f"--- from {df} ---")
                    for line in content.splitlines():
                        line = line.strip()
                        if line and not line.startswith(('#', '//', '/*')):
                            deps.append(f"  {line}")
                except:
                    deps.append(f"  (could not read {df})")
        return deps[:25]

    def _find_key_definitions(self, root: str) -> List[tuple]:
        """Find top-level classes and functions in Python files"""
        definitions = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
            for f in filenames:
                if f.endswith('.py'):
                    fpath = os.path.join(dirpath, f)
                    rel = os.path.relpath(fpath, root)
                    try:
                        with open(fpath) as fh:
                            tree = ast.parse(fh.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                definitions.append((node.name, rel, "class"))
                            elif isinstance(node, ast.FunctionDef):
                                # Only top-level functions
                                is_top = not any(isinstance(p, (ast.ClassDef, ast.FunctionDef))
                                                  for p in ast.walk(tree)
                                                  if p is not node and hasattr(p, 'body') and node in p.body)
                                if is_top:
                                    definitions.append((node.name, rel, "def"))
                    except:
                        pass
        return sorted(definitions, key=lambda x: x[0])


class SearchReplaceTool:
    name = "search_replace"
    description = "Search and replace text across multiple files using regex"

    schema = {
        "description": "Find and replace text patterns across files",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "replacement": {"type": "string", "description": "Replacement text"},
                "include": {"type": "string", "description": "File glob pattern (e.g. *.py, *.{ts,tsx})", "default": "*"},
                "path": {"type": "string", "description": "Root directory (default: current dir)", "default": None},
                "dry_run": {"type": "boolean", "description": "Preview changes without applying", "default": True}
            }
        }
    }

    def execute(self, pattern: str, replacement: str = "", include: str = "*",
                path: str = None, dry_run: bool = True) -> str:
        root = path or os.getcwd()
        matches = []
        import fnmatch

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
            for f in filenames:
                if not fnmatch.fnmatch(f, include):
                    continue
                fp = os.path.join(dirpath, f)
                try:
                    with open(fp, 'r', encoding='utf-8', errors='replace') as fh:
                        content = fh.read()
                    new_content, count = re.subn(pattern, replacement, content)
                    if count > 0:
                        rel = os.path.relpath(fp, root)
                        matches.append({
                            'file': rel,
                            'count': count,
                            'preview': self._get_preview(content, new_content, pattern)
                        })
                except:
                    continue

        if not matches:
            return f"No matches found for pattern: {pattern}"

        lines = []
        if dry_run:
            lines.append(f"🔍 DRY RUN — Found {len(matches)} file(s) with matches:\n")
        else:
            lines.append(f"✏️  Replaced in {len(matches)} file(s):\n")

        for m in sorted(matches, key=lambda x: -x['count']):
            lines.append(f"  {m['file']} — {m['count']} match(es)")
            if m['preview']:
                lines.append(f"    {m['preview']}")

        if not dry_run:
            for m in matches:
                fp = os.path.join(root, m['file'])
                try:
                    with open(fp, 'r', encoding='utf-8') as fh:
                        content = fh.read()
                    content = re.sub(pattern, replacement, content)
                    with open(fp, 'w', encoding='utf-8') as fh:
                        fh.write(content)
                except Exception as e:
                    lines.append(f"  [error writing {m['file']}: {e}]")

        lines.append(f"\n{'DRY RUN — no files modified' if dry_run else 'Done'}")
        return "\n".join(lines)

    def _get_preview(self, old: str, new: str, pattern: str) -> str:
        """Show a concise preview of what changed"""
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        changed = []
        for i, (ol, nl) in enumerate(zip(old_lines, new_lines)):
            if ol != nl:
                changed.append(f"L{i+1}: -{ol.strip()[:60]} -> +{nl.strip()[:60]}")
                if len(changed) >= 3:
                    changed.append("  ...")
                    break
        return "; ".join(changed) if changed else ""


class GitTool:
    name = "git"
    description = "Git operations: status, diff, log, commit, branch"

    schema = {
        "description": "Run git commands on the current repository",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Git action: status, diff, log, commit, branch, add",
                    "enum": ["status", "diff", "log", "commit", "branch", "add"]
                },
                "message": {"type": "string", "description": "Commit message (required for commit)", "default": ""},
                "files": {"type": "string", "description": "Files to add (for add/commit), space-separated", "default": "."},
                "max_count": {"type": "integer", "description": "Max log entries (for log)", "default": 10}
            }
        }
    }

    def execute(self, action: str, message: str = "", files: str = ".",
                max_count: int = 10) -> str:
        try:
            if action == "status":
                result = subprocess.run(["git", "status", "--short"],
                                        capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return f"[not a git repository or git error]\n{result.stderr}"
                output = result.stdout.strip()
                if not output:
                    return "Clean working tree (no changes)"
                lines = ["Git status:\n"]
                for line in output.splitlines():
                    status = line[:2]
                    file = line[3:]
                    if status == "M ":
                        lines.append(f"  📝 modified: {file}")
                    elif status == "??":
                        lines.append(f"  ➕ untracked: {file}")
                    elif status == "A ":
                        lines.append(f"  ✅ added: {file}")
                    elif status == "D ":
                        lines.append(f"  ❌ deleted: {file}")
                    elif status == "R ":
                        lines.append(f"  🔄 renamed: {file}")
                    else:
                        lines.append(f"  {status}: {file}")
                return "\n".join(lines)

            elif action == "diff":
                result = subprocess.run(["git", "diff", "--stat"],
                                        capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return f"[git error]\n{result.stderr}"
                stat = result.stdout.strip()
                result2 = subprocess.run(["git", "diff"],
                                         capture_output=True, text=True, timeout=10)
                diff = result2.stdout.strip()
                lines = [f"Diff stat:\n{stat}\n"] if stat else ["No diff\n"]
                if diff:
                    # Truncate long diffs
                    if len(diff) > 3000:
                        diff = diff[:3000] + "\n... (truncated)"
                    lines.append(diff)
                return "\n".join(lines)

            elif action == "log":
                result = subprocess.run(
                    ["git", "log", f"--max-count={max_count}",
                     "--pretty=format:%h %ad %s", "--date=short"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    return f"[git error]\n{result.stderr}"
                output = result.stdout.strip()
                if not output:
                    return "No commits"
                lines = [f"Last {max_count} commits:\n"]
                for entry in output.splitlines():
                    parts = entry.split(' ', 2)
                    if len(parts) == 3:
                        hash_, date, msg = parts
                        lines.append(f"  {hash_}  {date}  {msg}")
                return "\n".join(lines)

            elif action == "add":
                result = subprocess.run(["git", "add", files],
                                        capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return f"[git error]\n{result.stderr}"
                return f"Added: {files}"

            elif action == "commit":
                if not message:
                    return "[error: commit message required]"
                result = subprocess.run(["git", "add", files],
                                        capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return f"[git add failed]\n{result.stderr}"
                result = subprocess.run(["git", "commit", "-m", message],
                                        capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return f"[commit failed]\n{result.stderr}"
                return f"Committed:\n{result.stdout}"

            elif action == "branch":
                result = subprocess.run(["git", "branch", "-a"],
                                        capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    return f"[git error]\n{result.stderr}"
                branches = result.stdout.strip()
                return f"Branches:\n{branches}"

            return f"[unknown git action: {action}]"

        except subprocess.TimeoutExpired:
            return "[error: git command timed out]"
        except FileNotFoundError:
            return "[error: git not installed]"
        except Exception as e:
            return f"[error: {e}]"


class LintTool:
    name = "lint"
    description = "Format and lint Python code using autopep8 or ruff"

    schema = {
        "description": "Auto-format and check code quality",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory to lint", "default": "."},
                "fix": {"type": "boolean", "description": "Auto-fix issues when possible", "default": True}
            }
        }
    }

    def execute(self, path: str = ".", fix: bool = True) -> str:
        lines = []
        results = []

        # Try ruff first (fastest)
        ruff_ok = self._run_ruff(path, fix, results)
        # Fallback to autopep8
        if not ruff_ok:
            self._run_autopep8(path, fix, results)

        if not results:
            lines.append("No linter found. Install one:")
            lines.append("  pip install ruff")
            lines.append("  pip install autopep8")
        else:
            for r in results:
                lines.append(r)

        return "\n".join(lines) if lines else "No issues found"

    def _run_ruff(self, path: str, fix: bool, results: List[str]) -> bool:
        try:
            cmd = ["ruff", "check", path]
            if fix:
                cmd.append("--fix")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and not result.stdout:
                results.append("✓ ruff: No issues found")
            else:
                output = result.stdout.strip() or result.stderr.strip()
                if output:
                    # Truncate if too long
                    if len(output) > 2000:
                        output = output[:2000] + "\n... (truncated)"
                    results.append(f"ruff:\n{output}")
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_autopep8(self, path: str, fix: bool, results: List[str]):
        try:
            if fix:
                cmd = ["autopep8", "--in-place", "--max-line-length", "100", "-r", path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    results.append("✓ autopep8: Formatted")
                else:
                    results.append(f"autopep8 error: {result.stderr[:500]}")
            else:
                cmd = ["autopep8", "--diff", "--max-line-length", "100", "-r", path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.stdout.strip():
                    diff = result.stdout.strip()
                    if len(diff) > 2000:
                        diff = diff[:2000] + "\n... (truncated)"
                    results.append(f"autopep8 would change:\n{diff}")
                else:
                    results.append("✓ autopep8: No changes needed")
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            results.append("[autopep8 timed out]")


class BackgroundTool:
    name = "background"
    description = "Run processes in background, manage conda envs, view live logs"

    schema = {
        "description": "Start/stop/monitor background processes and run in conda environments",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "start / stop / logs / status / conda_run",
                    "enum": ["start", "stop", "logs", "status", "conda_run"]
                },
                "name": {"type": "string", "description": "Process name (for start/stop/logs)", "default": ""},
                "command": {"type": "string", "description": "Command to execute (for start/conda_run)", "default": ""},
                "env_name": {"type": "string", "description": "Conda env name (for conda_run)", "default": ""},
                "workdir": {"type": "string", "description": "Working directory", "default": None},
                "tail": {"type": "integer", "description": "Number of recent log lines to show (default 50)", "default": 50}
            },
            "required": ["action"]
        }
    }

    _processes: Dict[str, Dict] = {}

    def execute(self, action: str, name: str = "", command: str = "",
                env_name: str = "", workdir: str = None, tail: int = 50) -> str:
        if action == "start":
            return self._start(name, command, workdir)
        elif action == "stop":
            return self._stop(name)
        elif action == "logs":
            return self._logs(name, tail)
        elif action == "status":
            return self._status()
        elif action == "conda_run":
            return self._conda_run(command, env_name, workdir)
        return f"[unknown action: {action}]"

    def _start(self, name: str, command: str, workdir: str = None) -> str:
        if not name or not command:
            return "[error: name and command required]"
        if name in self._processes and self._processes[name]["process"].poll() is None:
            return f"[error: process '{name}' is already running]"

        cwd = workdir or os.getcwd()
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                text=True,
                bufsize=1,
            )
            self._processes[name] = {
                "process": proc,
                "command": command,
                "workdir": cwd,
                "started": time.strftime("%H:%M:%S"),
                "log": [],
            }
            return f"Started '{name}' (PID {proc.pid}) in {cwd}\nCommand: {command}"
        except Exception as e:
            return f"[error starting process: {e}]"

    def _stop(self, name: str) -> str:
        proc_info = self._processes.get(name)
        if not proc_info:
            return f"[error: no process named '{name}']"

        proc = proc_info["process"]
        if proc.poll() is not None:
            return f"Process '{name}' already exited (code {proc.returncode})"

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return f"Stopped '{name}' (PID {proc.pid})"

    def _logs(self, name: str, tail: int = 50) -> str:
        proc_info = self._processes.get(name)
        if not proc_info:
            return f"[error: no process named '{name}']"

        proc = proc_info["process"]
        # Read any new output from the pipe
        try:
            while True:
                line = proc.stdout.readline()
                if line:
                    proc_info["log"].append(line.rstrip())
                else:
                    break
        except Exception:
            pass

        log = proc_info["log"]
        status = "running" if proc.poll() is None else f"exited (code {proc.returncode})"
        recent = log[-tail:] if log else ["(no output yet)"]

        lines = [
            f"Process: {name}  |  Status: {status}  |  PID: {proc.pid}  |  Started: {proc_info['started']}",
            f"Command: {proc_info['command']}  |  CWD: {proc_info['workdir']}",
            f"Total log lines: {len(log)}  |  Showing last {min(tail, len(log))}\n",
        ]
        for i, line in enumerate(recent):
            lines.append(f"{i+1:>4}| {line}")

        return "\n".join(lines)

    def _status(self) -> str:
        if not self._processes:
            return "No background processes."

        lines = [f"{'Name':<20} {'PID':<8} {'Status':<12} {'Started':<10} Command", "-" * 80]
        for name, info in self._processes.items():
            proc = info["process"]
            status = "running" if proc.poll() is None else f"done ({proc.returncode})"
            cmd = info["command"][:50]
            lines.append(f"{name:<20} {proc.pid:<8} {status:<12} {info['started']:<10} {cmd}")

        return "\n".join(lines)

    def _conda_run(self, command: str, env_name: str, workdir: str = None) -> str:
        if not command:
            return "[error: command required]"
        if env_name:
            full_cmd = f"conda run -n {env_name} {command}"
        else:
            full_cmd = command

        cwd = workdir or os.getcwd()
        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=cwd,
            )
            output = []
            if result.stdout:
                output.append(result.stdout.rstrip())
            if result.stderr:
                output.append(f"[stderr]\n{result.stderr.rstrip()}")
            if result.returncode != 0:
                output.append(f"[exit code: {result.returncode}]")
            return "\n".join(output) if output else f"[command completed with exit code {result.returncode}, no output]"
        except subprocess.TimeoutExpired:
            return "[error: command timed out after 300s]"
        except Exception as e:
            return f"[error: {e}]"
