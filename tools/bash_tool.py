import subprocess
import os


class BashTool:
    name = "bash"
    description = "Execute a bash command on the system. Returns stdout, stderr, and exit code."

    schema = {
        "description": "Run a shell command and get its output",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default 120000)",
                    "default": 120000
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory (default: current directory)",
                    "default": None
                }
            },
            "required": ["command"]
        }
    }

    def execute(self, command: str, timeout: int = 120000, workdir: str = None) -> str:
        cwd = workdir or os.getcwd()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout / 1000,
                cwd=cwd,
                env=os.environ.copy()
            )
            output = []
            if result.stdout:
                output.append(result.stdout.rstrip())
            if result.stderr:
                output.append(f"[stderr]\n{result.stderr.rstrip()}")
            output.append(f"[exit code: {result.returncode}]")
            return "\n".join(output) if output else f"[command completed with exit code {result.returncode}, no output]"
        except subprocess.TimeoutExpired:
            return f"[error: command timed out after {timeout}ms]"
        except Exception as e:
            return f"[error: {e}]"