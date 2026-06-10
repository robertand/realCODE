import json
import re
import traceback
import subprocess
import os
import time
from typing import List, Dict, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
from rich.columns import Columns
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

from tools import TOOL_REGISTRY, get_tool_descriptions


TOOL_CALL_RE = re.compile(
    r"<tool>\s*\n?\s*(\{.*?\})\s*\n?\s*</tool>", re.DOTALL
)

MAX_CONTEXT_TOKENS = 32000
WARNING_THRESHOLD = 0.75
COMPRESS_THRESHOLD = 0.85

console = Console(soft_wrap=True, highlight=False)


def styled_text(text: str, style: str = "") -> Text:
    return Text(text, style=style)


class Conversation:
    def __init__(self, model_manager, config):
        self.mm = model_manager
        self.config = config
        self.messages = []
        self.original_message_count = 0
        self.summary_count = 0
        self.last_token_count = 0
        self._thinking = False
        self._last_status = ""

        self._init_prompt()

    def _init_prompt(self):
        commands = ["/exit", "/clear", "/model", "/cd", "/shell", "/context"]
        tool_names = list(TOOL_REGISTRY.keys())
        completer = WordCompleter(commands + tool_names, ignore_case=True, sentence=True)

        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):
            if self._thinking:
                console.print("\n[dim]Interrupted[/dim]")
                self._thinking = False
                raise KeyboardInterrupt()
            event.app.exit()

        self.prompt_session = PromptSession(
            history=FileHistory(os.path.expanduser("~/.antilocal_history")),
            completer=completer,
            key_bindings=kb,
            multiline=False,
            tempfile_suffix=".py",
        )

    def _draw_progress_bar(self, percentage: float, width: int = 30) -> Text:
        filled = int(width * percentage / 100)
        empty = width - filled

        if percentage >= 85:
            color = "red"
        elif percentage >= 75:
            color = "yellow"
        else:
            color = "green"

        bar = Text()
        bar.append("█" * filled, style=color)
        bar.append("░" * empty, style="bright_black")
        bar.append(f" {percentage:5.1f}%", style=color if percentage >= 75 else "")
        return bar

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            total_chars += len(content)
        return int(total_chars / 2.5)

    def _compress_conversation(self):
        console.rule("[yellow]Compressing context...[/yellow]")

        system_prompt = self.messages[0] if self.messages else None
        keep_recent = 10
        to_compress = self.messages[1:-keep_recent] if len(self.messages) > keep_recent + 1 else []

        if not to_compress:
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("[dim]Analyzing conversation history...", total=None)
            time.sleep(0.3)

            summary_prompt = f"""Please provide a concise summary of the following conversation so far. 
Focus on:
- What tasks have been completed
- What files were created/modified
- What commands were run
- Current state of the project
- Any important decisions made

Keep the summary under 500 words and focus on the most important information.

Conversation to summarize:
{self._format_messages_for_summary(to_compress)}"""

            try:
                task2 = progress.add_task("[yellow]Generating summary...", total=None)

                summary_messages = [
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user", "content": summary_prompt}
                ]

                summary = self.mm.generate(
                    summary_messages,
                    max_tokens=1000,
                    temperature=0.3,
                    top_p=0.9
                )

                progress.update(task2, completed=True)

                new_messages = [system_prompt] if system_prompt else []
                new_messages.append({
                    "role": "system",
                    "content": f"Previous conversation summary - {len(to_compress)} messages compressed\n{summary}\nEnd of summary"
                })
                new_messages.extend(self.messages[-keep_recent:])

                old_count = len(self.messages)
                self.messages = new_messages
                self.summary_count += 1

                progress.add_task(
                    f"[green]Compressed {old_count} -> {len(self.messages)} messages",
                    total=None, completed=True
                )

            except Exception as e:
                progress.add_task(f"[red]Compression failed: {e}", total=None, completed=True)
                self.messages = [self.messages[0]] + self.messages[-20:]

    def _format_messages_for_summary(self, messages: List[Dict]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 2000:
                content = content[:2000] + "..."
            formatted.append(f"{role.upper()}: {content}")
        return "\n\n".join(formatted)

    def _check_and_compress(self):
        estimated_tokens = self._estimate_tokens(self.messages)
        usage_percent = (estimated_tokens / MAX_CONTEXT_TOKENS) * 100
        self.last_token_count = estimated_tokens

        if usage_percent > COMPRESS_THRESHOLD * 100:
            self._compress_conversation()
            new_tokens = self._estimate_tokens(self.messages)
            new_percent = (new_tokens / MAX_CONTEXT_TOKENS) * 100
            return new_percent
        return usage_percent

    def _build_status_table(self, usage_percent: float = None) -> Table:
        if usage_percent is None:
            tokens = self._estimate_tokens(self.messages)
            usage_percent = (tokens / MAX_CONTEXT_TOKENS) * 100

        table = Table.grid(padding=(0, 1))
        table.add_column()

        context_bar = self._draw_progress_bar(usage_percent)
        row = Text()
        row.append("Context: ", style="bold")
        row.append(context_bar)
        table.add_row(row)

        gpu_mem = self._get_gpu_memory()
        if gpu_mem:
            gpu_percent = (gpu_mem['used'] / gpu_mem['total']) * 100
            gpu_bar = self._draw_progress_bar(gpu_percent)
            row = Text()
            row.append("GPU:     ", style="bold")
            row.append(gpu_bar)
            row.append(f"  ({gpu_mem['used']:.0f}/{gpu_mem['total']:.0f} MiB)", style="dim")
            table.add_row(row)

        table.add_row(Text(f"CWD: {os.getcwd()}", style="dim"))
        table.add_row(
            Text(f"Msgs: {len(self.messages)}  |  Tokens: {self.last_token_count:,}  |  Summaries: {self.summary_count}",
                 style="dim")
        )

        return table

    def _build_welcome(self) -> Panel:
        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold cyan")
        info.add_column()
        info.add_row("Model", self.mm.model_id)
        tool_list = ", ".join(TOOL_REGISTRY.keys())
        info.add_row("Tools", tool_list)
        info.add_row("Context", f"{MAX_CONTEXT_TOKENS:,} tokens")
        info.add_row("Compression", f"at {COMPRESS_THRESHOLD * 100}%")
        info.add_row("Commands", "/exit  /clear  /model <id>  /cd <path>  /shell <cmd>  /context  /bg")

        return Panel(
            info,
            title="[bold yellow]Antilocal[/bold yellow] — Local Coding Agent",
            border_style="yellow",
            box=box.HEAVY,
        )

    def _display_model_response(self, response: str):
        lines = response.split("\n")
        code_block = False
        buffer = []

        for line in lines:
            if line.startswith("```"):
                if code_block:
                    lang = buffer[0] if buffer else ""
                    code = "\n".join(buffer[1:]) if len(buffer) > 1 else ""
                    if code:
                        try:
                            syntax = Syntax(code, lang or "python", theme="monokai", line_numbers=True)
                            console.print(Panel(syntax, border_style="green"))
                        except Exception:
                            console.print(Panel(code, border_style="green"))
                    buffer = []
                    code_block = False
                else:
                    code_block = True
                    buffer = [line[3:].strip()]
                continue

            if code_block:
                buffer.append(line)
            else:
                console.print(line)

        if code_block and buffer:
            console.print("\n".join(buffer))

    def run(self):
        tool_desc = get_tool_descriptions()
        system = self.config["system_prompt"].replace("{tools}", tool_desc)
        self.messages.append({"role": "system", "content": system})

        console.print()
        console.print(self._build_welcome())
        console.print()
        console.print(self._build_status_table(0))
        console.print()

        while True:
            try:
                user_input = self.prompt_session.prompt(
                    HTML("<ansibrightgreen>\n>>> </ansibrightgreen>"),
                    style="",
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break

            if not user_input.strip():
                continue

            cmd = user_input.strip()

            if cmd == "/exit":
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break

            if cmd == "/clear":
                self.messages = [self.messages[0]]
                self.summary_count = 0
                console.print("[green]Conversation cleared[/green]")
                console.print(self._build_status_table(0))
                continue

            if cmd == "/context":
                tokens = self._estimate_tokens(self.messages)
                usage = (tokens / MAX_CONTEXT_TOKENS) * 100
                detail = Table.grid(padding=(0, 1))
                detail.add_column(style="bold")
                detail.add_column()
                detail.add_row("Messages", str(len(self.messages)))
                detail.add_row("Estimated tokens", f"{tokens:,} / {MAX_CONTEXT_TOKENS:,}")
                detail.add_row("Usage", f"{usage:.1f}%")
                detail.add_row("Summaries", str(self.summary_count))
                console.print(Panel(detail, title="Context Info", border_style="cyan"))
                console.print(self._build_status_table(usage))
                continue

            if cmd.startswith("/model "):
                new_model = cmd[7:].strip()
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    progress.add_task(f"[yellow]Switching to {new_model}...", total=None)
                    self.mm.unload()
                    self.mm.model_id = new_model
                    self.config.add_recent_model(new_model)
                    self.mm.load()
                    self.messages = [self.messages[0]]
                    self.summary_count = 0
                console.print(f"[green]Switched to {new_model}[/green]")
                console.print(self._build_status_table(0))
                continue

            if cmd.startswith("/cd "):
                new_dir = cmd[4:].strip()
                try:
                    os.chdir(new_dir)
                    console.print(f"[green]Changed working directory to: {os.getcwd()}[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to change directory: {e}[/red]")
                continue

            if cmd.startswith("/shell "):
                shell_cmd = cmd[7:].strip()
                console.print(f"[dim]Executing: {shell_cmd}[/dim]")
                result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
                if result.stdout:
                    console.print(Syntax(result.stdout, "bash", theme="monokai", word_wrap=True))
                if result.stderr:
                    console.print(Syntax(result.stderr, "bash", theme="monokai", word_wrap=True))
                console.print(f"[dim]exit code: {result.returncode}[/dim]")
                continue

            self.messages.append({"role": "user", "content": user_input})
            usage = self._check_and_compress()
            used_tools = self._run_tool_loop()

            # Generează un raport scurt după ce tool-urile au terminat
            if used_tools:
                console.print()
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("[yellow]Generating summary...", total=None)
                    self.messages.append({
                        "role": "user",
                        "content": "Summarize what you just did. Keep it short - 3-5 bullet points of the key actions and results."
                    })
                    summary = self.mm.generate(
                        self.messages,
                        max_tokens=512,
                        temperature=0.3,
                        top_p=0.9,
                    )
                    self.messages.pop()

                console.print(Panel(
                    Markdown(summary),
                    title="[bold yellow]Report[/bold yellow]",
                    border_style="yellow",
                    box=box.ROUNDED,
                ))

            console.print(self._build_status_table())

    def _run_tool_loop(self, max_iterations=50) -> bool:
        used_tools = False
        for iteration in range(max_iterations):
            console.print(self._build_status_table())

            self._thinking = True

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            ) as progress:
                progress.add_task("[yellow]Thinking...", total=None)
                start_time = time.time()

                response = self.mm.generate(
                    self.messages,
                    max_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    top_p=self.config["top_p"],
                )

            self._thinking = False
            elapsed = time.time() - start_time

            tool_matches = TOOL_CALL_RE.findall(response)
            self.messages.append({"role": "assistant", "content": response})

            if not tool_matches:
                if not used_tools:
                    console.print()
                    self._display_model_response(response)
                    console.print()
                return used_tools

            used_tools = True

            for match in tool_matches:
                try:
                    tool_call = json.loads(match)
                except json.JSONDecodeError as e:
                    console.print(f"[red]JSON error: {e}[/red]")
                    continue

                name = tool_call.get("name", "")
                arguments = tool_call.get("arguments", {})
                tool = TOOL_REGISTRY.get(name)

                args_preview = json.dumps(arguments, indent=2)
                if len(args_preview) > 500:
                    args_preview = args_preview[:500] + "\n  ..."

                console.print(Panel(
                    args_preview,
                    title=f"[bold blue]Using tool: {name}[/bold blue]",
                    border_style="blue",
                    box=box.ROUNDED,
                ))

                if tool is None:
                    result = f"[error: unknown tool '{name}'. Available: {', '.join(TOOL_REGISTRY.keys())}]"
                else:
                    try:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            console=console,
                            transient=True,
                        ) as progress:
                            progress.add_task(f"[dim]Executing {name}...", total=None)
                            tool_start = time.time()
                            result = tool.execute(**arguments)
                            tool_elapsed = time.time() - tool_start
                    except Exception as e:
                        result = f"[error: {e}]\n{traceback.format_exc()}"
                        tool_elapsed = 0

                # Display the result in a panel, with syntax highlighting if it looks like code
                result_stripped = result.replace("[error:", "").replace("[stderr]", "")
                if "import " in result[:200] or "def " in result[:200] or "class " in result[:200]:
                    display = Syntax(result, "python", theme="monokai", word_wrap=True)
                elif result.startswith("Total") or "Error" in result[:100]:
                    display = result
                else:
                    display = result

                console.print(Panel(
                    display,
                    title=f"[dim]Result ({tool_elapsed:.2f}s)[/dim]",
                    border_style="bright_black" if "[error:" in result else "green",
                    box=box.ROUNDED,
                ))

                self.messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result}\n</tool_response>",
                })

                console.print(self._build_status_table())

        if iteration == max_iterations - 1:
            console.print("[red]Reached max tool call iterations[/red]")
        return used_tools

    def _get_gpu_memory(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {"used": float(parts[0]), "total": float(parts[1])}
        except Exception:
            pass
        return None
