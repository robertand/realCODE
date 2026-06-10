from .bash_tool import BashTool
from .file_tools import ReadTool, WriteTool, EditTool, GlobTool, GrepTool
from .question_tool import QuestionTool
from .web_tool import WebTool
from .advanced_tools import (
    CodeAnalyzerTool, MemoryTool, PlanTool, ReasoningTool,
    ProjectAnalyzerTool, SearchReplaceTool, GitTool, LintTool,
    BackgroundTool,
)

TOOL_REGISTRY = {
    # Basic tools
    "bash": BashTool(),
    "read": ReadTool(),
    "write": WriteTool(),
    "edit": EditTool(),
    "glob": GlobTool(),
    "grep": GrepTool(),
    "question": QuestionTool(),
    "web": WebTool(),

    # Reasoning & memory
    "analyze_code": CodeAnalyzerTool(),
    "memory": MemoryTool(),
    "plan": PlanTool(),
    "reason": ReasoningTool(),

    # Project intelligence
    "project_analyze": ProjectAnalyzerTool(),
    "search_replace": SearchReplaceTool(),
    "git": GitTool(),
    "lint": LintTool(),

    # Background processes & environments
    "background": BackgroundTool(),
}

def get_tool_descriptions():
    lines = ["## Available Tools\n"]
    for name, tool in TOOL_REGISTRY.items():
        lines.append(f"### {name}")
        lines.append(tool.description)
        schema = getattr(tool, 'schema', None)
        if schema:
            params = schema.get("parameters", {}).get("properties", {})
            if params:
                required = schema.get("parameters", {}).get("required", [])
                for pname, pinfo in params.items():
                    req = " (required)" if pname in required else ""
                    lines.append(f"  - `{pname}`: {pinfo.get('description', '')}{req}")
        lines.append("")
    return "\n".join(lines)


def get_tool_schemas():
    schemas = {}
    for name, tool in TOOL_REGISTRY.items():
        schema = getattr(tool, 'schema', None)
        if schema:
            schemas[name] = schema
    return schemas