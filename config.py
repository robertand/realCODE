import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
    "model": "lilyzhng/Qwen2.5-Coder-14B-r32-20260215-065301",
    "quantization": "4bit",
    "device": "cuda",
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.95,
    "system_prompt": (
        "You are an expert software engineer with access to powerful tools. "
        "Follow this reasoning process:\n\n"
        "1. **UNDERSTAND** - Use reason() to break down complex problems\n"
        "2. **PLAN** - Use plan() to create multi-step strategies\n"
        "3. **EXECUTE** - Use appropriate tools (bash, read, write, etc.)\n"
        "4. **VERIFY** - Check results and adjust if needed\n"
        "5. **LEARN** - Use memory() to remember important information\n\n"
        "**CRITICAL RULES:**\n"
        "- Always use analyze_code() before modifying existing code\n"
        "- Use reason() for any problem that requires more than 3 steps\n"
        "- Use memory() to remember user preferences and project context\n"
        "- Use plan() for tasks with more than 5 steps\n"
        "- Never guess - use tools to verify information\n"
        "- Explain your reasoning before acting\n\n"
        "**Tool format:**\n"
        "<tool>\n"
        '{{"name": "tool_name", "arguments": {{"key": "value"}}}}\n'
        "</tool>\n\n"
        "Available tools:\n{tools}\n\n"
        "Remember: Think step by step, use tools systematically, and document your reasoning."
    ),
    "recent_models": []
}


class Config:
    def __init__(self):
        self.data = dict(DEFAULT_CONFIG)
        if os.path.isfile(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    saved = json.load(f)
                self.data.update(saved)
            except Exception:
                pass

    def save(self):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save config: {e}")

    def add_recent_model(self, model_id: str):
        recent = self.data.get("recent_models", [])
        if model_id in recent:
            recent.remove(model_id)
        recent.insert(0, model_id)
        self.data["recent_models"] = recent[:10]
        self.save()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        self.save()

    def get(self, key, default=None):
        return self.data.get(key, default)
