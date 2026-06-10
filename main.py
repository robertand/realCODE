#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from model_manager import ModelManager
from conversation import Conversation


def list_recent_models(config):
    recent = config.get("recent_models", [])
    if recent:
        print("Recent models:")
        for i, m in enumerate(recent, 1):
            print(f"  {i}. {m}")
    else:
        print("No recent models yet.")


def main():
    parser = argparse.ArgumentParser(
        description="Antilocal — Run HuggingFace coding LLMs locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py -m Qwen/Qwen2.5-Coder-7B-Instruct
  python main.py -m coder3101/Qwen3.5-2B-heretic -q 4bit
  python main.py -m lilyzhng/Qwen2.5-Coder-14B-r32-20260215-065301 -q 4bit
  python main.py --list-models
        """
    )

    parser.add_argument("-m", "--model", type=str, default=None,
                        help="HuggingFace model ID to load")
    parser.add_argument("-q", "--quantization", type=str,
                        choices=["4bit", "8bit", "none"],
                        default=None,
                        help="Quantization mode (default: 4bit)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        default=None,
                        help="Device to run on")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Generation temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max tokens per generation (default: 4096)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Custom system prompt file path")
    parser.add_argument("--list-models", action="store_true",
                        help="Show recently used models")

    args = parser.parse_args()
    config = Config()

    if args.list_models:
        list_recent_models(config)
        return

    model_id = args.model or config["model"]
    quantization = args.quantization or config["quantization"]
    device = args.device or config["device"]

    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.max_tokens is not None:
        config["max_tokens"] = args.max_tokens
    if args.system_prompt:
        try:
            with open(args.system_prompt, "r") as f:
                config["system_prompt"] = f.read()
        except Exception as e:
            print(f"Error reading system prompt file: {e}")
            sys.exit(1)

    config.add_recent_model(model_id)
    config.save()

    mm = ModelManager(
        model_id=model_id,
        quantization=quantization,
        device=device,
    )

    try:
        mm.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTips:")
        print("  - Make sure you have enough GPU memory")
        print("  - Try a smaller model or use -q none")
        print("  - Use --device cpu to force CPU mode")
        print("  - For 4bit/8bit: pip install bitsandbytes")
        print("  - Update transformers: pip install --upgrade transformers")
        sys.exit(1)

    conv = Conversation(mm, config)
    try:
        conv.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        mm.unload()


if __name__ == "__main__":
    main()
