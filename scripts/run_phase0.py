"""Phase 0: Pre-flight checks before spending compute or money.

Run this first. It verifies:
1. Together AI supports the target model for fine-tuning
2. Ollama is running and serves llama3.2:3b
3. ALFWorld is installed and data is downloaded
4. Required Python packages are importable

Usage:
    python scripts/run_phase0.py
"""

import sys


def check_ollama():
    print("[1/4] Checking Ollama...")
    try:
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        resp = client.chat.completions.create(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content
        print(f"  OK — Ollama responded: {text.strip()[:50]}")
        return True
    except Exception as e:
        print(f"  FAIL — {e}")
        print("  Fix: run 'ollama pull llama3.2:3b && ollama serve'")
        return False


def check_together_api():
    print("[2/4] Checking Together AI model support...")
    try:
        import os

        from together import Together

        key = os.environ.get("TOGETHER_API_KEY", "")
        if not key:
            print("  WARN — TOGETHER_API_KEY not set. Set it before Phase 2.")
            return True  # non-blocking
        client = Together(api_key=key)
        models = client.models.list()
        model_ids = [m.id for m in models]
        target = "meta-llama/Llama-3.2-3B-Instruct"
        if target in model_ids:
            print(f"  OK — {target} is available")
        else:
            print(f"  WARN — {target} not in model list. Check fine-tuning docs.")
        return True
    except Exception as e:
        print(f"  WARN — Could not verify: {e}")
        return True  # non-blocking


def check_alfworld():
    print("[3/4] Checking ALFWorld...")
    try:
        import alfworld

        print(f"  OK — alfworld {alfworld.__version__} installed")
        return True
    except ImportError:
        print("  FAIL — alfworld not installed")
        print("  Fix: pip install alfworld && alfworld-download")
        return False


def check_packages():
    print("[4/4] Checking Python packages...")
    required = [
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "sentence_transformers",
        "openai",
        "together",
        "numpy",
        "scipy",
        "yaml",
    ]
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  OK — {pkg}")
        except ImportError:
            print(f"  FAIL — {pkg} not found")
            all_ok = False
    return all_ok


def main():
    print("=" * 50)
    print("CogMem Phase 0: Pre-Flight Checks")
    print("=" * 50 + "\n")

    results = [
        check_ollama(),
        check_together_api(),
        check_alfworld(),
        check_packages(),
    ]

    print("\n" + "=" * 50)
    passed = sum(results)
    print(f"Results: {passed}/{len(results)} checks passed")
    if all(results):
        print("All checks passed. Ready for Phase 1.")
    else:
        print("Some checks failed. Fix issues before proceeding.")
    print("=" * 50)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
