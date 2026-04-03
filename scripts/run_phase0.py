"""Phase 0: Pre-flight checks before spending compute or money.

Run this first. It verifies:
1. Ollama is running and serves llama3.2:3b
2. Together AI supports the target model for fine-tuning
3. ALFWorld is installed (may be in WSL on Windows)
4. Required Python packages are importable

Split-environment support (Windows):
- Windows: Ollama, Together AI, ML packages (torch, transformers, etc.)
- WSL:     ALFWorld + episode collection (alfworld needs Linux)

Usage:
    python scripts/run_phase0.py          # Windows checks + WSL alfworld probe
    python3 scripts/run_phase0.py --wsl   # Run inside WSL (alfworld only)
"""

import platform
import shutil
import subprocess
import sys


def check_ollama():
    print("[1/4] Checking Ollama...")
    try:
        import json
        import urllib.request

        payload = json.dumps({
            "model": "llama3.2:3b",
            "prompt": "Say OK",
            "options": {"num_ctx": 2048},
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = data.get("response", "")
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
    # Direct import (works on Linux/WSL)
    try:
        import alfworld

        version = getattr(alfworld, "__version__", "unknown")
        print(f"  OK — alfworld {version} installed")
        return True
    except ImportError:
        pass

    # On Windows, probe WSL for alfworld
    if platform.system() == "Windows" and shutil.which("wsl"):
        print("  INFO — alfworld not available on Windows, checking WSL...")
        try:
            result = subprocess.run(
                [
                    "wsl", "-d", "Ubuntu", "-u", "chucky", "-e",
                    "python3", "-c", "import alfworld; print('OK')",
                ],
                capture_output=True, text=True, timeout=15,
            )
            if "OK" in result.stdout:
                print("  OK — alfworld available in WSL (episode collection will run there)")
                return True
        except Exception:
            pass

    print("  FAIL — alfworld not installed")
    print("  Fix: In WSL run: pip3 install alfworld --break-system-packages && alfworld-download")
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
    wsl_mode = "--wsl" in sys.argv

    print("=" * 50)
    if wsl_mode:
        print("CogMem Phase 0: Pre-Flight Checks (WSL mode)")
    else:
        print("CogMem Phase 0: Pre-Flight Checks")
    print("=" * 50 + "\n")

    if wsl_mode:
        # WSL mode: only check alfworld
        results = [check_alfworld()]
    else:
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
