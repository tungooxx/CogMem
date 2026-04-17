"""Tests for cogmem.benchmarks.bigcodebench modules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --- prompts tests ---

class TestPrompts:
    def test_format_task_prompt_instruct(self):
        from cogmem.benchmarks.bigcodebench.prompts import format_task_prompt

        task = {"instruct_prompt": "Write a function...", "complete_prompt": "def foo():"}
        assert format_task_prompt(task, use_instruct=True) == "Write a function..."

    def test_format_task_prompt_complete(self):
        from cogmem.benchmarks.bigcodebench.prompts import format_task_prompt

        task = {"instruct_prompt": "Write a function...", "complete_prompt": "def foo():"}
        assert format_task_prompt(task, use_instruct=False) == "def foo():"

    def test_format_messages(self):
        from cogmem.benchmarks.bigcodebench.prompts import format_messages

        task = {"instruct_prompt": "Write a function...", "complete_prompt": "def foo():"}
        msgs = format_messages(task, use_instruct=True)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Write a function..."

    def test_extract_code_python_block(self):
        from cogmem.benchmarks.bigcodebench.prompts import extract_code

        response = """Thought: I need to implement this.
Code:
```python
def foo():
    return 42
```"""
        code = extract_code(response)
        assert "def foo():" in code
        assert "return 42" in code

    def test_extract_code_generic_block(self):
        from cogmem.benchmarks.bigcodebench.prompts import extract_code

        response = """Here is the solution:
```
def bar():
    return "hello"
```"""
        code = extract_code(response)
        assert "def bar():" in code

    def test_extract_code_no_markers(self):
        from cogmem.benchmarks.bigcodebench.prompts import extract_code

        response = """def baz():
    return True"""
        code = extract_code(response)
        assert "def baz():" in code

    def test_extract_code_with_code_marker(self):
        from cogmem.benchmarks.bigcodebench.prompts import extract_code

        response = "Thought: simple\nCode:\ndef x():\n    pass"
        code = extract_code(response)
        assert "def x():" in code


# --- evaluator tests ---

class TestEvaluator:
    def test_build_test_script(self):
        from cogmem.benchmarks.bigcodebench.evaluator import _build_test_script

        task = {
            "complete_prompt": "import os\ndef task_func():",
            "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_it(self):\n        self.assertEqual(task_func(), 42)",
            "entry_point": "task_func",
        }
        script = _build_test_script(task, "    return 42")
        assert "import os" in script
        assert "return 42" in script
        assert "TestCases" in script
        assert "ALL_TESTS_PASSED" in script

    def test_evaluate_solution_pass(self):
        from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution

        task = {
            "complete_prompt": "def task_func(a, b):\n    \"\"\"Add two numbers.\"\"\"",
            "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(task_func(1, 2), 3)\n    def test_zero(self):\n        self.assertEqual(task_func(0, 0), 0)",
            "entry_point": "task_func",
        }
        result = evaluate_solution(task, "    return a + b", timeout=10)
        assert result["passed"] is True

    def test_evaluate_solution_fail(self):
        from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution

        task = {
            "complete_prompt": "def task_func(a, b):\n    \"\"\"Add two numbers.\"\"\"",
            "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(task_func(1, 2), 3)",
            "entry_point": "task_func",
        }
        result = evaluate_solution(task, "    return a * b", timeout=10)
        assert result["passed"] is False

    def test_evaluate_solution_timeout(self):
        from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution

        task = {
            "complete_prompt": "def task_func():",
            "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_it(self):\n        task_func()",
            "entry_point": "task_func",
        }
        result = evaluate_solution(task, "    import time; time.sleep(100)", timeout=2)
        assert result["passed"] is False
        assert "Timeout" in result["error"]

    def test_evaluate_solution_syntax_error(self):
        from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution

        task = {
            "complete_prompt": "def task_func():",
            "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_it(self):\n        task_func()",
            "entry_point": "task_func",
        }
        result = evaluate_solution(task, "    return !!!invalid", timeout=10)
        assert result["passed"] is False

    def test_evaluate_solution_invalid_mode(self):
        from cogmem.benchmarks.bigcodebench.evaluator import evaluate_solution

        task = {
            "complete_prompt": "def task_func():",
            "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_it(self):\n        task_func()",
            "entry_point": "task_func",
        }
        with pytest.raises(ValueError, match="Unknown eval mode"):
            evaluate_solution(task, "    pass", mode="typo")

    def test_evaluate_batch(self):
        from cogmem.benchmarks.bigcodebench.evaluator import evaluate_batch

        tasks = [
            {"task_id": "t1", "complete_prompt": "def task_func():", "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_it(self):\n        self.assertEqual(task_func(), 1)", "entry_point": "task_func"},
            {"task_id": "t2", "complete_prompt": "def task_func():", "test": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_it(self):\n        self.assertEqual(task_func(), 2)", "entry_point": "task_func"},
        ]
        solutions = {"t1": "    return 1", "t2": "    return 999"}
        result = evaluate_batch(tasks, solutions, timeout=10)
        assert result["total"] == 2
        assert result["passed"] == 1
        assert result["pass_rate"] == 0.5


# --- dataset tests ---

class TestDataset:
    def test_save_and_load_jsonl(self):
        from cogmem.benchmarks.bigcodebench.dataset import (
            load_bigcodebench_from_jsonl,
            save_tasks_jsonl,
        )

        tasks = [
            {"task_id": "t1", "instruct_prompt": "Do X"},
            {"task_id": "t2", "instruct_prompt": "Do Y"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name

        save_tasks_jsonl(tasks, path)
        loaded = load_bigcodebench_from_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["task_id"] == "t1"
        Path(path).unlink()

    def test_get_task_ids(self):
        from cogmem.benchmarks.bigcodebench.dataset import get_task_ids

        tasks = [{"task_id": "b"}, {"task_id": "a"}]
        assert get_task_ids(tasks) == ["a", "b"]

    def test_filter_by_ids(self):
        from cogmem.benchmarks.bigcodebench.dataset import filter_by_ids

        tasks = [{"task_id": "a"}, {"task_id": "b"}, {"task_id": "c"}]
        filtered = filter_by_ids(tasks, {"a", "c"})
        assert len(filtered) == 2

    def test_filter_by_split(self):
        from cogmem.benchmarks.bigcodebench.dataset import filter_by_split

        tasks = [
            {"task_id": "a", "split_name": "train"},
            {"task_id": "b", "split_name": "dev"},
            {"task_id": "c", "split_name": "train"},
        ]
        filtered = filter_by_split(tasks, "train")
        assert [task["task_id"] for task in filtered] == ["a", "c"]


# --- runner tests ---

class TestRunner:
    def test_make_episode(self):
        from cogmem.benchmarks.bigcodebench.runner import _make_episode

        task = {
            "task_id": "BigCodeBench/42",
            "instruct_prompt": "Write foo",
            "complete_prompt": "def foo():",
            "entry_point": "foo",
            "split_name": "train",
            "manifest_id": "manifest_123",
            "source_benchmark": "bigcodebench",
        }
        trajectory = [
            {"attempt": 1, "code": "return 1", "response": "Thought: ...\nCode: ...",
             "test_result": "PASS", "error": None},
        ]
        ep = _make_episode(task, trajectory=trajectory, success=True,
                           final_code="return 1")
        assert ep["task_id"] == "BigCodeBench/42"
        assert ep["success"] is True
        assert ep["q_value"] == 1.0  # success -> 1.0 initial signal
        assert ep["episode_helpfulness"] == 1.0
        assert "bigcode_" in ep["episode_id"]
        assert ep["trajectory"] == trajectory
        assert ep["final_code"] == "return 1"
        assert ep["split_name"] == "train"
        assert ep["manifest_id"] == "manifest_123"
        assert ep["source_benchmark"] == "bigcodebench"
        assert len(ep["task_hash"]) == 64

    def test_make_episode_failure(self):
        from cogmem.benchmarks.bigcodebench.runner import _make_episode

        task = {"task_id": "BigCodeBench/0", "instruct_prompt": "x", "entry_point": "x"}
        trajectory = [
            {"attempt": 1, "code": "", "response": "",
             "test_result": "FAIL", "error": "SyntaxError"},
        ]
        ep = _make_episode(task, trajectory=trajectory, success=False,
                           final_code=None)
        assert ep["success"] is False
        assert ep["q_value"] == 0.0  # failure -> 0.0 initial signal
        assert ep["episode_helpfulness"] == 0.0
        assert ep["num_attempts"] == 1

    def test_load_episodes(self):
        from cogmem.benchmarks.bigcodebench.runner import load_episodes

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write(json.dumps({"task_id": "t1", "success": True}) + "\n")
            f.write(json.dumps({"task_id": "t2", "success": False}) + "\n")
            path = f.name

        eps = load_episodes(path)
        assert len(eps) == 2
        assert eps[0]["task_id"] == "t1"
        Path(path).unlink()

    def test_load_episodes_truncated(self):
        from cogmem.benchmarks.bigcodebench.runner import load_episodes

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write(json.dumps({"task_id": "t1", "success": True}) + "\n")
            f.write('{"task_id": "t2", "succ')  # Truncated line (crash mid-write)
            path = f.name

        eps = load_episodes(path)
        assert len(eps) == 1  # Only the valid line
        assert eps[0]["task_id"] == "t1"
        Path(path).unlink()


# --- training data builder tests ---

class TestBuildTrainingData:
    def test_q_to_copies_success(self):
        from scripts.build_training_data_bigcode import q_to_copies

        # Success should get high copies
        assert q_to_copies(1.0) >= 3

    def test_q_to_copies_failure(self):
        from scripts.build_training_data_bigcode import q_to_copies

        # Failure should get low copies
        assert q_to_copies(-1.0) <= 3
        assert q_to_copies(-0.5) >= 1

    def test_build_training_data(self):
        from scripts.build_training_data_bigcode import build_training_data

        episodes = [
            {
                "task_description": "Write a function",
                "script": "Thought: simple\nCode:\n```python\ndef foo(): return 1\n```",
                "q_value": 1.0,
                "success": True,
            },
            {
                "task_description": "Write bar",
                "script": "Thought: tricky\nCode:\n```python\ndef bar(): pass\n```",
                "q_value": -1.0,
                "success": False,
            },
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(episodes, f)
            mb_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            out_path = f.name

        build_training_data(mb_path, out_path)

        with open(out_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) > 0
        # Should have both system and raw formats
        assert any("system" in str(line) for line in lines)

        Path(mb_path).unlink()
        Path(out_path).unlink()
