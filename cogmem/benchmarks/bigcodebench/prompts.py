"""System prompts and formatting for BigCodeBench tasks."""

SYSTEM_PROMPT = """You are an expert Python programmer. You will be given a programming task with a function signature and docstring. Write the complete implementation.

Rules:
1. Implement ONLY the function body — do not redefine the signature or imports already provided.
2. Think step-by-step before writing code.
3. Use only standard library and the packages mentioned in the docstring.
4. Your code must be correct and handle edge cases.

Your response MUST follow this exact format:

Thought: <your reasoning about how to solve this>
Code:
```python
<your implementation>
```"""

SYSTEM_PROMPT_RAW = """You are an expert Python programmer. Given a task description, write a complete Python function implementation. Think step-by-step, then provide your code."""


def format_task_prompt(task: dict, use_instruct: bool = True) -> str:
    """Format a BigCodeBench task into a prompt for the model.

    Args:
        task: BigCodeBench task dict with 'instruct_prompt' or 'complete_prompt'.
        use_instruct: Use instruction-style prompt (True) or completion-style (False).
    """
    if use_instruct and task.get("instruct_prompt"):
        return task["instruct_prompt"]
    return task.get("complete_prompt", "")


def format_messages(task: dict, use_instruct: bool = True) -> list[dict]:
    """Format task as chat messages for LLM client."""
    prompt = format_task_prompt(task, use_instruct=use_instruct)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def format_for_training(task_description: str, code: str) -> str:
    """Wrap clean code in Thought+Code format to match eval prompt.

    For episodes where the model output raw code (no Thought),
    this wraps it in the expected format so training data is consistent.
    """
    desc = task_description[:150].rsplit(" ", 1)[0].rstrip(".,;:")
    return (
        f"Thought: I need to implement a function that "
        f"{desc}.\n"
        f"Code:\n```python\n{code}\n```"
    )


def extract_code(response: str, task: dict | None = None) -> str:
    """Extract Python code from model response.

    Handles:
    - Code blocks with ```python ... ```
    - Code blocks with ``` ... ```
    - Raw code (no markers)
    """
    import re

    # Try ```python blocks first
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` blocks
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If response has "Code:" marker, take everything after it
    if "Code:" in response:
        code_part = response.split("Code:", 1)[1].strip()
        # Remove any remaining markdown fences
        code_part = re.sub(r"^```\w*\s*\n?", "", code_part)
        code_part = re.sub(r"\n?```\s*$", "", code_part)
        return code_part.strip()

    # Last resort: return everything that looks like code (has def/import/return)
    lines = response.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "import ", "from ", "class ", "return ", "    ")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return response.strip()
