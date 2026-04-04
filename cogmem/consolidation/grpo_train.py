"""GRPO training on middle-Q episodes (Step B of consolidation).

Group Relative Policy Optimization: the model generates multiple solutions
per problem, executes them against tests, and learns from the reward signal.

This is where the model gets genuinely smarter — not memorizing answers,
but learning to reason through practice and execution feedback.

Middle-Q episodes are optimal for GRPO: not too easy (already mastered),
not too hard (can't learn from), just right (maximum learning signal).

Two implementations:
1. trl GRPOTrainer: Uses HuggingFace TRL library (preferred)
2. Manual GRPO: Fallback if TRL API doesn't match (generate -> score -> update)

Hardware: A4000 16GB with 4-bit base + DoRA + G=4 generations = ~12-14GB.
"""

import gc
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_grpo_dataset(
    middle_episodes: list[dict],
    anchor_episodes: list[dict],
) -> Dataset:
    """Build dataset for GRPO. Only needs prompts — model generates solutions.

    Mixes middle-Q episodes (learning zone) with high-Q anchors (stability).
    Each row has a "prompt" column and metadata for reward computation.
    """
    rows = []

    for ep in middle_episodes:
        rows.append(_episode_to_grpo_row(ep, zone="middle"))

    for ep in anchor_episodes:
        rows.append(_episode_to_grpo_row(ep, zone="anchor"))

    print(
        f"  GRPO dataset: {len(rows)} problems "
        f"({len(middle_episodes)} middle-Q + {len(anchor_episodes)} anchors)"
    )
    return Dataset.from_list(rows)


def _episode_to_grpo_row(ep: dict, zone: str) -> dict:
    instruction = ep.get("task_description", "")
    return {
        "prompt": _format_grpo_prompt(instruction),
        "task_id": ep.get("task_id", ep.get("episode_id", "")),
        "zone": zone,
        "original_q": ep.get("q_value", 0.0),
        # Metadata for reward computation
        "test_code": ep.get("test", ""),
        "complete_prompt": ep.get("complete_prompt", ""),
        "entry_point": ep.get("entry_point", ""),
        "task_type": ep.get("task_type", ""),
    }


def _format_grpo_prompt(instruction: str) -> str:
    return (
        "You are an expert Python programmer. "
        "Write a complete function that solves the following task. "
        "Include all necessary imports. "
        "Put your code in a ```python code block.\n\n"
        f"Task: {instruction}\n"
    )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def create_bigcodebench_reward_fn(dataset: Dataset):
    """Create a reward function that scores code by executing BigCodeBench tests.

    Returns continuous reward (fraction of tests passed) — NOT binary.
    Continuous reward gives much better RL gradient signal.

    The reward function is called by GRPOTrainer or manual_grpo_step
    with a list of completions to score.
    """
    # Build lookup from prompt -> task metadata
    _task_lookup: dict[str, dict] = {}
    for row in dataset:
        _task_lookup[row["prompt"]] = {
            "test": row["test_code"],
            "complete_prompt": row["complete_prompt"],
            "entry_point": row["entry_point"],
        }

    def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
        """Score completions by executing against test cases.

        Args:
            completions: Generated code strings.
            prompts: Corresponding prompts (used to look up test cases).

        Returns:
            List of float rewards in [-0.5, 1.0].
        """
        rewards = []
        for i, completion in enumerate(completions):
            code = extract_python_code(completion)

            if not code or len(code.strip()) < 10:
                rewards.append(-0.5)
                continue

            # Look up task metadata from prompt
            prompt = prompts[i] if prompts and i < len(prompts) else None
            task_meta = _task_lookup.get(prompt) if prompt else None

            if task_meta and task_meta.get("test"):
                reward = _execute_and_score(code, task_meta)
            else:
                # No test cases — score based on code quality heuristics
                reward = _heuristic_score(code)

            rewards.append(reward)

        return rewards

    return reward_fn


def _execute_and_score(code: str, task_meta: dict) -> float:
    """Execute code against unittest test cases, return fraction passed.

    Uses the same test execution approach as the BigCodeBench evaluator.
    Returns continuous reward: -0.5 (crash) to 1.0 (all tests pass).
    """
    complete_prompt = task_meta.get("complete_prompt", "")
    test_code = task_meta.get("test", "")

    # Build test script
    script = f"""{complete_prompt}
{code}

{test_code}

import unittest, sys
loader = unittest.TestLoader()
suite = loader.loadTestsFromTestCase(TestCases)
runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=0)
result = runner.run(suite)
passed = result.testsRun - len(result.failures) - len(result.errors)
total = result.testsRun
print(f"GRPO_REWARD:{{passed}}/{{total}}")
if result.wasSuccessful():
    print("ALL_TESTS_PASSED")
sys.exit(0 if result.wasSuccessful() else 1)
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr

        # Parse GRPO_REWARD:passed/total
        match = re.search(r"GRPO_REWARD:(\d+)/(\d+)", output)
        if match:
            passed = int(match.group(1))
            total = int(match.group(2))
            if total == 0:
                return -0.3
            return passed / total  # continuous: 0.0 to 1.0

        if "ALL_TESTS_PASSED" in output:
            return 1.0

        return -0.3  # ran but couldn't parse results

    except subprocess.TimeoutExpired:
        return -0.2
    except Exception:
        return -0.5
    finally:
        Path(script_path).unlink(missing_ok=True)


def _heuristic_score(code: str) -> float:
    """Fallback scoring when no test cases available."""
    score = 0.0
    if "def " in code:
        score += 0.2
    if "return " in code:
        score += 0.1
    if "import " in code:
        score += 0.1
    if len(code) > 50:
        score += 0.1
    return min(score, 0.5)


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM output."""
    # Try ```python ... ``` block
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ``` block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If text starts with import/def/class, treat as code
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ")):
        return stripped

    return stripped


# ---------------------------------------------------------------------------
# TRL GRPOTrainer approach
# ---------------------------------------------------------------------------

def train_grpo_trl(
    dataset: Dataset,
    reward_fn,
    config,
    output_name: str = "grpo_dora",
) -> str:
    """Train DoRA adapter via GRPO using TRL's GRPOTrainer.

    Args:
        dataset: Dataset with "prompt" column.
        reward_fn: Callable(completions, prompts) -> list[float].
        config: ConsolidationConfig.
        output_name: Subdirectory name under adapters_dir.

    Returns:
        Path to saved adapter.
    """
    from trl import GRPOTrainer, GRPOConfig

    output_dir = str(Path(config.adapters_dir) / output_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.grpo_lora_rank,
        lora_alpha=config.grpo_lora_alpha,
        lora_dropout=config.grpo_lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=config.grpo_use_dora,
    )

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=config.grpo_group_size,
        max_completion_length=config.grpo_max_completion_length,
        temperature=config.grpo_temperature,
        beta=config.grpo_beta,
        learning_rate=config.grpo_lr,
        num_train_epochs=config.grpo_epochs,
        max_steps=config.grpo_max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        seed=config.seed,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        peft_config=lora_config,
    )

    print(f"  Training GRPO DoRA (G={config.grpo_group_size}, "
          f"lr={config.grpo_lr}, beta={config.grpo_beta})...")

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log reward progression
    rewards = [
        entry.get("reward", 0)
        for entry in trainer.state.log_history
        if "reward" in entry
    ]
    if rewards:
        print(f"  Reward: start={rewards[0]:.3f} -> end={rewards[-1]:.3f}")

    _save_grpo_log(output_dir, config, dataset, rewards)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ---------------------------------------------------------------------------
# Manual GRPO fallback
# ---------------------------------------------------------------------------

def train_grpo_manual(
    dataset: Dataset,
    reward_fn,
    config,
    output_name: str = "grpo_dora",
) -> str:
    """Manual GRPO implementation — fallback if TRL's GRPOTrainer fails.

    GRPO algorithm:
    1. Generate G completions per prompt
    2. Score each with reward_fn
    3. Compute advantage = (reward - mean) / std  (group-relative)
    4. Weight log-prob loss by advantage
    5. Gradient step with KL penalty
    """
    from peft import get_peft_model

    output_dir = str(Path(config.adapters_dir) / output_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.grpo_lora_rank,
        lora_alpha=config.grpo_lora_alpha,
        lora_dropout=config.grpo_lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=config.grpo_use_dora,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.grpo_lr, weight_decay=0.01
    )

    G = config.grpo_group_size
    all_rewards = []

    print(f"  Manual GRPO training (G={G}, max_steps={config.grpo_max_steps})...")

    model.train()
    step = 0
    for epoch in range(config.grpo_epochs):
        for i, row in enumerate(dataset):
            if step >= config.grpo_max_steps:
                break

            prompt = row["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # 1. Generate G completions
            completions = []
            with torch.no_grad():
                for _ in range(G):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config.grpo_max_completion_length,
                        temperature=config.grpo_temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    completions.append(text)

            # 2. Score completions
            rewards = reward_fn(completions, prompts=[prompt] * G)
            rewards_t = torch.tensor(rewards, dtype=torch.float32)
            all_rewards.extend(rewards)

            # 3. Group-relative advantage
            mean_r = rewards_t.mean()
            std_r = rewards_t.std() + 1e-8
            advantages = (rewards_t - mean_r) / std_r

            # 4. Weighted loss — only update on non-trivial advantages
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
            n_updates = 0

            for comp, adv in zip(completions, advantages):
                if abs(adv.item()) < 0.01:
                    continue

                comp_ids = tokenizer(
                    prompt + comp, return_tensors="pt", truncation=True,
                    max_length=config.grpo_max_completion_length + 256,
                ).to(model.device)

                outputs = model(**comp_ids, labels=comp_ids["input_ids"])
                nll = outputs.loss

                # GRPO loss: -advantage * log_prob
                loss = -adv.to(model.device) * (-nll)
                total_loss = total_loss + loss
                n_updates += 1

            if n_updates > 0:
                avg_loss = total_loss / n_updates
                # KL penalty (simplified: regularize toward low loss)
                final_loss = avg_loss + config.grpo_beta * avg_loss.abs()
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            step += 1

            if step % 5 == 0:
                recent = all_rewards[-G * 5:] if len(all_rewards) >= G * 5 else all_rewards
                avg_r = sum(recent) / len(recent) if recent else 0
                print(f"    Step {step}/{config.grpo_max_steps}: "
                      f"mean_reward={avg_r:.3f}, group_rewards={rewards}")

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    _save_grpo_log(output_dir, config, dataset, all_rewards)

    print(f"  Manual GRPO saved to {output_dir}")
    if all_rewards:
        early = all_rewards[:G * 3]
        late = all_rewards[-G * 3:]
        print(f"  Reward: early={sum(early)/len(early):.3f} -> "
              f"late={sum(late)/len(late):.3f}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir


# ---------------------------------------------------------------------------
# Entry point — tries TRL first, falls back to manual
# ---------------------------------------------------------------------------

def train_grpo(
    dataset: Dataset,
    reward_fn,
    config,
    output_name: str = "grpo_dora",
) -> str:
    """Train GRPO adapter. Tries TRL GRPOTrainer, falls back to manual.

    Args:
        dataset: Dataset with "prompt" column and task metadata.
        reward_fn: Callable(completions, prompts) -> list[float].
        config: ConsolidationConfig.
        output_name: Subdirectory name.

    Returns:
        Path to saved adapter directory.
    """
    try:
        return train_grpo_trl(dataset, reward_fn, config, output_name)
    except Exception as e:
        print(f"  TRL GRPOTrainer failed: {e}")
        print(f"  Falling back to manual GRPO implementation...")
        return train_grpo_manual(dataset, reward_fn, config, output_name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_grpo_log(
    output_dir: str,
    config,
    dataset: Dataset,
    rewards: list[float],
) -> None:
    n_middle = sum(1 for row in dataset if row.get("zone") == "middle")
    n_anchor = sum(1 for row in dataset if row.get("zone") == "anchor")

    with open(f"{output_dir}/training_log.json", "w") as f:
        json.dump({
            "method": "grpo_dora",
            "base_model": config.base_model,
            "middle_q_episodes": n_middle,
            "anchor_episodes": n_anchor,
            "group_size": config.grpo_group_size,
            "lr": config.grpo_lr,
            "beta": config.grpo_beta,
            "max_steps": config.grpo_max_steps,
            "use_dora": config.grpo_use_dora,
            "reward_summary": {
                "count": len(rewards),
                "mean": sum(rewards) / len(rewards) if rewards else 0,
                "min": min(rewards) if rewards else 0,
                "max": max(rewards) if rewards else 0,
            },
        }, f, indent=2)
