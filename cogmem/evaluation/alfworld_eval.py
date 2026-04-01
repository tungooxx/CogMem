import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LocalLoRAModel:
    def __init__(self, base_model: str, adapter_path: str | None = None):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        if adapter_path and Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_alfworld_task(
    task_description: str,
    env,
    model: LocalLoRAModel | None = None,
    llm_client=None,
    max_steps: int = 50,
) -> dict:
    """Run a single ALFWorld task. Uses either local model or LLM client.

    Returns: {"success": bool, "steps": int, "trajectory": list}
    """
    obs, info = env.reset()
    trajectory = []

    for step in range(max_steps):
        if model is not None:
            prompt = _build_prompt(task_description, obs, trajectory)
            action_text = model.generate(prompt)
        elif llm_client is not None:
            prompt = _build_prompt(task_description, obs, trajectory)
            action_text = llm_client.generate(prompt)
        else:
            raise ValueError("Either model or llm_client must be provided")

        action = _extract_action(action_text)
        obs, reward, done, info = env.step(action)
        trajectory.append({"step": step + 1, "action": action, "observation": obs})

        if done:
            return {"success": reward > 0, "steps": step + 1, "trajectory": trajectory}

    return {"success": False, "steps": max_steps, "trajectory": trajectory}


def _build_prompt(task: str, obs: str, trajectory: list[dict]) -> str:
    lines = [f"Task: {task}\n"]
    for t in trajectory[-5:]:  # last 5 steps for context window
        lines.append(f"Action: {t['action']}")
        lines.append(f"Observation: {t['observation']}")
    lines.append(f"Current observation: {obs}")
    lines.append("What is your next action?")
    return "\n".join(lines)


def _extract_action(text: str) -> str:
    text = text.strip()
    for prefix in ["Action:", "action:", "> "]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.split("\n")[0].strip()
