"""Score and rank code candidates using the verifier DoRA adapter.

The verifier was trained via DPO to prefer high-Q code over low-Q code.
At inference time, we compute log-probability as a proxy for quality:
    reward(prompt, code) = log P_verifier(code|prompt) - log P_base(code|prompt)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_verifier(base_model_name: str, adapter_path: str, device: str = "cuda:0"):
    """Load base model + verifier DoRA adapter."""
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def score_candidate(
    model, tokenizer, prompt: str, code: str,
) -> float:
    """Score a candidate solution using log-probability.

    Higher log-prob = verifier thinks this is better code.
    """
    full_text = f"### Instruction:\n{prompt}\n\n### Response:\n{code}"
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Compute log-prob of response tokens only (skip prompt)
    prompt_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    prompt_len = len(tokenizer(prompt_text)["input_ids"])

    if prompt_len >= logits.shape[1]:
        return -100.0  # degenerate case

    response_logits = logits[0, prompt_len - 1:-1]
    response_tokens = inputs["input_ids"][0, prompt_len:]

    if response_tokens.shape[0] == 0:
        return -100.0

    log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(
        1, response_tokens.unsqueeze(1)
    ).squeeze()

    return float(token_log_probs.mean())


def rank_candidates(
    model, tokenizer, prompt: str, candidates: list[str],
) -> list[tuple[str, float]]:
    """Rank candidates by verifier score (highest first)."""
    scored = []
    for code in candidates:
        score = score_candidate(model, tokenizer, prompt, code)
        scored.append((code, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
