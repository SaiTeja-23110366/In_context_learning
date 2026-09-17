from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # distributes across all available GPUs
    )
    model.eval()
    return tok, model


def _clip(text: str, n: int = 2000) -> str:
    """
    Clip text to at most n characters (was 400 — that was causing very
    poor accuracy because the LLM could not see enough context).
    2000 chars ~ 400-500 tokens, well within a 4096 context window.
    """
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def build_prompt(
    shots: list[tuple[str, int]], query: str, label_names: list[str]
) -> str:
    labels = " | ".join(label_names)
    lines = [
        f"Classify the text into exactly one class: {labels}.",
        "Output only the class name.",
        "",
    ]
    for text, y in shots:
        lines.append(f"Text: {_clip(text)}")
        lines.append(f"Class: {label_names[y]}")
        lines.append("")
    lines.append(f"Text: {_clip(query)}")
    lines.append("Class:")
    return "\n".join(lines)


def parse_label(text: str, label_names: list[str]) -> int | None:
    first = (
        text.strip().splitlines()[0].strip().strip(".").strip()
        if text.strip()
        else ""
    )
    # Exact match (case-insensitive)
    for i, name in enumerate(label_names):
        if first.lower() == name.lower():
            return i
    # Substring match (unique)
    lowered = text.lower()
    hits = [i for i, name in enumerate(label_names) if name.lower() in lowered]
    if len(hits) == 1:
        return hits[0]
    # Numeric fallback
    m = re.search(r"\b([0-9]+)\b", text)
    if m:
        v = int(m.group(1))
        if 0 <= v < len(label_names):
            return v
    return None


@torch.no_grad()
def predict(
    tok,
    model,
    shots: list[tuple[str, int]],
    query: str,
    label_names: list[str],
    max_new_tokens: int = 16,
) -> int | None:
    prompt = build_prompt(shots, query, label_names)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(
        text, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    gen = tok.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return parse_label(gen, label_names)
