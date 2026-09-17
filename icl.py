from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_utils import AGNEWS_LABELS


def load_model(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    return tok, model


def _clip(text: str, n: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "..."


def build_prompt(shots: list[tuple[str, int]], query: str) -> str:
    labels = " | ".join(AGNEWS_LABELS)
    lines = [
        f"Classify the news into exactly one class: {labels}.",
        "Output only the class name.",
        "",
    ]
    for text, y in shots:
        lines.append(f"Article: {_clip(text)}")
        lines.append(f"Class: {AGNEWS_LABELS[y]}")
        lines.append("")
    lines.append(f"Article: {_clip(query)}")
    lines.append("Class:")
    return "\n".join(lines)


def parse_label(text: str) -> int | None:
    first = text.strip().splitlines()[0].strip().strip(".").strip() if text.strip() else ""
    for i, name in enumerate(AGNEWS_LABELS):
        if first.lower() == name.lower():
            return i
    lowered = text.lower()
    hits = [i for i, name in enumerate(AGNEWS_LABELS) if name.lower() in lowered]
    if len(hits) == 1:
        return hits[0]
    m = re.search(r"\b([0-3])\b", text)
    if m:
        return int(m.group(1))
    return None


@torch.no_grad()
def predict(tok, model, shots: list[tuple[str, int]], query: str, max_new_tokens: int = 8) -> int | None:
    prompt = build_prompt(shots, query)
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return parse_label(gen)
