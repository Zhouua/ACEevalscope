#!/usr/bin/env python3
"""Run the sample questions with a direct LLM baseline and generate a report."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import List

from transformers import AutoTokenizer, pipeline  # type: ignore[import-untyped]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/data/models/openai/gpt-oss-20b",
        help="Local directory that contains the model weights.",
    )
    parser.add_argument(
        "--questions",
        default="questions.json",
        help="Path to the questions JSON file.",
    )
    parser.add_argument(
        "--output",
        default="reports/questions_report.md",
        help="Path to write the markdown report.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="2,3",
        help="Comma-separated CUDA device ids to expose (default: 2,3).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Maximum number of tokens for generation.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_answer(text: str) -> str:
    trimmed = text.strip()
    marker = "assistantfinal"
    if marker in trimmed:
        trimmed = trimmed.split(marker, 1)[1].strip()
    if trimmed.startswith(marker):
        trimmed = trimmed[len(marker) :].strip()
    return trimmed


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def truncate(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    questions_path = Path(args.questions)
    data = load_questions(questions_path)
    print(f"Loaded {len(data)} questions from {questions_path}.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    text_pipe = pipeline(
        "text-generation",
        model=args.model_path,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="bfloat16",
        return_full_text=False,
    )

    prompt_template = (
        "你是一名资深火灾调查鉴定专家，回答要详尽、条理清晰、可执行。\n"
        "Reasoning: low\n\n"
        "问题：{question}\n"
        "请给出分步骤、可操作的专业答复。"
    )

    results = []
    for idx, item in enumerate(data, start=1):
        question = item["question"]
        reference = item["answer"]
        messages = [
            {
                "role": "system",
                "content": "你是专业火灾调查顾问，回答必须准确且结构化。",
            },
            {"role": "user", "content": prompt_template.format(question=question)},
        ]
        print(f"[{idx:02d}/{len(data)}] Generating answer...")
        outputs = text_pipe(messages, max_new_tokens=args.max_new_tokens)
        raw_text = outputs[0]["generated_text"]
        final_answer = extract_answer(raw_text)
        score = similarity(final_answer, reference)
        results.append(
            {
                "id": f"q{idx:02d}",
                "question": question,
                "prediction": final_answer,
                "reference": reference,
                "similarity": score,
            }
        )

    scores = [item["similarity"] for item in results]
    avg_score = mean(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = []
    lines.append("# Questions Test Report (Direct Mode)")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Model: `{args.model_path}`")
    lines.append(f"- CUDA devices: `{args.cuda_visible_devices}`")
    lines.append(f"- Max new tokens: {args.max_new_tokens}")
    lines.append(f"- Samples: {len(results)}")
    lines.append(
        f"- Similarity (avg/min/max): {avg_score:.2%} / {min_score:.2%} / {max_score:.2%}"
    )
    lines.append("")
    lines.append("## Per-Question Summary")
    lines.append("")
    lines.append("| # | Similarity | Question | Prediction (truncated) |")
    lines.append("|---|------------|----------|------------------------|")
    for item in results:
        lines.append(
            f"| {item['id']} | {item['similarity']:.2%} | "
            f"{truncate(item['question'])} | {truncate(item['prediction'])} |"
        )
    lines.append("")
    lines.append("## Detailed Responses")
    lines.append("")
    for item in results:
        lines.append(f"### {item['id']} — Similarity {item['similarity']:.2%}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(item["question"])
        lines.append("")
        lines.append("**Model Prediction**")
        lines.append("")
        lines.append(item["prediction"])
        lines.append("")
        lines.append("**Reference Answer**")
        lines.append("")
        lines.append(item["reference"])
        lines.append("")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
