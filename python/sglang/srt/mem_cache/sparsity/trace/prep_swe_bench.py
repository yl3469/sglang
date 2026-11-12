"""Stage 0: SWE-bench -> bench_serving ``custom`` JSONL.

Emits the ``custom`` dataset schema consumed by
``benchmark/datasets/custom.py``::

    {"conversations": [{"role": "user", "content": <prompt>},
                       {"role": "assistant", "content": <answer>}]}

(The loader also accepts ``value`` keys; we use ``content``.)

SWE-bench prompts are long-context by nature (issue text + retrieved code),
which is exactly the regime HiSparse targets. We build each prompt from the
issue statement plus any provided text context and filter to a configurable
token window so the collected decode trace exercises a realistic context
length. Token counting uses the served model's tokenizer when available,
falling back to a cheap word-count proxy so the prep step has no hard
dependency on the model weights.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Optional, Tuple

__all__ = ["build_prompt", "estimate_tokens", "iter_swe_examples", "write_jsonl"]


# A compact instruction wrapper so the model produces a patch-style answer,
# mirroring the SWE-bench task framing used in the reference offloading runs.
_INSTRUCTION = (
    "You are a software engineer. Read the following GitHub issue and the "
    "referenced repository context, then propose a concrete code change that "
    "resolves the issue. Explain your reasoning, then provide the patch.\n\n"
)


def build_prompt(example: dict) -> str:
    """Assemble a single long-context user prompt from a SWE-bench example.

    Uses the fields commonly present across SWE-bench variants
    (``problem_statement``, ``text``, ``hints_text``, ``patch`` context). Any
    missing field is skipped.
    """
    parts: List[str] = [_INSTRUCTION]
    repo = example.get("repo")
    if repo:
        parts.append(f"Repository: {repo}\n")
    instance = example.get("instance_id")
    if instance:
        parts.append(f"Instance: {instance}\n")
    # Prefer an already-assembled `text` field (SWE-bench "oracle"/"BM25"
    # retrieval variants ship one); otherwise fall back to the raw statement.
    if example.get("text"):
        parts.append(str(example["text"]))
    else:
        if example.get("problem_statement"):
            parts.append("Issue:\n" + str(example["problem_statement"]) + "\n")
        if example.get("hints_text"):
            parts.append("Hints:\n" + str(example["hints_text"]) + "\n")
    return "\n".join(parts).strip()


def _reference_answer(example: dict) -> str:
    """A short reference completion (the gold patch, if present).

    The custom loader only needs a non-empty second turn; the assistant content
    is not actually sent as input (bench_serving generates it). We store the
    gold patch when available for provenance.
    """
    return str(example.get("patch") or example.get("gold_patch") or "(patch)")


def estimate_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    # Cheap proxy: ~0.75 tokens/word is typical for code+prose; use word count
    # as a conservative lower bound (over-counts slightly on code punctuation).
    return len(text.split())


def _load_tokenizer(model_path: Optional[str]):
    if not model_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - optional dependency/path
        print(f"tokenizer load failed ({exc}); using word-count proxy")
        return None


def iter_swe_examples(
    dataset_name: str,
    split: str,
    hf_cache_dir: Optional[str] = None,
) -> Iterable[dict]:
    """Yield raw SWE-bench examples from a HF dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, cache_dir=hf_cache_dir)
    for row in ds:
        yield dict(row)


def _select(
    examples: Iterable[dict],
    num_prompts: int,
    min_tokens: int,
    max_tokens: int,
    tokenizer,
) -> List[Tuple[str, str]]:
    selected: List[Tuple[str, str]] = []
    for ex in examples:
        prompt = build_prompt(ex)
        n = estimate_tokens(prompt, tokenizer)
        if n < min_tokens or (max_tokens > 0 and n > max_tokens):
            continue
        selected.append((prompt, _reference_answer(ex)))
        if len(selected) >= num_prompts:
            break
    return selected


def write_jsonl(path: str, rows: List[Tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for user, assistant in rows:
            record = {
                "conversations": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default="princeton-nlp/SWE-bench_oracle",
        help="HF dataset id. Variants shipping a pre-assembled `text` field "
        "(oracle/BM25) give the longest, most realistic context.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=20000,
        help="Drop prompts shorter than this (long-context focus).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120000,
        help="Drop prompts longer than this (0 = no cap).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Model path/id for accurate token counting (optional).",
    )
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    args = parser.parse_args(argv)

    tokenizer = _load_tokenizer(args.tokenizer_path)
    examples = iter_swe_examples(
        args.dataset_name, args.split, hf_cache_dir=args.hf_cache_dir
    )
    rows = _select(
        examples,
        num_prompts=args.num_prompts,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        tokenizer=tokenizer,
    )
    if not rows:
        raise SystemExit(
            "no SWE-bench prompts matched the token window; widen "
            "--min-tokens/--max-tokens or check the dataset variant"
        )
    write_jsonl(args.out, rows)
    lens = [estimate_tokens(u, tokenizer) for u, _ in rows]
    print(
        f"Wrote {len(rows)} prompts to {args.out} "
        f"(token estimate min={min(lens)} max={max(lens)} "
        f"mean={sum(lens) // len(lens)})"
    )


if __name__ == "__main__":
    main()
