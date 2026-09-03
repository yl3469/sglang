"""Stage 0 (AA-LCR variant): AA-LCR -> bench_serving ``custom`` JSONL.

AA-LCR (Artificial Analysis Long Context Reasoning,
``ArtificialAnalysis/AA-LCR`` on HF) is a genuine long-context benchmark:
each question references a *set* of source documents (71k-114k input tokens,
mean ~95k) and expects a short, exact answer. This is squarely HiSparse's
target regime (vs SWE-bench's ~7k-token prompts).

The dataset ships as:
  * ``AA-LCR_Dataset.csv`` -- one row per question with columns
    ``document_category, document_set_id, question_id, question, answer,
    data_source_filenames`` (``;``-separated), ``input_tokens``.
  * ``extracted_text/AA-LCR_extracted-text.zip`` -- the plain-text documents
    under ``lcr/<document_category>/<document_set_id>/<filename>.txt``.

We assemble each prompt as [instruction + concatenated source documents +
question] and emit the ``custom`` schema consumed by
``benchmark/datasets/custom.py``::

    {"conversations": [{"role": "user", "content": <prompt>},
                       {"role": "assistant", "content": <gold answer>}]}

The gold answer is stored as the assistant turn so that -- exactly as with
``prep_swe_bench.py`` -- a run that OMITS ``--sharegpt-output-len`` will take
each request's output length from the tokenized answer (custom.py:125-127).
AA-LCR answers are short, so for a decode-heavy trace you will usually pass an
explicit ``--sharegpt-output-len``; the point of this dataset is the *input*
(prefill / KV-context) length, which is where HiSparse's offload matters.

Token counting uses the served model's tokenizer when available, else a
word-count proxy, so the prep step has no hard dependency on model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from typing import Dict, List, Optional, Tuple

__all__ = [
    "build_prompt",
    "estimate_tokens",
    "load_documents",
    "iter_aa_lcr_examples",
    "write_jsonl",
]


_INSTRUCTION = (
    "You are given a set of source documents. Read them carefully, then answer "
    "the question at the end using only information supported by the documents. "
    "Be precise and concise.\n\n"
)


def estimate_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    # Cheap proxy (~0.75 tok/word for prose); word count is a conservative
    # lower bound on token count.
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


def load_documents(zip_path: str) -> Dict[str, str]:
    """Map every ``.txt`` document to its text, keyed by both ``basename`` and
    ``document_set_id/basename`` so lookups are unambiguous across sets."""
    docs: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt"):
                continue
            try:
                text = zf.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue
            base = os.path.basename(name)
            parts = name.split("/")
            docs[base] = text
            if len(parts) >= 3:
                set_id = parts[-2]
                docs[f"{set_id}/{base}"] = text
    return docs


def build_prompt(row: dict, docs: Dict[str, str]) -> Tuple[str, int]:
    """Assemble the long-context prompt for one AA-LCR question.

    Returns ``(prompt, n_docs_found)``. Documents are joined in the order
    listed in ``data_source_filenames``; each is prefixed with its filename as
    a lightweight section header.
    """
    set_id = row.get("document_set_id", "")
    filenames = [
        f.strip() for f in row.get("data_source_filenames", "").split(";") if f.strip()
    ]
    parts: List[str] = [_INSTRUCTION]
    found = 0
    for fn in filenames:
        text = docs.get(f"{set_id}/{fn}") or docs.get(fn)
        if text is None:
            continue
        found += 1
        parts.append(f"===== DOCUMENT: {fn} =====\n{text}\n")
    parts.append("===== QUESTION =====\n" + str(row.get("question", "")).strip())
    return "\n".join(parts).strip(), found


def _gold_answer(row: dict) -> str:
    return str(row.get("answer") or "(answer)")


def iter_aa_lcr_examples(csv_path: str) -> List[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _select(
    rows: List[dict],
    docs: Dict[str, str],
    num_prompts: int,
    min_tokens: int,
    max_tokens: int,
    tokenizer,
) -> List[Tuple[str, str]]:
    selected: List[Tuple[str, str]] = []
    skipped_missing = 0
    for row in rows:
        prompt, found = build_prompt(row, docs)
        expected = len(
            [f for f in row.get("data_source_filenames", "").split(";") if f.strip()]
        )
        if found == 0 or found < expected:
            # Partial/empty document set -> the prompt would silently misrepresent
            # the context length; skip it rather than bias the trace.
            skipped_missing += 1
            if found == 0:
                continue
        n = estimate_tokens(prompt, tokenizer)
        if n < min_tokens or (max_tokens > 0 and n > max_tokens):
            continue
        selected.append((prompt, _gold_answer(row)))
        if len(selected) >= num_prompts:
            break
    if skipped_missing:
        print(f"note: {skipped_missing} rows had missing/partial documents")
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
        "--csv",
        required=True,
        help="Path to AA-LCR_Dataset.csv (from ArtificialAnalysis/AA-LCR).",
    )
    parser.add_argument(
        "--docs-zip",
        required=True,
        help="Path to extracted_text/AA-LCR_extracted-text.zip.",
    )
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
        default=200000,
        help="Drop prompts longer than this (0 = no cap). AA-LCR tops out "
        "~115k input tokens; keep this above your served max context.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Model path/id for accurate token counting (optional).",
    )
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    if not os.path.isfile(args.docs_zip):
        raise SystemExit(f"docs zip not found: {args.docs_zip}")

    tokenizer = _load_tokenizer(args.tokenizer_path)
    docs = load_documents(args.docs_zip)
    print(f"loaded {len(docs)} document keys from {os.path.basename(args.docs_zip)}")
    rows = iter_aa_lcr_examples(args.csv)
    selected = _select(
        rows,
        docs,
        num_prompts=args.num_prompts,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        tokenizer=tokenizer,
    )
    if not selected:
        raise SystemExit(
            "no AA-LCR prompts matched the token window; widen "
            "--min-tokens/--max-tokens or check the docs zip path"
        )
    write_jsonl(args.out, selected)
    lens = [estimate_tokens(u, tokenizer) for u, _ in selected]
    print(
        f"Wrote {len(selected)} prompts to {args.out} "
        f"(token estimate min={min(lens)} max={max(lens)} "
        f"mean={sum(lens) // len(lens)})"
    )


if __name__ == "__main__":
    main()
