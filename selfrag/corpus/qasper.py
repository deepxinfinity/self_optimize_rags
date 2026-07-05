"""Ingest QASPER (questions over NLP papers) into a corpus + real held-out questions.

Parses the official AllenAI QASPER JSON directly (the HF loading-script route is deprecated in
datasets 4.x). Each paper -> one document (title + abstract + full text); its answerable human
questions -> the real held-out set (gold_doc_id = the paper). Papers that contribute no answerable
question are still written as corpus distractors, which is what makes retrieval non-trivial.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional


def paper_to_text(paper: dict[str, Any]) -> str:
    parts = [f"# {paper['title']}".strip(), "", "## Abstract", (paper.get("abstract") or "").strip()]
    for sec in paper.get("full_text", []):
        paras = [p for p in sec.get("paragraphs", []) if p and p.strip()]
        if not paras:
            continue
        name = (sec.get("section_name") or "").strip()
        if name:
            parts.append(f"## {name}")
        parts.extend(paras)
    return "\n\n".join(parts).strip()


def _answer_text(answer_entry: dict[str, Any]) -> Optional[str]:
    a = answer_entry.get("answer", {})
    if a.get("unanswerable"):
        return None
    if a.get("yes_no") is not None:
        return "Yes" if a["yes_no"] else "No"
    if a.get("free_form_answer"):
        return a["free_form_answer"].strip()
    spans = a.get("extractive_spans")
    if spans:
        return "; ".join(spans).strip()
    return None


def _paper_questions(pid: str, paper: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    for qa in paper.get("qas", []):
        question = (qa.get("question") or "").strip()
        if not question:
            continue
        answer = next((t for ans in qa.get("answers", []) if (t := _answer_text(ans))), None)
        if answer:
            out.append({"question": question, "answer": answer, "gold_doc_id": f"{pid}.txt"})
    return out


def ingest_qasper(source: str | Path, corpus_dir: str | Path, real_out: str | Path,
                  n_papers: int = 50, n_questions: int = 100, seed: int = 42,
                  min_words: int = 200) -> dict[str, int]:
    data = json.loads(Path(source).read_text())
    pids = list(data.keys())
    random.Random(seed).shuffle(pids)

    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    questions: list[dict[str, str]] = []
    for pid in pids:
        if written >= n_papers:
            break
        paper = data[pid]
        text = paper_to_text(paper)
        if len(text.split()) < min_words:
            continue
        paper_qs = _paper_questions(pid, paper)
        if not paper_qs:
            continue  # keep papers that contribute >=1 real question (still adds distractors)
        (corpus_dir / f"{pid}.txt").write_text(text, encoding="utf-8")
        written += 1
        questions.extend(paper_qs)

    random.Random(seed + 1).shuffle(questions)
    questions = questions[:n_questions]

    real_out = Path(real_out)
    real_out.parent.mkdir(parents=True, exist_ok=True)
    with real_out.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return {"papers": written, "questions": len(questions),
            "gold_papers_covered": len({q["gold_doc_id"] for q in questions})}
