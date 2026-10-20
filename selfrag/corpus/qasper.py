"""QASPER (questions over NLP papers) -> corpus + real held-out questions.

Reads the official AllenAI JSON. Each paper becomes one document; its answerable questions become
the real held-out set, with gold_doc_id = the paper and evidence = the annotated paragraphs
(table/figure evidence is dropped because figures are not in the corpus text).

QASPER questions were written by people looking at one paper ("what baseline did they use?"), so
in a pooled corpus many cannot be tied to their paper at all. `scope_with_title` prefixes the
paper title, the way a user of a multi-paper index would ask.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional


def scope_question(question: str, title: str) -> str:
    return f'In the paper "{title}": {question}'


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


def _answer_text(answer: dict[str, Any]) -> Optional[str]:
    if answer.get("unanswerable"):
        return None
    if answer.get("yes_no") is not None:
        return "Yes" if answer["yes_no"] else "No"
    if answer.get("free_form_answer"):
        return answer["free_form_answer"].strip()
    spans = answer.get("extractive_spans")
    if spans:
        return "; ".join(spans).strip()
    return None


def _text_evidence(answer: dict[str, Any]) -> list[str]:
    return [e.strip() for e in answer.get("evidence", [])
            if e.strip() and not e.startswith("FLOAT SELECTED")]


def _paper_questions(pid: str, paper: dict[str, Any],
                     scope_with_title: bool = False) -> list[dict[str, Any]]:
    title = paper["title"].strip()
    out = []
    for qa in paper.get("qas", []):
        question = (qa.get("question") or "").strip()
        if not question:
            continue
        # annotators disagree; prefer the first answer that also has text evidence
        candidates = [(t, _text_evidence(a["answer"])) for a in qa.get("answers", [])
                      if (t := _answer_text(a["answer"]))]
        if not candidates:
            continue
        answer, evidence = next(((t, ev) for t, ev in candidates if ev), candidates[0])
        rec: dict[str, Any] = {"question": question, "answer": answer,
                               "gold_doc_id": f"{pid}.txt", "evidence": evidence}
        if scope_with_title:
            rec["raw_question"] = question
            rec["question"] = scope_question(question, title)
        out.append(rec)
    return out


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ingest_qasper(source: str | Path, corpus_dir: str | Path, real_out: str | Path,
                  n_papers: int = 50, n_questions: int = 100, seed: int = 42,
                  min_words: int = 200, scope_with_title: bool = False,
                  require_evidence: bool = False,
                  unscoped_out: Optional[str | Path] = None) -> dict[str, int]:
    if unscoped_out and not scope_with_title:
        raise ValueError("unscoped_out only makes sense together with scope_with_title")
    data = json.loads(Path(source).read_text())
    pids = list(data.keys())
    random.Random(seed).shuffle(pids)

    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    questions: list[dict[str, Any]] = []
    for pid in pids:
        if written >= n_papers:
            break
        paper = data[pid]
        text = paper_to_text(paper)
        if len(text.split()) < min_words:
            continue
        paper_qs = _paper_questions(pid, paper, scope_with_title)
        if not paper_qs:
            continue
        (corpus_dir / f"{pid}.txt").write_text(text, encoding="utf-8")
        written += 1
        questions.extend(paper_qs)

    n_answerable = len(questions)
    if require_evidence:
        questions = [q for q in questions if q["evidence"]]
    random.Random(seed + 1).shuffle(questions)
    questions = questions[:n_questions]

    _write_jsonl(real_out, questions)
    if unscoped_out:
        # same questions in the same order, so ids line up and the two sets can be paired
        _write_jsonl(unscoped_out, [{**{k: v for k, v in q.items() if k != "raw_question"},
                                     "question": q["raw_question"]} for q in questions])

    return {"papers": written, "answerable": n_answerable, "questions": len(questions),
            "with_evidence": sum(1 for q in questions if q["evidence"]),
            "gold_papers_covered": len({q["gold_doc_id"] for q in questions})}
