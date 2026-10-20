from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pytest

from selfrag.corpus.loader import parse_file
from selfrag.web.ingest import (
    IngestError,
    IngestLimits,
    ingest_bytes,
    iter_zip_safely,
    parse_hf_spec,
    sanitize_filename,
    validate_public_url,
)
from selfrag.web.projects import ProjectStore
from selfrag.web.report import corpus_report

CONFIG = "configs/default.yaml"
LIMITS = IngestLimits(max_file_bytes=1024 * 1024, max_project_bytes=4 * 1024 * 1024)


def test_sanitize_strips_directories_and_bad_chars():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("..\\..\\win\\evil.md") == "evil.md"
    assert sanitize_filename("weird name$$.txt") == "weird_name_.txt"
    assert sanitize_filename(".hidden.txt") == "hidden.txt"


def test_disallowed_extension_rejected(tmp_path):
    with pytest.raises(IngestError, match="not allowed"):
        ingest_bytes(tmp_path, "malware.exe", b"MZ...", LIMITS)
    with pytest.raises(IngestError, match="not allowed"):
        ingest_bytes(tmp_path, "script.py", b"import os", LIMITS)


def test_per_file_size_cap(tmp_path):
    with pytest.raises(IngestError, match="per-file cap"):
        ingest_bytes(tmp_path, "big.txt", b"x" * (LIMITS.max_file_bytes + 1), LIMITS)


def test_magic_mismatch_rejected(tmp_path):
    with pytest.raises(IngestError, match="does not match"):
        ingest_bytes(tmp_path, "fake.pdf", b"just text, not a pdf", LIMITS)


def test_txt_ingest_normalizes_to_corpus(tmp_path):
    entry = ingest_bytes(tmp_path, "notes.txt", b"hello corpus world " * 20, LIMITS)
    assert entry["stored"] == "notes.txt"
    assert (tmp_path / "corpus" / "notes.txt").exists()
    # same name again -> deduped target, no overwrite
    entry2 = ingest_bytes(tmp_path, "notes.txt", b"other content " * 20, LIMITS)
    assert entry2["stored"] == "notes-1.txt"


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_zip_path_traversal_rejected():
    data = _zip_bytes({"../../evil.txt": b"pwned"})
    with pytest.raises(IngestError, match="escapes"):
        list(iter_zip_safely(data, LIMITS))


def test_zip_absolute_path_rejected():
    data = _zip_bytes({"/etc/evil.txt": b"pwned"})
    with pytest.raises(IngestError, match="escapes"):
        list(iter_zip_safely(data, LIMITS))


def test_zip_bomb_ratio_rejected():
    # highly compressible entry: 10 MB of zeros -> tiny compressed size
    data = _zip_bytes({"bomb.txt": b"\0" * (10 * 1024 * 1024)})
    with pytest.raises(IngestError, match="ratio|size cap"):
        list(iter_zip_safely(data, LIMITS))


def test_zip_skips_disallowed_and_yields_allowed():
    data = _zip_bytes({"ok.txt": b"fine content", "skip.exe": b"MZ", "also_ok.md": b"# hi"})
    names = [n for n, _ in iter_zip_safely(data, LIMITS)]
    assert set(names) == {"ok.txt", "also_ok.md"}


def test_zip_entry_cap():
    data = _zip_bytes({f"f{i}.txt": b"x" for i in range(1001)})
    with pytest.raises(IngestError, match="entries"):
        list(iter_zip_safely(data, LIMITS))


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x.zip",
    "http://localhost:8123/x.zip",
    "http://10.0.0.5/internal.zip",
    "http://192.168.1.1/router.zip",
    "http://169.254.169.254/latest/meta-data",   # cloud metadata endpoint
    "http://[::1]/x.zip",
    "ftp://example.com/x.zip",
    "file:///etc/passwd",
])
def test_ssrf_blocked(url):
    with pytest.raises(IngestError):
        validate_public_url(url)


def test_parse_html_stdlib(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><head><script>evil()</script><style>x{}</style></head>"
                 "<body><h1>Title</h1><p>Real content here.</p></body></html>")
    text = parse_file(p)
    assert "Real content here." in text
    assert "evil" not in text


def test_parse_csv_rowwise(tmp_path):
    pytest.importorskip("pandas")
    p = tmp_path / "table.csv"
    p.write_text("name,role\nada,engineer\ngrace,admiral\n")
    text = parse_file(p)
    assert "name: ada | role: engineer" in text


def test_parse_docx(tmp_path):
    docx = pytest.importorskip("docx")
    p = tmp_path / "doc.docx"
    d = docx.Document()
    d.add_paragraph("Hello from a Word document.")
    d.save(str(p))
    assert "Hello from a Word document." in parse_file(p)


def test_hf_spec_parsing():
    assert parse_hf_spec("hf://squad") == ("squad", None, None)
    assert parse_hf_spec("hf://user/name:cfg:train") == ("user/name", "cfg", "train")
    assert parse_hf_spec("https://huggingface.co/datasets/squad") == ("squad", None, None)
    with pytest.raises(IngestError):
        parse_hf_spec("hf://")


def test_corpus_report(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("word " * 500)
    (corpus / "b.txt").write_text("word " * 500)      # duplicate of a
    (corpus / "tiny.txt").write_text("too short")
    rep = corpus_report(tmp_path)
    assert rep["n_docs"] == 3
    assert rep["duplicate_groups"] == [["a.txt", "b.txt"]]
    assert any("tiny.txt" in w for w in rep["warnings"])
    assert rep["estimates"]["est_chunks"] >= 3


def test_project_config_generation(tmp_path):
    store = ProjectStore(tmp_path)
    meta = store.create("My Docs")
    pid = meta["id"]
    assert pid == "my-docs"
    (store.path(pid) / "corpus" / "d.txt").write_text("content " * 100)

    cfg_path = store.write_config(pid, CONFIG, {
        "generator": "llama3.2:3b",
        "arm": {"id": "user_arm", "provider": "ollama", "name": "gemma2:9b"},
        "n_pairs": 40, "stage1_trials": 7,
    })
    import yaml
    raw = yaml.safe_load(cfg_path.read_text())
    assert raw["models"]["generator"]["name"] == "llama3.2:3b"
    assert [a["id"] for a in raw["models"]["eval_generators"]] == ["user_arm"]
    assert raw["dataset"]["real_questions_path"] is None
    assert raw["evalgen"]["n_pairs_per_arm"] == 40
    assert raw["optimization"]["stage1_retrieval"]["n_trials"] == 7
    assert Path(raw["dataset"]["corpus_path"]).is_absolute()

    # no arm chosen: first arm of the base config
    cfg_path = store.write_config(pid, CONFIG, {})
    raw = yaml.safe_load(cfg_path.read_text())
    assert len(raw["models"]["eval_generators"]) == 1


def test_api_project_roundtrip(tmp_path, monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.chdir(Path(__file__).parent.parent)
    from selfrag.config import load_config
    from selfrag.ui.app import create_app

    import yaml
    raw = yaml.safe_load(Path(CONFIG).read_text())
    raw.setdefault("runtime", {})["projects_dir"] = str(tmp_path / "projects")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(raw))

    c = testclient.TestClient(create_app(str(cfg_file)))

    r = c.post("/api/projects", json={"name": "API Test"})
    assert r.status_code == 201
    pid = r.json()["id"]

    r = c.post(f"/api/projects/{pid}/data",
               files=[("files", ("doc.txt", b"retrieval systems need documents " * 40))])
    assert r.status_code == 200
    assert r.json()["ingested"][0]["stored"] == "doc.txt"
    assert r.json()["report"]["n_docs"] == 1

    r = c.post(f"/api/projects/{pid}/data",
               files=[("files", ("evil.sh", b"rm -rf /")) ])
    assert r.status_code == 422

    r = c.put(f"/api/projects/{pid}/config", json={"stage1_trials": 3})
    assert r.status_code == 200
    assert r.json()["models"]["arms"] == ["synthetic"]

    # too few real questions are rejected; enough are picked up by the config
    bad = "\n".join(json.dumps({"question": f"q{i}", "gold_doc_id": "doc.txt"}) for i in range(3))
    r = c.post(f"/api/projects/{pid}/real-questions",
               files={"file": ("real.jsonl", bad.encode())})
    assert r.status_code == 422
    good = "\n".join(json.dumps({"question": f"q{i}", "gold_doc_id": "doc.txt"}) for i in range(12))
    r = c.post(f"/api/projects/{pid}/real-questions",
               files={"file": ("real.jsonl", good.encode())})
    assert r.status_code == 200
    r = c.put(f"/api/projects/{pid}/config", json={})
    assert r.json()["has_real_questions"] is True

    r = c.delete(f"/api/projects/{pid}")
    assert r.status_code == 200


def test_job_lifecycle_with_fake_command(tmp_path):
    # `sleep` stands in for the real pipeline
    from selfrag.web import jobs

    projects_root = tmp_path
    pdir = tmp_path / "p1"
    (pdir / "jobs").mkdir(parents=True)
    cfg = pdir / "config.yaml"
    cfg.write_text("fake: true")

    rec = jobs.start_job(projects_root, pdir, cfg, cmd=["sleep", "30"])
    assert rec["status"] == "running"

    with pytest.raises(jobs.JobError, match="running job"):
        jobs.start_job(projects_root, pdir, cfg, cmd=["sleep", "30"])

    out = jobs.cancel_job(pdir, rec["job_id"])
    assert out["status"] in ("cancelling", "cancelled")
    for _ in range(50):
        time.sleep(0.1)
        if jobs.get_job(pdir, rec["job_id"])["status"] == "cancelled":
            break
    assert jobs.get_job(pdir, rec["job_id"])["status"] == "cancelled"


def test_failed_job_surfaces_log_tail(tmp_path):
    from selfrag.web import jobs

    pdir = tmp_path / "p2"
    (pdir / "jobs").mkdir(parents=True)
    cfg = pdir / "config.yaml"
    cfg.write_text("fake: true")

    rec = jobs.start_job(tmp_path, pdir, cfg,
                         cmd=["python3", "-c", "import sys; print('boom traceback'); sys.exit(2)"])
    for _ in range(50):
        time.sleep(0.1)
        status = jobs.get_job(pdir, rec["job_id"])
        if status["status"] != "running":
            break
    assert status["status"] == "failed"
    assert any("boom traceback" in line for line in status["error_tail"])


def test_pipeline_regenerates_eval_sets_made_from_another_corpus(tmp_path, monkeypatch):
    from selfrag import cli
    generated, optimized = [], []
    monkeypatch.setattr(cli, "_require_ollama", lambda *specs: None)
    monkeypatch.setattr(cli, "_generate_evalsets", lambda cfg, arms: generated.append([a.id for a in arms]))
    monkeypatch.setattr(cli, "cmd_optimize", lambda args: optimized.append(1) or 0)
    docs, other = tmp_path / "docs", tmp_path / "other"
    docs.mkdir(); other.mkdir()
    out = tmp_path / "evalsets"
    argv = ["pipeline", "--config", CONFIG, "--set", f"evalgen.out_dir={out}",
            "--set", f"dataset.corpus_path={docs}"]

    assert cli.main(argv) == 0 and generated == [["synthetic"]]
    (out / "synthetic").mkdir(parents=True)
    manifest = out / "synthetic" / "manifest.json"
    manifest.write_text(json.dumps({"corpus_path": str(docs.resolve())}))
    assert cli.main(argv) == 0 and generated == [["synthetic"]]          # same corpus: reused
    manifest.write_text(json.dumps({"corpus_path": str(other.resolve())}))
    assert cli.main(argv) == 0 and generated == [["synthetic"], ["synthetic"]]
    assert len(optimized) == 3
