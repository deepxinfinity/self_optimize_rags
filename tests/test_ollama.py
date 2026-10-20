from __future__ import annotations

import httpx
import pytest

from selfrag import cli
from selfrag.models import ollama_adapter
from selfrag.models.ollama_adapter import OllamaLLM, OllamaUnavailable, check_ollama


class _Client:
    def __init__(self, error=None, models=("qwen2.5:7b", "llama3.1:8b")):
        self.error = error
        self.models = models

    def chat(self, **kwargs):
        if self.error:
            raise self.error
        return {"message": {"content": "hi"}}

    def list(self):
        if self.error:
            raise self.error
        return {"models": [{"model": m} for m in self.models]}


def test_generate_timeout_becomes_a_readable_error(monkeypatch):
    monkeypatch.setattr(ollama_adapter, "_server_answers", lambda: False)
    llm = OllamaLLM("qwen2.5:7b", timeout=1)
    llm._client = _Client(error=httpx.ReadTimeout("slow"))
    with pytest.raises(OllamaUnavailable, match="suspended"):
        llm.generate("q")


def test_generate_connection_refused_becomes_a_readable_error():
    llm = OllamaLLM("qwen2.5:7b")
    llm._client = _Client(error=ConnectionError("refused"))
    with pytest.raises(OllamaUnavailable, match="ollama serve"):
        llm.generate("q")


def test_num_ctx_and_max_tokens_reach_the_options():
    llm = OllamaLLM("m", max_tokens=64, num_ctx=8192)
    assert llm._options(seed=3) == {"temperature": 0.0, "num_predict": 64, "num_ctx": 8192,
                                    "seed": 3}


def test_check_ollama_reports_missing_models(monkeypatch):
    monkeypatch.setattr(ollama_adapter.ollama, "Client", lambda **kw: _Client())
    check_ollama(["qwen2.5:7b"])
    with pytest.raises(OllamaUnavailable, match="gemma2:9b"):
        check_ollama(["qwen2.5:7b", "gemma2:9b"])


def test_check_ollama_accepts_the_latest_tag(monkeypatch):
    monkeypatch.setattr(ollama_adapter.ollama, "Client",
                        lambda **kw: _Client(models=("mxbai-embed-large:latest",)))
    check_ollama(["mxbai-embed-large"])


def test_check_ollama_when_the_server_does_not_answer(monkeypatch):
    monkeypatch.setattr(ollama_adapter.ollama, "Client",
                        lambda **kw: _Client(error=httpx.ConnectTimeout("stopped")))
    with pytest.raises(OllamaUnavailable, match="Ctrl\\+Z"):
        check_ollama(["qwen2.5:7b"])


def test_cli_prints_the_problem_instead_of_a_traceback(monkeypatch, capsys):
    def down(names):
        raise OllamaUnavailable("ollama is not answering")
    monkeypatch.setattr(cli, "check_ollama", down)
    assert cli.main(["gen-eval", "--config", "configs/qasper.yaml"]) == 2
    assert "ollama is not answering" in capsys.readouterr().err


def test_cli_skips_the_check_for_api_models(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "check_ollama", seen.append)
    cli._require_ollama({"provider": "api", "name": "claude-haiku-4-5"},
                        {"provider": "ollama", "name": "qwen2.5:7b"}, None)
    assert seen == [["qwen2.5:7b"]]


def test_timeout_with_a_live_server_points_at_a_runaway_model(monkeypatch):
    monkeypatch.setattr(ollama_adapter, "_server_answers", lambda: True)
    llm = OllamaLLM("llama3.2:3b", timeout=1)
    llm._client = _Client(error=httpx.ReadTimeout("slow"))
    with pytest.raises(OllamaUnavailable, match="max_tokens"):
        llm.generate("q")


def test_every_ollama_model_in_the_study_config_has_a_token_cap():
    from selfrag.config import load_config
    cfg = load_config("configs/qasper.yaml")
    specs = [cfg.models.generator, cfg.models.judge, *cfg.models.eval_generators]
    capped = [s.model_dump().get("max_tokens") for s in specs]
    capped.append(cfg.optimization["stage2_prompts"]["proposer"]["max_tokens"])
    assert all(capped), capped


def test_repeat_limit_abort_becomes_a_model_output_error():
    from selfrag.models.ollama_adapter import ModelOutputError
    llm = OllamaLLM("gemma2:9b")
    llm._client = _Client(error=ollama_adapter.ollama.ResponseError(
        "prediction aborted, token repeat limit reached", 500))
    with pytest.raises(ModelOutputError):
        llm.generate("q")


def test_other_server_errors_still_raise():
    llm = OllamaLLM("gemma2:9b")
    llm._client = _Client(error=ollama_adapter.ollama.ResponseError(
        "model requires more system memory", 500))
    with pytest.raises(ollama_adapter.ollama.ResponseError):
        llm.generate("q")


class _Aborting:
    name = "loop"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system=None, **kw):
        from selfrag.models.ollama_adapter import ModelOutputError
        self.calls += 1
        raise ModelOutputError("loop: prediction aborted")


def test_judge_records_a_cut_off_reply_as_missing(tmp_path):
    from selfrag.metrics.judged import JudgedEvaluator
    from selfrag.models.judge import LLMJudge
    judge = LLMJudge(_Aborting(), "loop", cache_dir=str(tmp_path))
    assert judge.score("faithfulness", "q", "a", context="c") is None

    class Mixed:
        def score(self, metric, question, answer, context="", reference=""):
            return None if question == "q0" else 1.0

    summary, items = JudgedEvaluator(Mixed(), ("faithfulness",)).evaluate(
        [{"question": f"q{i}", "answer": "a", "context": "c"} for i in range(3)])
    assert summary["faithfulness"] == 1.0 and summary["faithfulness_n"] == 2
    assert items[0]["faithfulness"] is None


def test_answerer_treats_a_cut_off_reply_as_no_answer():
    from selfrag.generate.answerer import Answerer
    assert Answerer(_Aborting(), "prompt").answer("q", []) == ""
