from __future__ import annotations

from conftest import CountingEmbedder, make_factory, params

from selfrag.utils.timing import TimingLog


def test_same_chunking_builds_one_index(factory):
    factory.build(params(128, 32, top_k=3))
    factory.build(params(128, 32, top_k=7, hybrid_alpha=0.9))
    assert factory.n_indices_built == 1
    assert factory._embedder.calls == 1   # nothing retrieved yet, so no query embeddings
    assert factory.timing.count("index_cache_hit") == 1


def test_different_chunking_builds_another_index(factory):
    factory.build(params(128, 0))
    factory.build(params(128, 64))
    assert factory.n_indices_built == 2
    assert factory._embedder.calls == 2


def test_corpus_loaded_once_across_builds(factory):
    factory.build(params(128, 0))
    factory.build(params(256, 0))
    factory.build(params(128, 0))
    assert factory.timing.count("load_corpus") == 1


def test_params_are_honoured_not_shared(factory):
    a = factory.build(params(top_k=3))
    b = factory.build(params(top_k=9))
    assert a.params.top_k == 3 and b.params.top_k == 9
    assert a.chunks is b.chunks


def test_reranker_only_attached_when_requested(factory):
    assert factory.build(params(use_reranker=False)).pipeline.reranker is None
    assert factory.build(params(use_reranker=True)).pipeline.reranker is not None


def test_reranker_model_not_built_for_plain_configs(corpus, tmp_path):
    f = make_factory(corpus, tmp_path)
    f._reranker = None
    f.build(params(use_reranker=False))
    assert f._reranker is None


def test_answer_prompt_override_applies(factory):
    rag = factory.build(params(), answer_prompt="  custom prompt  ")
    assert rag.answerer.answer_prompt == "custom prompt"


def test_default_answer_prompt_comes_from_config(factory):
    assert "ONLY the provided context" in factory.build(params()).answerer.answer_prompt


def test_context_budget_comes_from_config(corpus, tmp_path):
    f = make_factory(corpus, tmp_path, "retrieval.context_budget_words=300")
    assert f.build(params()).pipeline.context_budget == 300


def test_embeddings_are_reused_from_disk_by_a_new_factory(corpus, tmp_path):
    first = make_factory(corpus, tmp_path)
    first.build(params(128, 32))
    assert first._embedder.calls == 1

    second = make_factory(corpus, tmp_path)
    rag = second.build(params(128, 32))
    assert second._embedder.calls == 0
    assert second.timing.count("embed_cache_hit") == 1
    assert rag.pipeline.hybrid.n == len(rag.chunks)


def test_embedding_cache_key_changes_with_the_chunking(corpus, tmp_path):
    f = make_factory(corpus, tmp_path)
    f.build(params(128, 0))
    f.build(params(256, 0))
    assert len(list((tmp_path / "emb").glob("*.npy"))) == 2


def test_embedding_cache_can_be_disabled(corpus, tmp_path):
    f = make_factory(corpus, tmp_path, "runtime.embedding_cache_dir=null")
    f.build(params())
    assert f.embedding_cache is None
    assert not (tmp_path / "emb").exists()


def test_timing_record_logs_zero_duration_events():
    t = TimingLog()
    t.record("index_cache_hit", 0.0, chunk_size=256)
    assert t.count("index_cache_hit") == 1
    assert t.records[-1] == {"label": "index_cache_hit", "chunk_size": 256, "seconds": 0.0}


def test_counting_embedder_is_unit_length():
    import numpy as np
    vecs = CountingEmbedder().embed(["a b c", "d"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_timing_total_counts_nested_steps_once():
    t = TimingLog()
    with t.timed("outer"):
        with t.timed("inner"):
            pass
        t.record("marker", 5.0)
    outer = next(r for r in t.records if r["label"] == "outer")
    assert t.total() == outer["seconds"]
    assert next(r for r in t.records if r["label"] == "inner")["depth"] == 1


def test_generator_is_cached_at_temperature_zero(corpus, tmp_path):
    from selfrag.utils.cache import CachedLLM
    f = make_factory(corpus, tmp_path, f"runtime.cache_dir={tmp_path / 'cache'}")
    assert isinstance(f.generator, CachedLLM)
    g = make_factory(corpus, tmp_path, "models.generator.temperature=0.7")
    assert not isinstance(g.generator, CachedLLM)


def test_cached_llm_calls_the_model_once(tmp_path):
    from selfrag.utils.cache import CachedLLM, DiskCache

    class Model:
        name, temperature, max_tokens, num_ctx = "m", 0.0, 64, 2048
        calls = 0

        def generate(self, prompt, system=None, **kw):
            Model.calls += 1
            return prompt.upper()

    llm = CachedLLM(Model(), DiskCache(tmp_path))
    assert llm.generate("hi", system="s") == "HI"
    assert CachedLLM(Model(), DiskCache(tmp_path)).generate("hi", system="s") == "HI"
    assert llm.generate("hi", system="other") == "HI"
    assert Model.calls == 2


def test_disk_cache_ignores_a_truncated_entry(tmp_path):
    from selfrag.utils.cache import DiskCache
    cache = DiskCache(tmp_path)
    key = cache.key({"a": 1})
    cache.set(key, "ok")
    cache._path(key).write_text('{"value": "tru')
    assert cache.get(key) is None
    assert cache.get_or_compute({"a": 1}, lambda: "again") == "again"
    assert cache.get(key) == "again"


def test_reranker_scores_each_pair_once():
    from selfrag.models.st_adapter import STReranker

    class Model:
        seen = 0

        def predict(self, pairs, show_progress_bar=False):
            Model.seen += len(pairs)
            return [float(len(p)) for _, p in pairs]

    rr = STReranker("x")
    rr._model = Model()
    assert rr.rerank("q", ["aa", "b"]) == [2.0, 1.0]
    assert rr.rerank("q", ["b", "ccc", "aa"]) == [1.0, 3.0, 2.0]
    assert Model.seen == 3


def test_embed_command_fills_the_cache_for_every_chunking(corpus, tmp_path, monkeypatch):
    from selfrag import cli

    monkeypatch.setattr(cli.RagFactory, "embedder", property(lambda self: CountingEmbedder()))
    rc = cli.main(["embed", "--config", "configs/qasper.yaml",
                   "--set", f"dataset.corpus_path={corpus}",
                   "--set", f"runtime.embedding_cache_dir={tmp_path / 'emb'}"])
    assert rc == 0
    # 3 sizes x 3 overlaps, plus the 256/32 baseline which is already in the grid
    assert len(list((tmp_path / "emb").glob("*.npy"))) == 9
