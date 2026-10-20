"""Stage 1: tune the retrieval knobs on val with Optuna, scored by retrieval metrics only."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
from optuna.samplers import TPESampler

from ..config import Config
from ..evalgen.generator import QAPair
from ..evaluate import evaluate_retrieval
from ..params import RetrievalParams, baseline_retrieval_params, describe, params_from_dict
from ..rag import RagFactory
from ..utils.runs import append_jsonl
from .search_space import suggest_retrieval_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

PASSAGE_METRICS = ("passage_hit_rate", "passage_mrr")


@dataclass
class Stage1Result:
    best_params: RetrievalParams
    best_value: float
    baseline_value: float
    metric: str
    n_trials: int
    total_seconds: float
    best_metrics: dict[str, Any]


class Stage1Optimizer:
    def __init__(self, cfg: Config, pairs: list[QAPair],
                 trial_log_path: Optional[str | Path] = None,
                 factory: Optional[RagFactory] = None,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg
        self.pairs = pairs
        stage1 = cfg.optimization.get("stage1_retrieval", {})
        self.metric = stage1.get("metric", "passage_hit_rate")
        self.n_trials = int(stage1.get("n_trials", 30))
        self.seed = cfg.splits.seed
        self.trial_log_path = Path(trial_log_path) if trial_log_path else None
        self.factory = factory or RagFactory(cfg)
        self.log = log
        self.trials: list[dict[str, Any]] = []

        if self.metric in PASSAGE_METRICS and not any(p.evidence for p in pairs):
            raise ValueError(f"stage-1 metric {self.metric!r} needs evidence on the val pairs; "
                             "these eval sets predate it, regenerate them with gen-eval")

    def _run_trial(self, params: RetrievalParams, label: Optional[str] = None) -> float:
        t0 = time.perf_counter()
        metrics, _ = evaluate_retrieval(self.factory.build(params), self.pairs)
        record: dict[str, Any] = {
            "trial": len(self.trials),
            "params": asdict(params),
            "metric": self.metric,
            "value": metrics.get(self.metric, 0.0),
            "metrics": metrics,
            "time_sec": round(time.perf_counter() - t0, 4),
        }
        if label:
            record["label"] = label
        self.trials.append(record)
        if self.trial_log_path is not None:
            append_jsonl(self.trial_log_path, record)
        if self.log:
            best = max(t["value"] for t in self.trials)
            self.log(f"  trial {record['trial']:>2}/{self.n_trials}: "
                     f"{self.metric}={record['value']:.3f} (best {best:.3f})"
                     f"  {describe(params)}  {record['time_sec']:.0f}s")
        return record["value"]

    def _objective(self, trial: optuna.Trial) -> float:
        params = suggest_retrieval_params(trial, self.cfg.search_space)
        if params.chunk_overlap >= params.chunk_size:
            raise optuna.TrialPruned()
        return self._run_trial(params)

    @staticmethod
    def _rank(trial: dict[str, Any]) -> tuple:
        # ties are common on small val sets; break them towards the finer metric, then the
        # cheaper config
        m, p = trial["metrics"], trial["params"]
        return (trial["value"], m.get("passage_mrr", 0.0), m.get("mrr", 0.0),
                not p["use_reranker"], -p["top_k"])

    def run(self) -> Stage1Result:
        t0 = time.perf_counter()
        # the baseline is scored as trial 0 so the winner can never be worse than it on val
        baseline_value = self._run_trial(baseline_retrieval_params(self.cfg), label="baseline")
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.seed))
        study.optimize(self._objective, n_trials=self.n_trials)

        best = max(self.trials, key=self._rank)
        return Stage1Result(
            best_params=params_from_dict(best["params"]),
            best_value=round(best["value"], 4),
            baseline_value=round(baseline_value, 4),
            metric=self.metric,
            n_trials=len(self.trials),
            total_seconds=round(time.perf_counter() - t0, 4),
            best_metrics=best["metrics"],
        )
