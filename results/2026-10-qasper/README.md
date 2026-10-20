Output of the October 2026 run. It was made with an earlier version of this code; the only
changes since are log formatting and a corpus check in the `pipeline` command, neither of
which affects these numbers.

- `optimize/best_configs.json`: chosen configs, all scores, paired tests (questions with the paper title)
- `optimize/*_stage1_trials.jsonl`, `*_stage2_trials.jsonl`: every configuration and prompt tried
- `reeval-unscoped/reeval.json`: the same configs on the questions without the title
- `evalsets/*_manifest.json`: how many generated pairs were kept and why the others were dropped
- `*/run_meta.json`: start time and command-line overrides

Per-question results and timing logs were left out to keep the repo small.
