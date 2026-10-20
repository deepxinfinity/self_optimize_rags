# selfrag: tune a RAG pipeline on your own documents

selfrag takes a folder of documents and finds a good RAG setup for them. An LLM writes test
questions from the documents, selfrag tries different retrieval setups and answer prompts
against those questions, and reports which one works best, with significance tests against an
untuned baseline. It runs locally through Ollama; API models work too.

It tunes settings (chunking, search, reranking, the answer prompt), not model weights. There's
no fine-tuning involved.

The obvious worry with this approach is that the test questions come from an LLM, not from your
users. So the second half of this README is a study on [QASPER](https://allenai.org/data/qasper)
that checks whether tuning on LLM-written questions actually helps on real ones. Short answer:
yes, by about 7–8 points of answer correctness, as long as it's clear which document a question
is about.

## What it does

1. **Writes test questions.** A model writes question-answer pairs from passages of your
   documents and quotes the sentence that supports each answer. Pairs whose quote isn't really
   in the passage are thrown out. The rest are split into train / val / test (60 / 20 / 20).
2. **Tunes retrieval (stage 1).** Tries 40 retrieval setups and keeps the one that most often
   puts the passage with the answer in front of the answering model.
3. **Tunes the answer prompt (stage 2).** Another model rewrites the answer prompt, and a
   rewrite is kept only if it answers the val questions clearly better.
4. **Tests once.** The tuned setup and the untuned baseline answer the held-out questions: the
   synthetic test split, plus real questions if you have some. A judge model compares each
   answer with the reference answer, and paired tests compare the two setups question by
   question.

You get the winning settings and prompt (`best_configs.json`), every setup that was tried, and
per-question results. `selfrag ui` opens a local dashboard where you can watch runs, chat with
the baseline or the tuned setup, or upload documents and start a run from the browser.

## How the tuning works

### The pipeline

```
question
  -> search all chunks with BM25 (keywords) and embeddings, blended by alpha
  -> optionally rerank the top 20 or 40 hits with a cross-encoder
  -> keep the top k chunks, up to 1600 words in total
  -> the answer model reads those chunks and answers, following the answer prompt
```

Everything in that pipeline except the models is a setting the tuner can change.

### Stage 1: retrieval settings

| Setting | What it does | Values tried | Baseline |
|---|---|---|---|
| Chunk size | words per chunk | 128, 256, 384 | 256 |
| Chunk overlap | words shared by neighbouring chunks | 0, 32, 64 | 32 |
| Chunks kept (k) | how many chunks the model reads | 3 to 12 | 5 |
| Keyword vs embedding mix | 0 = keywords only, 1 = embeddings only | 0.0 to 1.0 | 0.5 |
| Reranker | rescore the search hits with a cross-encoder | on, off | off |
| Reranker pool | how many hits the reranker looks at | 20, 40 | (off) |

Optuna picks 40 combinations, steering towards the ones that scored well so far (TPE sampler).
The baseline is always scored first, so the result can't be worse than it. Each combination is
scored on the val questions: for each question, did the chunk containing the answer (matched
against the quoted evidence) end up in the text the model reads? No language model is involved
in this stage, so it's fast. The best score wins; ties go to the setup that ranks the answer
chunk higher, then to the cheaper one.

Every setup gets the same 1600 words of context. Without that limit, bigger chunks or more
chunks win just by showing the model more of the document, which isn't the same as retrieving
better.

### Stage 2: the answer prompt

Retrieval is frozen at the stage 1 winner, and the answer prompt starts as:

> Answer the question using ONLY the provided context. If the context is insufficient, say so.
> Be concise and cite nothing outside the context.

The current prompt answers 20 train questions. Each round, a second model sees three it got
wrong (question, reference answer, our answer, judge score), a different three each time, and
writes a new prompt. The new prompt answers 40 val questions and is kept only if its correctness
is more than 0.02 higher. The examples come from train and the decision from val, so the
rewriter never sees the questions it's judged on.

### Keeping the test honest

The held-out questions can't be read until tuning is finished: the code raises an error if
anything tries, and every read is logged in the results. Without that, it's easy to end up
choosing a setup on the same questions you report.

## Using it on your own documents

You need Python 3.11, [Ollama](https://ollama.com) and about 15 GB of disk for the models.

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
ollama pull qwen2.5:7b && ollama pull llama3.1:8b && ollama pull llama3.2:3b
OLLAMA_MAX_LOADED_MODELS=1 nohup ollama serve > ~/ollama.log 2>&1 &

python -m selfrag pipeline --config configs/default.yaml \
    --set dataset.corpus_path=./my_docs \
    --set dataset.real_questions_path=null \
    --set evalgen.out_dir=./data/evalsets_my_docs
```

That writes the test questions, runs both stages and prints the results; everything is saved in
`runs/<date>-optimize/`. Expect a few hours on a laptop (roughly 4–5 on a 16 GB MacBook Air),
most of it spent waiting on the language models. `.txt`, `.md`, `.pdf`, `.docx` and `.html`
files are read.

- **Have some real questions?** Put them in a JSONL file, one per line:
  `{"question": ..., "answer": ..., "gold_doc_id": "<file name relative to corpus_path>"}`, and
  pass it as `dataset.real_questions_path`. Add `"evidence": ["<sentence with the answer>"]` to
  also get passage-level retrieval scores. Without real questions, all scores are on synthetic
  questions, which overstate quality (see below).
- **To try it quickly** on the three short documents in `data/corpus/sample/`, run the same
  command without the `--set` lines and add `--set evalgen.n_pairs_per_arm=15`. It takes about
  10 minutes; with so little text every setup scores perfectly, so it only shows that things work.
- **Models** are set in `configs/default.yaml`. To judge with an API model instead, swap in the
  commented-out `judge` block and set `ANTHROPIC_API_KEY`; every API call is cached on disk.
- **To compare question writers**, add more entries under `eval_generators`. That's what the
  study below does.
- **Then chat with the result:** `python -m selfrag ui --config configs/default.yaml` and open
  http://127.0.0.1:8000.

## Does it work? A study on QASPER

QASPER is a set of NLP papers with questions written by researchers and annotated answers. I
used 279 papers and 300 of their questions, each with the paragraph that contains the answer.
Those real questions were never shown to the tuner; they're only used for the final test.

To also check whether the question-writing model matters, the tuning ran twice: once on
questions written by llama3.2:3b and once on questions written by gemma2:9b (400 each). Answers
come from qwen2.5:7b and the judge is llama3.1:8b. Everything ran locally on a 16 GB MacBook Air;
the main run took about 14 hours. The full output is in
[results/2026-10-qasper](results/2026-10-qasper).

### The short version

- **Tuning helped on real questions, but less than the synthetic numbers suggested.** Answer
  correctness went from 0.66 to 0.73–0.74 (significant, p ≈ 0.02–0.03). The synthetic test set
  said 0.96–0.98.
- **Most of the gain seems to come from one thing: a reranker.** Every top configuration used a
  cross-encoder reranker over 20 candidates, with mostly keyword search. Rewriting the answer
  prompt added little.
- **The gain depends on the question naming its paper.** With questions as people actually wrote
  them ("what baseline did they use?"), the system finds the right paper only half the time and
  the improvement shrinks to +0.02–0.035, too small to detect with 150 questions. On the half
  where the right paper is found, it's still there (+0.11 for one of the two setups).
- **The model that wrote the questions didn't matter.** Questions from a 3B model and from a
  9B model led the tuner to almost the same configuration and the same real-world results.

### Questions that name the paper

Each question was prefixed with its paper's title, e.g. *In the paper "Sentiment Analysis of
Twitter Data for Predicting Stock Market Movements": What dataset is used to train the model?*
The synthetic questions got the same prefix, so tuning and testing see the same kind of question.
Retrieval is scored on all 300 questions, answers on 150.

| | Answer correctness | Answer passage retrieved |
|---|---|---|
| Baseline (untuned) | 0.663 | 43% |
| Tuned on llama3.2:3b questions | **0.743** (+0.080, p = 0.02) | **53%** |
| Tuned on gemma2:9b questions | **0.733** (+0.070, p = 0.03) | **54%** |

Faithfulness (does the answer stick to the retrieved text) didn't change: 0.82–0.87.

### Same questions, as people wrote them

| | Answer correctness | Right paper retrieved | Answer passage retrieved |
|---|---|---|---|
| Baseline (untuned) | 0.553 | 49% | 27% |
| Tuned on llama3.2:3b questions | 0.588 (+0.035, not significant) | 52% | 28% |
| Tuned on gemma2:9b questions | 0.573 (+0.020, not significant) | 52% | 30% |

Without the title, most questions don't say which paper they're about, so half of them never
reach the right paper and nothing in the tuned settings can fix that. Looking only at the
questions where both the baseline and the tuned setup found the right paper, the tuned setup
still answers better: 0.72 → 0.83 for the llama-tuned one (p = 0.01) and 0.71 → 0.78 for the
gemma-tuned one (p = 0.09).

### Synthetic vs real

| | Synthetic test questions | Real questions |
|---|---|---|
| Answer correctness, tuned | 0.96–0.98 | 0.73–0.74 |
| Improvement over baseline | +0.11 / +0.19 | +0.08 / +0.07 |

The synthetic questions are much easier: each one is written while looking at a single passage
and tends to reuse its wording. So they overstate both how good the system is and how much
tuning helped.

### What the tuning picked

| Setting | Baseline | Tuned on llama3.2:3b questions | Tuned on gemma2:9b questions |
|---|---|---|---|
| Chunk size / overlap | 256 / 32 | 256 / 64 | 256 / 32 |
| Chunks kept | 5 | 8 (6 fit in 1600 words) | 6 |
| Keyword vs embedding mix | 0.5 | 0.17 | 0.12 |
| Reranker | off | on, 20 candidates | on, 20 candidates |
| Answer prompt | starting prompt | one rewrite accepted | starting prompt |

The reranker made the biggest difference: the best stage-1 scores without it were 0.89 and
0.80, with it 0.99 and 0.93. The accepted prompt rewrite added instructions like "focus on
extracting relevant information" and "do not introduce external knowledge". In both runs only
the first two prompt rounds produced anything new; after that the rewriter kept repeating itself.

### The first attempt didn't work

My first full run (July 2026) said the opposite: the untuned baseline scored best. Looking into
it, the problem was mostly the evaluation, not the tuning:

- The main metric was faithfulness, which gives full marks to "the context doesn't say". The
  configuration that retrieved the least text won. I switched to judging correctness against
  the reference answer.
- Retrieval was scored by "did any chunk from the right paper come back". That was almost always
  true for synthetic questions and capped at about 40% for real ones, so it couldn't tell
  configurations apart. It now checks whether the chunk with the actual answer came back.
- There were only 24 val questions, and a prompt rewrite was accepted for gaining one judge
  point. Now there are 80, with a minimum improvement before a rewrite is kept.
- The tuner could return something worse than the baseline. The baseline is now always the
  first trial.

### Limitations

- The judge is a local 8B model and I haven't checked it against human judgment yet. It gives
  0.35–0.39 correctness even when the right paper wasn't retrieved, which looks generous.
- Answers were judged on 150 questions per configuration, from one run with one random seed.
- The two question writers are small local models standing in for the comparison I actually
  care about: a fine-tuned question generator vs a frontier API model.
- I haven't yet run the baseline with only the reranker added, so "the reranker did most of it"
  is an inference from the tuning logs, not a measured result.
- Prompt tuning barely ran: after two real attempts per setup, the rewriting model kept proposing
  prompts it had already tried.
- The val set ended up at 0.99, so it couldn't separate the best few configurations.
- I chose the title prefix and the passage-level metric after looking at how retrieval did on
  the real questions. No configuration was picked using them, but the evaluation design saw them.
- Faithfulness can't tell when an answer comes from the wrong paper; it only checks the answer
  against whatever was retrieved.

### Reproducing it

Download the QASPER dev set (`qasper-dev-v0.3.json`) into `data/qasper_raw/`, pull `gemma2:9b`
as well as the models above, then:

```bash
# papers + real questions (writes both the titled and untitled versions)
python -m selfrag ingest-qasper --n-papers 1000 --n-questions 300 \
    --scope-with-title --require-evidence \
    --real-out data/real_questions_qasper_scoped.jsonl \
    --unscoped-out data/real_questions_qasper_unscoped.jsonl

python -m selfrag embed --config configs/qasper.yaml --set models.embedding.device=mps   # ~1 h
python -m selfrag gen-eval --config configs/qasper.yaml                                  # ~3 h
python -m selfrag optimize --config configs/qasper.yaml                                  # ~14 h

# the tuned setups again on the untitled questions (use your own run folder)
python -m selfrag reeval --run runs/20261017-234738-optimize --config configs/qasper.yaml \
    --set dataset.real_questions_path=./data/real_questions_qasper_unscoped.jsonl     # ~8.5 h
```

Every model call and embedding is cached on disk, so if a run dies you can start the same
command again and it skips what's done.

## Notes for a 16 GB Mac

- Keep the lid open. `caffeinate` stops idle sleep but not lid-close sleep.
- Don't run the embedder or reranker on the GPU during a long run; they hold 4–8 GB and Ollama
  needs it. The configs keep them on CPU, and `selfrag embed` does the one-off bulk embedding
  on the GPU in its own process.
- Run Ollama in the background (as above) rather than in a terminal you might Ctrl+Z.
  A suspended server keeps its port open and requests just wait.
- Small models sometimes loop forever in JSON mode, so every model has a `max_tokens` cap.

## Code

```
selfrag/
  corpus/      loading documents, chunking, QASPER import
  index/       BM25, dense search, hybrid scoring
  retrieve/    retrieval pipeline (search, rerank, context budget)
  generate/    answering, query rewriting
  evalgen/     synthetic question generation, splits, held-out lock
  metrics/     retrieval metrics, judge-based metrics, token F1
  optimize/    stage 1, stage 2, the full run
  analysis/    paired significance tests
  models/      Ollama, sentence-transformers and API adapters, the judge
  ui/, web/    dashboard, chat, project upload API
configs/       default.yaml (your documents), qasper.yaml (the study)
results/       output of the October 2026 study
tests/         python -m pytest -q  (no models or network needed)
```
