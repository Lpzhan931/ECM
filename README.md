# ECM: Expected Contribution Matching

ECM is a training-free KV cache compression method for long-context LLMs.
It combines pair-level eviction selection with global semantic similarity
matching and a closed-form solution for optimal value redistribution.

> **EMNLP 2026 Findings:** This repository is the official implementation of
> *Expected Contribution Matching: Optimal Values Merging for KV Cache
> Compression*, accepted to Findings of EMNLP 2026.

## Installation

```bash
pip install -e .
```

## Usage

> **Note:** The current prediction pipeline has only been validated in
> single-process execution. Multi-process and multi-node output writing has not
> yet been validated.

```bash
python pred.py --model Llama-3.1-8B-Instruct
python pred.py --model Mistral-7B-Instruct-v0.3
```

The prediction script saves results to a timestamped directory and prints the
directory when inference finishes:

```text
pred/<model>/ecm_<timestamp>/
```

Pass the generated directory name to `eval.py` through `--mode`:

```bash
python eval.py \
  --model Llama-3.1-8B-Instruct \
  --mode ecm_YYYYMMDD_HHMMSS
```

For LongBench-E, both prediction and evaluation require `--e`:

```bash
python pred.py --model Llama-3.1-8B-Instruct --e
python eval.py \
  --model Llama-3.1-8B-Instruct \
  --mode ecm_YYYYMMDD_HHMMSS \
  --e
```

The LongBench-E outputs are stored under
`pred_e/<model>/ecm_<timestamp>/`.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | Llama-3.1-8B-Instruct | Model name |
| `--ratio` | None | Compression ratio (0-1], defaults to recent_size=2048 |
| `--start_size` | 32 | Sink token window (never compressed) |
| `--e` | False | Evaluate on LongBench-E |

## Dependencies

The code has been validated with the following environment. In particular, the
custom attention implementations depend on the Transformers 4.44.2 model APIs.

- Python 3.9.25
- torch 2.8.0+cu128
- transformers 4.44.2
- datasets 4.5.0
- tqdm 4.67.3
- numpy 1.26.4
- accelerate 1.10.1
- jieba 0.42.1
- fuzzywuzzy 0.18.0
- rouge 1.0.1

## Citation

[TBD]
