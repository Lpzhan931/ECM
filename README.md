# ECM: Expected Contribution Matching

ECM is a training-free KV cache compression method for long-context LLMs.
It combines pair-level eviction selection with global semantic similarity
matching and a closed-form solution for optimal value redistribution.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python pred.py --model Llama-3.1-8B-Instruct
python pred.py --model Mistral-7B-Instruct-v0.3
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | Llama-3.1-8B-Instruct | Model name |
| `--ratio` | None | Compression ratio (0-1], defaults to recent_size=2048 |
| `--start_size` | 32 | Sink token window (never compressed) |
| `--e` | False | Evaluate on LongBench-E |

## Dependencies

- torch
- transformers >= 4.42
- datasets
- tqdm
- numpy
- accelerate

## Citation

[TBD]
