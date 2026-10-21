# Repository Guidelines

## Project Overview

Kronos is a decoder-only foundation model for financial candlestick (K-line) sequences. It has two stages: a hierarchical tokenizer that quantizes continuous OHLCV data into discrete tokens, and an autoregressive Transformer trained over those tokens. The repository provides:

- Inference/forecasting via `KronosPredictor` (single and batch).
- Fine-tuning on Qlib data (`finetune/`) or arbitrary CSV data (`finetune_csv/`).
- A Flask web UI for loading local CSVs and visualizing forecasts (`webui/`).
- Pre-trained weights on Hugging Face under `NeoQuasar/Kronos-*` and `NeoQuasar/Kronos-Tokenizer-*`.

Supported public models: `Kronos-mini` (context 2048), `Kronos-small` and `Kronos-base` (context 512). `Kronos-large` is not released.

## Architecture & Data Flow

### Model stack

| Layer | Location | Responsibility |
|---|---|---|
| `KronosTokenizer` | `src/kronos/kronos.py` | Encoder + `BSQuantizer` + decoder; `encode(x, half=False)` / `decode(z, half=False)`; `forward` returns `((z_pre, z), bsq_loss, quantized, z_indices)`. `half=True` uses only the s1 codebook. |
| `Kronos` | `src/kronos/kronos.py` | Causal Transformer over hierarchical s1/s2 token IDs plus temporal embeddings; `decode_s1` / `decode_s2` drive autoregressive inference; `DualHead` computes next-token losses. |
| `KronosPredictor` | `src/kronos/kronos.py` | Facade around model + tokenizer: validation, normalization/clipping, context truncation, sampling, inverse normalization, `predict` / `predict_batch`. |
| Building blocks | `src/kronos/module.py` | RoPE attention (self/cross), `RMSNorm`, SwiGLU feed-forward, `TemporalEmbedding`, `HierarchicalEmbedding`, `DependencyAwareLayer`, `DualHead`, `BSQuantizer` / `BinarySphericalQuantizer`. |

Both `KronosTokenizer` and `Kronos` extend `nn.Module` and `PyTorchModelHubMixin`, so checkpoints are loaded/saved with `from_pretrained(...)` / `save_pretrained(...)`. Constructor arguments are the checkpoint config; changing them breaks weight loading.

### Inference data flow

```
raw OHLCV DataFrame + x_timestamp/y_timestamp
  -> column validation, optional volume/amount zero-fill
  -> per-window z-score using lookback statistics only, clip to +/-5
  -> tokenizer.encode(..., half=True) -> (s1_ids, s2_ids)
  -> rolling max_context buffer; sample s1, then s2 conditioned on s1
  -> tokenizer.decode(..., half=True)
  -> inverse normalization -> DataFrame indexed by y_timestamp
```

`KronosPredictor.__init__(model, tokenizer, device=None, max_context=512, clip=5)` auto-selects `cuda:0` > `mps` > `cpu`. `predict(..., T=1.0, top_k=0, top_p=0.9, sample_count=1)` averages `sample_count` sampled paths. Inputs longer than `max_context` are truncated by the predictor; do not manually pre-truncate.

### Training data flow

- **Qlib pipeline** (`finetune/`): `qlib_data_preprocess.py` reads Qlib fields, computes `vol`/`amt`, splits by time, and writes `train/val/test_data.pkl`. `QlibDataset` emits sliding windows of `lookback + predict + 1`, normalizing each window from its lookback segment only. Tokenizer training optimizes reconstruction (pre/full) + BSQ loss; predictor training freezes the tokenizer, encodes `half=True`, shifts tokens by one, and optimizes cross-entropy through `DualHead`.
- **CSV pipeline** (`finetune_csv/`): YAML config selects a CSV; `finetune_tokenizer.py` produces `{base_path}/{exp_name}/tokenizer/best_model`, then `finetune_base_model.py` produces `{base_path}/{exp_name}/basemodel/best_model`. The predictor stage fails with `FileNotFoundError` if the fine-tuned tokenizer is absent. `train_sequential.py` orchestrates both stages.

## Key Directories

| Path | Purpose |
|---|---|
| `src/kronos/` | Installable package: tokenizer, autoregressive model, predictor, neural modules. |
| `pyproject.toml`, `uv.lock` | Package metadata and reproducible dependency lock; `uv build` emits wheel/sdist. |
| `finetune/` | End-to-end Qlib fine-tuning, dataset, config, backtest, DDP utilities. |
| `finetune_csv/` | CSV-based fine-tuning, YAML configs, sequential trainer, sample data. |
| `webui/` | Flask app, templates, launcher, prediction-result JSON output. |
| `tests/` | CPU regression/integration tests and golden CSV fixtures. |
| `examples/` | Runnable forecasting and backtest scripts, including AKShare data fetchers. |
| `figures/` | README/demo images only. |

## Development Commands

Requires Python 3.10+ and pip. There is no `pyproject.toml`, lockfile, Makefile, or CI workflow; install dependencies directly.

```bash
pip install -e .                   # installs the kronos package and core deps
pip install -e ".[qlib]"           # only for the finetune/ Qlib pipeline
uv build                           # wheel + sdist in dist/
```

Forecast smoke test from the repository root:

```python
from kronos import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)
pred_df = predictor.predict(
    df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
    pred_len=120, T=1.0, top_p=0.9, sample_count=1,
)
```

Runnable examples (scripts append `../` to `sys.path`, so run them from `examples/`):

```bash
cd examples
python prediction_example.py
python prediction_wo_vol_example.py
python prediction_batch_example.py
```

Qlib fine-tuning (Qlib pipeline scripts import `config` as a sibling module; run from `finetune/`):

```bash
cd finetune
python qlib_data_preprocess.py
torchrun --standalone --nproc_per_node=NUM_GPUS train_tokenizer.py
torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py
python qlib_test.py --device cuda:0
```

CSV fine-tuning (run from `finetune_csv/`):

```bash
cd finetune_csv
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-existing
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-basemodel   # tokenizer only
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-tokenizer   # predictor only
DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=8 train_sequential.py --config configs/config_ali09988_candle-5min.yaml
```

Web UI:

```bash
pip install -e ".[webui]"   # from the repository root
cd webui
python run.py               # or ./start.sh
# http://localhost:7070
```

## Code Conventions & Common Patterns

- **Style:** PEP 8-ish Python, 4-space indentation, Google-style docstrings with `Args`/`Returns`, type hints on new functions (existing code is partially annotated). No formatter or linter is configured; match surrounding code instead of introducing one.
- **Naming:** `snake_case` functions/variables/modules, `CamelCase` classes, `UPPER_CASE` constants (e.g. `MAX_CTX_LEN`, `FEATURE_NAMES`). Private helpers use a leading underscore (`_resolve_dynamic_paths`).
- **PyTorch idioms:** `torch.nn as nn`, `torch.nn.functional as F`, `einops` (`rearrange`, `reduce`), explicit `.to(device)` / `torch.device`, `non_blocking=True` for transfers. Data is `float32`; inference runs under `torch.no_grad()`.
- **Dependency direction:** `src/kronos/` must not import `finetune*`, `webui`, or test code. `KronosPredictor` receives constructed `model` and `tokenizer` objects (dependency injection) rather than loading them itself.
- **Preprocessing invariants:** compute normalization statistics from the lookback window only (never the target window) to avoid leakage; clip to `[-5, 5]`; use the tokenizer's `half=True` split consistently between training and inference.
- **Config:** `finetune/config.py` is a mutable plain `Config` class whose `__dict__` is passed around; add defaults there when adding required keys. `finetune_csv/` uses YAML plus `ConfigLoader`/`CustomFinetuneConfig`, with dot-path reads (`get("data.lookback_window")`) and no environment-variable or CLI interpolation. Empty `base_save_path`/`finetuned_tokenizer` resolve from `{base_path}/{exp_name}`.
- **Errors/logging:** raise `ValueError` for invalid input and `FileNotFoundError` for missing checkpoints; `NotImplementedError` signals unknown model names. Progress uses `tqdm` and `print`; distributed scripts guard logging by rank. There is no custom exception hierarchy or logging framework.
- **Async:** none. All pipelines are synchronous PyTorch/Flask code; do not introduce `asyncio` without a concrete need.
- **Path/import assumptions:** `from kronos import ...` is the canonical entry point (source in `src/kronos/`). Scripts append `../src` and use sibling imports (`from config import Config`, `from config_loader import ...`), so run them from their documented directories (`examples/`, `finetune/`, `finetune_csv/`). Preserve this layout or update every script together.
- **Pitfalls:**
  - `Kronos.forward` samples s1 tokens with `torch.multinomial` when teacher forcing is off; results are nondeterministic. Use greedy decoding (`top_k=1`) for deterministic checks.
  - Tokenizer `embed_dim` (`s1_bits + s2_bits`) must be divisible by `group_size` (asserted in `BSQuantizer`).
  - `Kronos-small`/`Kronos-base` context is 512; `Kronos-mini` is 2048. Mismatched `max_context` degrades or breaks inference.
  - Fine-tuning scripts are CUDA/NCCL-oriented and expect DDP env vars; they raise without `WORLD_SIZE` when launched directly.
  - `predict_batch` requires equal lookback and `pred_len` across series; missing volume/amount are zero-filled.
  - Do not edit `webui/prediction_results/*.json`; they are generated artifacts.

## Important Files

| File | Purpose |
|---|---|
| `src/kronos/kronos.py` | `KronosTokenizer`, `Kronos`, `KronosPredictor`, sampling, autoregressive inference, timestamp features. |
| `src/kronos/module.py` | Attention/RoPE, embeddings, normalization, feed-forward, quantization heads. |
| `src/kronos/__init__.py` | Public exports and `model_dict` / `get_model_class`. |
| `pyproject.toml` | Distribution metadata, dependencies/extras, pytest path, `uv_build` backend. |
| `finetune/config.py` | All Qlib pipeline paths, hyperparameters, and checkpoint names. |
| `finetune/dataset.py` | `QlibDataset` sliding-window construction and normalization. |
| `finetune/train_tokenizer.py`, `finetune/train_predictor.py` | DDP training loops; tokenizer must be trained first. |
| `finetune/qlib_data_preprocess.py`, `finetune/qlib_test.py` | Qlib ingestion; inference, signals, backtest, plots. |
| `finetune_csv/train_sequential.py` | Two-stage CSV training orchestration. |
| `finetune_csv/config_loader.py` | YAML loading, dot-path access, dynamic checkpoint path resolution. |
| `finetune_csv/configs/config_ali09988_candle-5min.yaml` | Reference config schema and defaults. |
| `webui/app.py`, `webui/run.py` | Flask routes/model lifecycle; launcher on port 7070. |
| `tests/test_kronos_regression.py` | Golden-output and MSE regression tests. |
| `tests/data/generate_regression_output.py` | Regenerates golden prediction CSVs. |

## Runtime/Tooling Preferences

- **Python:** 3.10+ (`pyproject.toml`). A committed `uv.lock` pins the full resolution; `uv sync` creates `.venv` and installs the package editable. `requirements.txt` remains for pip-only users. `torch` is not present in a bare interpreter.
- **Packages:** use `uv` (`uv sync`, `uv lock`, `uv build`) or pip (`pip install -e .`). Core deps are declared under `[project.dependencies]`; extras are `test`, `examples`, `webui`, and `qlib`. `webui/requirements.txt` remains for standalone UI setup. Build artifacts go to `dist/` (gitignored).
- **Hardware:** inference supports CUDA, Apple MPS, and CPU. Full fine-tuning assumes NVIDIA CUDA + `torchrun` (NCCL); the CSV pipeline accepts `DIST_BACKEND=gloo` for CPU/mixed setups.
- **Weights:** downloaded from Hugging Face on first use and cached in `~/.cache/huggingface`; tests pin model revisions via commit SHA. Network is required on a cold cache.
- **Web UI:** Flask development server, host `0.0.0.0`, port `7070`, debug enabled. It lazily loads models via `/api/load-model`, reads `.csv`/`.feather` from the project-root `data/` directory, and writes timestamped JSON to `webui/prediction_results/`.
- **CSV schema:** `timestamps, open, high, low, close, volume, amount` (volume/amount may be zero). Timestamp columns accepted by the UI are `timestamps`, `timestamp`, or `date`.

## Testing & QA

- **Framework:** `pytest`; one module with four parametrized CPU integration tests (no `conftest.py`, fixtures, coverage config, or CI).
- **Run from the repository root**; `pyproject.toml` adds `src/` to `pytest`'s import path:

  ```bash
  python -m pytest tests/test_kronos_regression.py -v
  ```

- **What is asserted:**
  - `test_kronos_predictor_regression[256|512]`: predicts 8 rows and compares all six OHLCV/amount columns against `tests/data/regression_output_{256,512}.csv` with `np.testing.assert_allclose(..., rtol=1e-5)`.
  - `test_kronos_predictor_mse[256|512]`: predicts 30 rows for 4 sampled starts and asserts average OHLC MSE against hardcoded constants (`0.008979`, `0.003741`) within `1e-6`.
  - Tests use `NeoQuasar/Kronos-small` + `NeoQuasar/Kronos-Tokenizer-base` at pinned revisions, `DEVICE="cpu"`, seed 123, and greedy decoding (`top_k=1`).
- **Fixture regeneration:** `python tests/data/generate_regression_output.py` rewrites the golden CSVs. It does **not** update `MSE_EXPECTED`; recompute those constants manually when intentionally changing model numerics, and treat any unplanned delta as a regression. Never loosen tolerances to make a change pass.
- **QA scope:** run the regression suite for edits affecting `src/kronos/`, tokenization, normalization, or sampling. For pipeline/config changes, exercise the relevant stage command (or a one-off smoke script) because the suite does not cover `finetune/`, `finetune_csv/`, or `webui/`. UI changes require manually loading a CSV and a model in the browser.
