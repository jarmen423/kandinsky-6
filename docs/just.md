# just recipes

Run every recipe from the repository root. You need [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

`just setup` installs the Kandinsky pipeline into `.venv` (Python 3.13 or 3.14). vLLM-Omni, SGLang, and FastVideo each get a Python 3.12 environment under `.venvs/` and do not replace `.venv`. A generate or serve recipe stops and names the setup recipe when its environment is missing. A second setup reuses the environment already on disk. vLLM-Omni also reuses its source checkout.

`$KANDINSKY_HOME` is the root for weights, logs, and runs. When it is unset, that root is `~/.cache/kandinsky`.

`just --list` prints the recipes.

| Recipe | Guide |
|---|---|
| `just setup`, `just generate`, `just generate-sr`, `just download` | [Quick start](quickstart/quick_start.md) |
| `just setup-vllm`, `just generate-vllm`, `just serve-vllm` | [vLLM-Omni](quickstart/quick_start_vllm.md) |
| `just setup-sglang`, `just generate-sglang`, `just serve-sglang` | [SGLang](quickstart/quick_start_sglang.md) |
| `just setup-fastvideo`, `just generate-fastvideo` | [FastVideo](quickstart/quick_start_fastvideo.md) |
| `just bench` | Eight native t2v timings on one GPU: SD and Full HD. JSON files go to `profiles/` in the run folder. |
| `just memfit` | One generation per 16, 24, and 32 GB consumer preset. A driver allocation leaves that much memory free on each GPU. |

Generation modes for `just generate` and the Python pipeline are in [generation modes](generation_modes.md).
