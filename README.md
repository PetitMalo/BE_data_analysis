# BE Data Analysis

## Authors

* Ella Fournier
* Maël Dacher

## Description

[Insert your project description here. Describe the dataset, the analysis goals, and the methodology used.]

## Requirements

* Python 3.12 or higher
* Ollama (System-wide installation for LLM features) (optional)

## Installation

### Using uv (Recommended)

This project uses `uv` for dependency management. To set up the environment and install all dependencies:

```shell
uv sync

```

To install **uv** on Linux or macOS, run the following command:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh

```

For more details, refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Without uv

If you do not have `uv` installed, you can use standard `pip`. It is recommended to use a virtual environment:

```shell
python -m venv .venv
source .venv/bin/activate
pip install .

```


## LLM Setup (Optional)

The classification module requires a local Ollama server.

1. **Install Ollama server**
* Linux: `curl -fsSL https://ollama.com/install.sh | sh`
* macOS/Windows: Download from [ollama.com](https://ollama.com)


2. **Download the model**
```shell
ollama pull llama3.2:1b

```




---
