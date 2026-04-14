# HuggingFace Model Transfer Workflows

This repository provides GitHub Actions workflows for downloading, converting, and archiving machine learning models.

## Workflows

### 1. Transfer HuggingFace Models (`transfer.yml`)

Downloads a complete HuggingFace model repository and uploads it as a GitHub release.

**Use case:** Archive or mirror a model from HuggingFace to GitHub.

**Inputs:**
- `model_name` (required): HuggingFace model repository name (e.g., `meta-llama/Llama-2-7b-hf`)
- `split_size` (optional): Archive split size in MB (default: `800`, max ~2GB per GitHub release asset)
- `hf_token` (optional): HuggingFace token for private models

**Example:**
```
model_name: unsloth/Qwen3.5-8B-GGUF
split_size: 800
```

---

### 2. Transfer GGUF Models (`gguf.yml`)

Downloads specific GGUF files from HuggingFace using file patterns, with optional merge of split files.

**Use case:** Download specific quantization variants (e.g., Q4_K_M) from GGUF model repos.

**Inputs:**
- `model_name` (required): HuggingFace model repo (e.g., `unsloth/Qwen3.5-27B-GGUF`)
- `file_filter` (required): Glob pattern(s) for files (e.g., `*Q4_K_M.*`, or `*Q4_K_M.*;*Q8_0.*` for multiple)
- `split_size` (optional): Archive split size in MB (default: `800`)
- `merge_gguf` (optional): Set to `true` to merge split GGUF files (e.g., `model-00001-of-00002.gguf` + `model-00002-of-00002.gguf` → single file)

**Example:**
```
model_name: unsloth/Qwen3.5-27B-GGUF
file_filter: *Q4_K_M.*
merge_gguf: true
split_size: 800
```

---

### 3. Ollama to GGUF (`ollama2gguf.yml`)

Pulls a model from Ollama registry, converts it to GGUF format, and uploads to GitHub.

**Use case:** Convert Ollama models (e.g., `phi3:mini`, `llama3.2:latest`) to GGUF format for use with llama.cpp.

**Inputs:**
- `model_name` (required): Ollama model name (e.g., `qwen2.5:7b`, `phi3:mini`, `llama3.2:latest`)
- `split_size` (optional): Archive split size in MB (default: `800`)

**Example:**
```
model_name: qwen2.5:7b
split_size: 800
```

---

## How to Use

1. Navigate to the **Actions** tab in your GitHub repository
2. Select the desired workflow from the left sidebar
3. Click **Run workflow**
4. Fill in the required inputs
5. Click **Run workflow** again to start

## Notes

- All workflows run on `windows-latest` runners
- Large models are split into 7z volumes to stay under GitHub's 2GB asset limit
- Releases include SHA256 checksums where applicable
- For private HuggingFace models, ensure `HF_TOKEN` secret is configured in repository settings
