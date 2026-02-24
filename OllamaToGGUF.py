import os
import json
import shutil
import sys
import platform

# ---------------------------------------------------------------------------
# Directory resolution — respects OLLAMA_MODELS env var (CI-friendly)
# ---------------------------------------------------------------------------

def get_ollama_models_dir():
    env_val = os.environ.get("OLLAMA_MODELS", "").strip()
    if env_val:
        return env_val
    if platform.system() == "Windows":
        return os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), ".ollama", "models")
    return os.path.join(os.path.expanduser("~"), ".ollama", "models")

models_dir   = get_ollama_models_dir()
manifest_dir = os.path.join(models_dir, "manifests", "registry.ollama.ai")
blob_dir     = os.path.join(models_dir, "blobs")

# Output goes next to the script by default
current_dir      = os.path.dirname(os.path.abspath(__file__))
outputModels_dir = os.path.join(current_dir, "Output")

print("\nOllama To GGUF\n")
print("Confirming Directories:\n")
print(f"Manifest Directory: {manifest_dir}")
print(f"Blob Directory: {blob_dir}")
print(f"Output Models Directory: {outputModels_dir}")

if not os.path.exists(outputModels_dir):
    os.makedirs(outputModels_dir)
    print("Output Models Directory Created.")
else:
    print("Output Models Directory Confirmed.")

# ---------------------------------------------------------------------------
# Manifest scanning
# ---------------------------------------------------------------------------

files = []
for dirpath, _, filenames in os.walk(manifest_dir):
    for filename in filenames:
        files.append(os.path.join(dirpath, filename))
manifest_locations = files[:]

if not manifest_locations:
    print("No manifest files found.")
else:
    print("Manifest files found")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_size(layers, blob_directory):
    total_size = 0
    for layer_info in layers:
        digest = layer_info["digest"]
        sha    = digest.split(":")[1]
        blob   = os.path.join(blob_directory, f"sha256-{sha}")
        if os.path.exists(blob):
            total_size += os.path.getsize(blob)
    return total_size


def recombine_model(manifest_path, blob_directory, output_directory, final_output_override=None):
    """
    Recombine Ollama blob layers into a single GGUF file.

    Args:
        manifest_path:         Path to the JSON manifest.
        blob_directory:        Directory containing sha256-* blobs.
        output_directory:      Default output root (Output/<modelName>/<file>.gguf).
        final_output_override: If provided, write the GGUF directly to this path
                               instead of the default Output subdirectory.
                               Used by CI (argv[2]).
    """
    with open(manifest_path) as f:
        obj = json.load(f)

    config = obj.get("config")
    if not config:
        raise ValueError("Config section missing from manifest JSON.")

    digest = config.get("digest")
    if not digest:
        raise ValueError("Digest missing from config section.")

    sha_value    = digest.split(":")[-1]
    sha_file     = os.path.join(blob_directory, f"sha256-{sha_value}")

    with open(sha_file) as f:
        config_data = json.load(f)

    try:
        modelQuant = config_data["file_type"]
        assert isinstance(modelQuant, str) and len(modelQuant) > 0
    except Exception as e:
        raise ValueError("Invalid or missing `file_type` in model config blob.") from e

    trained_on = str(config_data.get("model_type", "unknown"))

    layers = obj.get("layers")
    if not layers:
        raise ValueError("Layers section is required but missing from manifest.")

    modelName = os.path.basename(os.path.dirname(manifest_path))

    # Determine output path
    if final_output_override:
        final_output_filepath = final_output_override
        os.makedirs(os.path.dirname(os.path.abspath(final_output_filepath)), exist_ok=True)
    else:
        target_subdir         = os.path.join(output_directory, modelName)
        combined_filename     = f"{modelName}-{trained_on}-{modelQuant}.gguf"
        final_output_filepath = os.path.join(target_subdir, combined_filename)
        os.makedirs(target_subdir, exist_ok=True)

    print(f"Output file: {final_output_filepath}")

    prefix = ""  # initialise so except block can always reference it
    source_blob = ""
    try:
        print("Reading layers...")
        with open(final_output_filepath, "wb") as final_fobj:
            for layer_index, layer_info in enumerate(layers):
                print(f"  Layer {layer_index}: {layer_info.get('mediaType', 'unknown')}")
                mediaType   = layer_info["mediaType"]
                digest      = layer_info["digest"]
                sha         = digest.split(":")[1]
                source_blob = os.path.join(blob_directory, f"sha256-{sha}")

                prefix = f"\t[{modelName}] [{mediaType}]"
                msg    = prefix + f"\tBlob: [{source_blob}] "
                status = "Reading"
                sys.stdout.write(msg.ljust(80) + status.rjust(48) + "\r\n")

                with open(source_blob, "rb") as layer_fobj:
                    shutil.copyfileobj(layer_fobj, final_fobj, length=10 * 1024 * 1024)

    except Exception as excp:
        msg    = prefix + f"\tFailed reading [{source_blob}]: {excp}"
        status = "Failed"
        sys.stdout.write(msg.ljust(80) + status.rjust(48) + "\r\n")
        raise

    return final_output_filepath


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def load_model_info(manifest_path):
    """Load display info for one manifest entry."""
    modelName         = os.path.basename(os.path.dirname(manifest_path))
    manifest_filename = os.path.basename(manifest_path)
    try:
        with open(manifest_path) as f:
            obj = json.load(f)
        config = obj.get("config", {})
        digest = config.get("digest", "")
        sha_value  = digest.split(":")[-1]
        sha_file   = os.path.join(blob_dir, f"sha256-{sha_value}")
        with open(sha_file) as f:
            config_data = json.load(f)
        modelQuant = config_data.get("file_type", "Unknown")
        layers     = obj.get("layers", [])
        size_bytes = get_model_size(layers, blob_dir)
        size_str   = f"{size_bytes / (1024 * 1024):.2f} MB" if size_bytes > 0 else "Unknown"
    except Exception:
        modelQuant = "Unknown"
        size_str   = "Unknown"
    return modelName, manifest_filename, modelQuant, size_str


def select_noninteractive(target_model):
    """
    Find the manifest matching target_model (e.g. 'qwen3-embedding:4b').
    Falls back to the first available manifest if no match.
    """
    name_part = target_model.split(":")[0].lower().strip()
    tag_part  = target_model.split(":")[1].lower().strip() if ":" in target_model else None

    # Exact match: directory name == model name AND filename == tag
    for path in manifest_locations:
        dir_name  = os.path.basename(os.path.dirname(path)).lower()
        file_name = os.path.basename(path).lower()
        if dir_name == name_part and (tag_part is None or file_name == tag_part):
            return path

    # Name-only match
    for path in manifest_locations:
        dir_name = os.path.basename(os.path.dirname(path)).lower()
        if dir_name == name_part:
            return path

    # Fallback
    print(f"[WARNING] No manifest matched '{target_model}'. Using first available.")
    return manifest_locations[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # CLI args: argv[1] = model name (non-interactive), argv[2] = output file path
    cli_model  = sys.argv[1].strip() if len(sys.argv) > 1 else None
    cli_output = sys.argv[2].strip() if len(sys.argv) > 2 else None

    # Also accept model name via env var
    if not cli_model:
        cli_model = os.environ.get("OLLAMA_MODEL_NAME", "").strip() or None

    if not manifest_locations:
        print("No manifest files found.")
        sys.exit(1)

    # --- Non-interactive (CI) mode ---
    if cli_model:
        manifest_path = select_noninteractive(cli_model)
        modelName, manifest_filename, modelQuant, size_str = load_model_info(manifest_path)
        print(f"\n[CI] Auto-selected: {modelName} (Manifest: {manifest_filename}, "
              f"Quantization: {modelQuant}, Size: {size_str})\n")

        out_path = recombine_model(manifest_path, blob_dir, outputModels_dir,
                                   final_output_override=cli_output)

        # If a specific output path was requested, ensure the file is there
        if cli_output and os.path.abspath(out_path) != os.path.abspath(cli_output):
            shutil.copy2(out_path, cli_output)
            print(f"Copied output to: {cli_output}")

        print(f"Successfully converted to GGUF: {cli_output or out_path}")
        return

    # --- Interactive mode (local use) ---
    while True:
        print("\nAvailable Ollama Models to Convert:\n")
        for index, manifest_path in enumerate(manifest_locations, start=1):
            modelName, manifest_filename, modelQuant, size_str = load_model_info(manifest_path)
            print(f"  {index}. {modelName} (Manifest: {manifest_filename}, "
                  f"Quantization: {modelQuant}, Size: {size_str})")

        try:
            choice = int(input("\nEnter the number of the model you want to convert (or 0 to exit): "))
            if choice == 0:
                print("Exiting.")
                break
            elif 1 <= choice <= len(manifest_locations):
                manifest_path = manifest_locations[choice - 1]
                out_path = recombine_model(manifest_path, blob_dir, outputModels_dir)
                print(f"Successfully converted to GGUF: {out_path}")
            else:
                print(f"Invalid choice. Enter 1–{len(manifest_locations)} or 0 to exit.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:
            print("\nNo input available (non-interactive terminal). Use: python OllamaToGGUF.py <model> <output.gguf>")
            sys.exit(1)


if __name__ == "__main__":
    main()
