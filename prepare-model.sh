#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# != 1 || ! -f $1/model.safetensors.index.json ]]; then
    echo "Usage: bash prepare-model.sh PATH_TO_LING_HF_SNAPSHOT" >&2
    exit 2
fi
source_dir=$(cd -- "$1" && pwd)
models="$root/models"
python_bin=${LAENGINE_PYTHON:-python3}
runtime=${LAENGINE_RUNTIME:-$root/runtime}
export LD_LIBRARY_PATH="$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -x "$runtime/llama-quantize" || ! -f "$root/.llama.cpp/convert_hf_to_gguf.py" ]]; then
    echo "Build the runtime and llama-quantize with build.sh first." >&2
    exit 1
fi

mkdir -p "$models"
if [[ -e "$models/Ling-3.2-air-IQ4_NL-Packed-Conv.gguf" ]]; then
    echo "The packed model already exists in $models."
    exit 0
fi
scratch=$(mktemp -d "$models/.convert.XXXXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$python_bin" "$root/.llama.cpp/convert_hf_to_gguf.py" "$source_dir" \
    --outtype bf16 --use-temp-file --outfile "$scratch/ling-bf16.gguf"
"$runtime/llama-quantize" "$scratch/ling-bf16.gguf" "$scratch/ling-iq4.gguf" IQ4_NL "${LAENGINE_QUANT_THREADS:-12}"
PYTHONPATH="$root/.llama.cpp/gguf-py${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" "$root/pack-kda.py" "$scratch/ling-iq4.gguf" "$scratch/ling-packed.gguf" --conv
mv -n -- "$scratch/ling-packed.gguf" "$models/Ling-3.2-air-IQ4_NL-Packed-Conv.gguf"
