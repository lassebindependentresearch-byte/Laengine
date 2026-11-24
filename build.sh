#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="$root/.llama.cpp"
build_dir="$root/.build"
revision=52975342ff6b85127ff31930d39e36725fd08809
patch="$root/laengine-v0.1.patch"

if [[ ! -d "$source_dir/.git" ]]; then
    git init -q "$source_dir"
    git -C "$source_dir" fetch --depth 1 https://github.com/ggml-org/llama.cpp.git "$revision"
    git -C "$source_dir" checkout --detach FETCH_HEAD
fi
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$revision" ]]; then
    echo "Expected llama.cpp $revision in $source_dir" >&2
    exit 1
fi
if ! git -C "$source_dir" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$source_dir" apply --check "$patch"
    git -C "$source_dir" apply "$patch"
fi

cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=ON -DGGML_NATIVE=ON \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build "$build_dir" --config Release --target llama-server llama-quantize -j "${LAENGINE_JOBS:-4}"
mkdir -p "$root/runtime"
cp -a "$build_dir/bin"/{llama-server,llama-quantize,libggml*.so*,libllama*.so*,libmtmd.so*} "$root/runtime/"
