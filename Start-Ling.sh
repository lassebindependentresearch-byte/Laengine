#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime="${LAENGINE_RUNTIME:-$here/runtime}"
packed=true
model="${LAENGINE_MODEL:-$here/models/Ling-3.2-air-IQ4_NL-Packed-Conv.gguf}"
draft=(--spec-type draft-mtp --spec-draft-n-max 2)
mode=${1:-}
copy_width=128
match_width=10

case "$mode" in
    --edit)
        draft=(--spec-type ngram-simple,draft-mtp --spec-draft-n-max 2)
        shift
        ;;
    --fast|--reference)
        packed=false
        if [[ $1 == --fast ]]; then
            model="${LAENGINE_MODEL:-$here/models/Ling-3.2-air-MTP-Q3_K_M.gguf}"
        else
            model="${LAENGINE_MODEL:-$here/models/Ling-3.2-air-MTP-IQ4_NL.gguf}"
        fi
        shift
        ;;
esac

context=(--ctx-size 8192)
if [[ ${1:-} == --265k ]]; then
    context=(--ctx-size 266240 --rope-scaling yarn --rope-scale 1.015625
             --yarn-orig-ctx 262144 --flash-attn on --batch-size 1024
             --ubatch-size 512 --no-context-shift)
    if [[ $packed == true ]]; then
        export GGML_VK_MLA_GROUP_QUERIES=8 GGML_VK_MLA_WAVE32=1 GGML_VK_MLA_MASK_OPT=1
        if [[ $mode == --edit ]]; then
            export LING_SPEC_DEVICE_CHECKPOINT=1
            export GGML_VK_MAX_NODES_PER_SUBMIT="${GGML_VK_MAX_NODES_PER_SUBMIT:-512}"
            if [[ ${LING_MLA_FP16_VERIFY:-1} == 0 ]]; then
                unset LING_MLA_FP16_VERIFY
            else
                export LING_MLA_FP16_VERIFY=1
            fi
            context+=(--threads 4)
            draft=(--spec-type ngram-simple)
            copy_width=7
            match_width=6
        else
            export LING_MTP_WINDOW=4096
            draft=(--spec-type draft-mtp --spec-draft-n-max 1)
        fi
    fi
    shift
fi
if [[ $mode == --edit ]]; then
    draft+=(--spec-ngram-simple-prompt-only --spec-ngram-simple-size-n "$match_width"
           --spec-ngram-simple-size-m "$copy_width")
fi

export LD_LIBRARY_PATH="$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$runtime/llama-server" \
    --model "$model" --alias ling-3.2-air \
    --host 127.0.0.1 --port 8088 \
    --device Vulkan0 -ngl 99 -np 1 "${context[@]}" \
    "${draft[@]}" --reasoning off "$@"
