![Laengine v0.1 banner featuring Max Stirner in neon sunglasses](laengine.svg)

# Laengine v0.1

Laengine is a specialized llama.cpp Vulkan runtime for **Ling-3.2-air** on **AMD Strix Halo**. It serves the text model through a local OpenAI-compatible API. The reference platform is a Ryzen AI Max+ 395 with Radeon 8060S graphics and 128 GB of unified memory, running CachyOS with Mesa RADV.

On that machine, a cached **196,840-token** prompt yielded **60.5 tokens/s median** across three complete-file edits and **61.1 tokens/s** on the fastest trial. Fresh prose at about the same depth ran at **33.0 tokens/s**. [Benchmark details and reproduction](BENCHMARKS.md)

## Build and run

You need Linux x86-64, a Vulkan-capable driver, Git, CMake, a C/C++ toolchain, Vulkan headers and loader, SPIRV-Headers, and `glslc` from Shaderc. The tested build used GCC 16.2.1 and Mesa RADV 26.2.6. The 67.4 GiB packed weights and runtime need substantial GPU-accessible memory. The 128 GB unified-memory configuration is the tested target.

```bash
git clone https://github.com/lassebindependentresearch-byte/Laengine.git
cd Laengine
bash build.sh
```

The script fetches the [pinned llama.cpp revision](https://github.com/ggml-org/llama.cpp/commit/52975342ff6b85127ff31930d39e36725fd08809), checks and applies the [Laengine patch](laengine-v0.1.patch), and builds the server and model quantizer. Set `LAENGINE_JOBS` to change the build parallelism.

### Prepare the model

Place an existing Laengine packed GGUF at `models/Ling-3.2-air-IQ4_NL-Packed-Conv.gguf`, or build it locally from the official [Ling-3.2-air checkpoint](https://huggingface.co/inclusionAI/Ling-3.2-air/tree/1613b1a4dcc6aa0d8f346606628cd9a92061a525). The repository contains the runtime source patch and conversion scripts. Download the weights from InclusionAI.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r .llama.cpp/requirements/requirements-convert_hf_to_gguf.txt huggingface_hub

hf download inclusionAI/Ling-3.2-air \
  --revision 1613b1a4dcc6aa0d8f346606628cd9a92061a525 \
  --local-dir models/ling-hf
bash prepare-model.sh models/ling-hf
```

The source snapshot is about 121 GiB. Plan for about **650 GiB of free disk space** when downloading and converting on one volume. The BF16 intermediate is about 238 GiB and the converter uses disk-backed temporary tensors. The conversion keeps the built-in MTP head, quantizes to IQ4_NL, and packs the recurrent projections and convolutions without requantizing them.

### Start serving

| Command | Use |
| --- | --- |
| `bash Start-Ling.sh` | 8,192-token chat and new code |
| `bash Start-Ling.sh --edit` | 8,192-token complete-file edits with prompt lookup |
| `bash Start-Ling.sh --265k` | Long-context chat and new text |
| `bash Start-Ling.sh --edit --265k` | Long-context edits that return most of the original file |

Set `LAENGINE_MODEL` to use a model at another path. The server defaults to `Vulkan0`, one request slot, all layers on the GPU, reasoning off, and loopback port 8088. Extra llama-server arguments go after the profile flags.

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ling-3.2-air","messages":[{"role":"user","content":"Reply with READY."}]}'
```

## Engine specifications

| Component | Configuration |
| --- | --- |
| Model architecture | 124B total, 5.5B active. 35 KDA and seven gated MLA layers, plus one NextN draft layer. 512 routed experts, eight active per token, one shared expert. |
| Dimensions | Hidden width 2,560, 32 attention heads, 157,184-token vocabulary. |
| Weights | GGUF v3 IQ4_NL with mixed tensor types selected by the upstream quantizer. Packed model size 72.4 GB, or 67.4 GiB. |
| Vulkan path | Packs five recurrent projections and three convolutions per KDA layer. Groups MLA query heads, uses 32-lane subgroups where supported, and skips mask work for uniform blocks. |
| Long context | 266,240 tokens total, including the response. YaRN factor 1.015625 from the model's 262,144-token window, with context shifting disabled. FP16 KV cache, Flash Attention enabled, batch 1,024, microbatch 512. |
| Long edit profile | Drafts up to seven tokens from a matching prompt suffix. Speculative rollback snapshots stay on the GPU. Verification batches of five to eight tokens use FP16 attention accumulation. |
| Long chat profile | One MTP draft token with a 4,096-token draft window. The main model verifies proposals using its full context. |
| Serving | OpenAI-compatible Chat Completions, streaming and text tool calls. The release is tested for text. |

Set `LING_MLA_FP16_VERIFY=0` before launching the long edit profile to use FP32 accumulation for all verification batches.

## Hermes Desktop

Under **Settings → Providers → Custom Endpoints**, select **Chat Completions**, enter `http://127.0.0.1:8088/v1`, and choose `ling-3.2-air`. If a key is required, enter `local-only`. Start the server on the same machine as the Hermes backend.

The [example profile](Hermes-Ling.example.yaml) matches the long-context launcher, requests prefill progress, and disables automatic summarization. Set `HERMES_STREAM_READ_TIMEOUT=5400` in that profile's `.env` to allow for long cold prefills. The context allocation includes the system prompt, tool definitions, and generated response. A chat and a real memory-tool call passed with the installed Hermes agent.

## Credits

Built on [llama.cpp](https://github.com/ggml-org/llama.cpp), distributed under the [MIT license](LICENSE). The [Ling-3.2-air model card](https://huggingface.co/inclusionAI/Ling-3.2-air) provides the weights, architecture, and upstream license. The SVG banner uses the project owner's supplied Stirner artwork.
