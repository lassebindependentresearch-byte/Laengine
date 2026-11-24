# Benchmark results

Tested on Ryzen AI Max+ 395, Radeon 8060S, 128 GB unified memory, CachyOS Linux 7.3.3, Mesa RADV 26.2.6. The model was the packed IQ4_NL build, served with one slot. Rates measure output generation and exclude prompt processing and startup.

| Workload | Input tokens | Output tokens | Generation |
| --- | ---: | ---: | ---: |
| Short full-file edit | 385 | 336 | 158.0 tok/s |
| Short full-file edit with repeated code | 1,071 | 1,022 | 193.1 tok/s |
| Full-file edit, warm prompt, three runs | 196,840 | 1,036 | 60.3, 60.5, 61.1 tok/s |
| New prose, warm shared prefix | 196,905 | 394 | 33.0 tok/s |
| Full-file edit, warm prompt | 264,970 | 1,061 | 50.6 tok/s |
| New prose, warm shared prefix | 265,035 | 401 | 29.3 tok/s |

The full-file edit test compares the returned Python syntax tree to the expected edit. Long runs also check that the answer recovers a seal near the start of the prompt. Prompt lookup helps when most of the output repeats the input file. The short-edit figures are measurements of that workload.

An earlier cold prefill of the 265K test took about 48 minutes on this hardware. The identical warm edit started streaming in about 0.3 seconds.

## Reproduce

Start `bash Start-Ling.sh --edit --265k` from the repo root. Run the first command to prefill and test the model, then run the second for the cached measurement.

```bash
LING_DEPTH=196865 python3 bench-long-context.py
LING_DEPTH=196865 LING_CACHE_PROMPT=1 python3 bench-long-context.py
```

Use `LING_DEPTH=265000` for the longer case. Add `LING_LONG_TASK=chat` for a new-prose request. It appends a new instruction to the archive prompt. For the ordinary chat profile, start `bash Start-Ling.sh --265k` and populate its cache with the archive request before measuring the prose variant. The script emits JSON Lines with prompt counts, time to first token, cache usage, generation timing, and validation results.

For short edits, launch `bash Start-Ling.sh --edit` and run `python3 bench-edit.py`. Set `LING_EDIT_COPIES=3` for the repeated-code workload.
