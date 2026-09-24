# GLM-5.3-Flash on one GPU + 64 GB RAM: MoE expert streaming from NVMe (llama.cpp patch)

> **Experimental and AI-written.** Claude (Anthropic) wrote this code while I directed it. I'm not a
> C++/CUDA developer and haven't reviewed it line by line. I only checked the results (see
> [How it was checked](#how-it-was-checked)), and only on one machine. It is not meant to go upstream
> in this form. Use at your own risk.

A single patch for Unsloth's llama.cpp fork. GLM-5.3 isn't in llama.cpp master yet, so the base is
[`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp), branch `glm5next/upstream`, commit
`86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`.

The model (UD-IQ4_XS, 148 GB) is more than twice the RAM. Instead of letting mmap thrash the page cache,
the routed experts are streamed from two NVMe SSDs.

## What's in the patch

20 files, +2611 / −15 lines. It has two parts:

**1. [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) "llama : stream MoE routed experts from disk" by
[@freedomljc](https://github.com/freedomljc), ported onto the glm5next branch** (~1.45k lines, their work, still
an open PR upstream). This provides the core: an expert cache with O_DIRECT reads on cache misses and an async I/O pool.

**2. Changes on top of it** (~1.2k lines):

- **Cache follows `--n-cpu-moe`.** Only the CPU-offloaded layers are streamed through a host-RAM cache. The
  other layers stay in VRAM.
- **Full-layer staging for prefill.** For large batches, the whole expert block of the next layer (~3.3 GiB) is
  read into one of two pinned host buffers while the GPU computes the current layer. On this branch, the
  PR's own wave-partitioned prefill gave wrong output when combined with op offload, so staging replaces it here.
- **Second SSD as a mirror.** A byte-identical copy of the GGUF shards on another disk. Staging and decode
  reads pull from a shared queue per disk, so each disk takes what it can deliver.
- Prefill copies experts that are already in the RAM cache instead of reading them again.
- Decode prefetches the predicted top-N experts of the next layer (only from the mirror disk, see limitations).
- Streaming statistics in the server log.
- **CUDA fix (6 lines, `ggml/src/ggml-cuda/mmq.cu`).** In MMQ `MUL_MAT_ID` with a shared gate/up input, the src1
  padding was sized from `ne11` (= 1) instead of the flattened column count. The kernel then reads past the end of
  the buffer. With `-ub 2048` that buffer can end exactly at a VMM mapping boundary, which causes an illegal
  memory access (found with `compute-sanitizer`). This *might* be related to
  [#28282](https://github.com/ggml-org/llama.cpp/issues/28282) (same symptom, same model and GPU family), but the
  reporter there suspects a different cause and I haven't verified that it's the same bug.

## Tested setup

- Ryzen 7 9800X3D, 64 GB DDR5, RTX 5090 32 GB
- Crucial T700 4 TB (PCIe 5.0, LUKS-encrypted) holding the model, Kingston 4 TB (PCIe 4.0, unencrypted) holding the mirror copy
- Linux (Pop!_OS, kernel 7.1), NVIDIA driver 580.173, CUDA 13.0
- Model: Unsloth GLM-5.3-Flash UD-IQ4_XS (5 shards, 148 GB)

## Build

```bash
git clone https://github.com/unslothai/llama.cpp
cd llama.cpp
git checkout 86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9
git apply /path/to/glm5next-moe-ssd-streaming.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```

## Run (my settings)

```bash
export NVIDIA_TF32_OVERRIDE=0                         # recommended in PR #27754 for correct output
export LLAMA_MOE_STREAM_MIRROR=/path/to/mirror-dir    # same shard file names as the model dir; omit if you have one disk
export LLAMA_MOE_STREAM_MIRROR_THREADS=6              # staging readers serving the mirror (of 16)
export LLAMA_MOE_STREAM_MIRROR_DECODE=2               # decode reads use the mirror too
export GGML_CUDA_REGISTER_HOST=1 GGML_CUDA_REGISTER_HOST_RW=1   # pin the staging buffers
export LLAMA_MOE_STREAM_PREFETCH=4                    # next-layer expert prefetch (0 = off)
export LLAMA_MOE_STREAM_STATS=1                       # streaming stats in the log

./build/bin/llama-server \
  --model /path/to/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf --no-mmproj \
  -ngl 99 --n-cpu-moe 42 --load-mode none \
  --moe-stream-cache 32 --moe-stream-direct --moe-stream-io-threads 16 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --ctx-size 262144 --parallel 1 --batch-size 2048 --ubatch-size 2048 --threads 8 \
  --cache-ram 2048 --ctx-checkpoints 8 \
  --jinja --reasoning-format deepseek \
  --chat-template-kwargs '{"reasoning_effort":"high","clear_thinking":true}' \
  --temp 1.0 --top-p 0.95
```

`--moe-stream-cache 32` is the expert cache in GiB.

## Results (measured 2026-09-24)

Same 7,943-token prompt on both builds:

| | Prefill | Decode |
|---|---|---|
| Unpatched glm5next build, mmap + `--fit on`, `-ub 1024` | 29 t/s | 2.4–2.9 t/s |
| This patch | 240 t/s | 5.8–6.0 t/s |

A real agentic run (Qwen Code in plan mode: it read 9 files and wrote an implementation plan):

- Context grew from 11.5k to 35k tokens; 31k new prompt tokens at ~230 t/s (220–260 on large chunks)
- 3.7k generated tokens (including thinking) at 5.6 t/s; 13 min total
- 78 % of expert lookups hit the RAM cache; misses cost ~1.3 GB per token from the SSDs (~18 GB/s combined)
- Each 2048-token prefill batch reads ~99 GB from disk (~13 GB/s) plus ~34 GB from the RAM cache
- The server process alone uses ~29–30 GB VRAM and ~43–46 GB RAM

## How it was checked

- Greedy outputs are bit-identical to the unpatched mmap build at the same ubatch size on my regression
  prompts. A different ubatch size changes the rounding slightly, as usual.
- Staged expert data was byte-compared against the GGUF file: ~150 layer loads, 0 mismatches.
- Tool calls and reasoning output work in real agent sessions.
- The code was reviewed only by AI (Claude), not by a human.

## Known limitations

- Linux only (O_DIRECT, pinned host memory). Tested only with CUDA on one RTX 5090.
- **VRAM is tight with `-ub 2048`.** If the desktop and other apps use more than ~2.5 GB VRAM, loading fails
  with `cudaMalloc failed: out of memory`. Fallback: `--batch-size 1024 --ubatch-size 1024`, which roughly
  halves prefill speed.
- **dm-crypt/LUKS.** O_DIRECT reads from an encrypted disk are decrypted *inside the destination buffer*, so
  nothing else may write into the same 4 KiB block while a read is in flight. The code handles this, but keep
  it in mind if you change the staging code.
- Prefetch reads only from the unencrypted mirror disk. On the encrypted disk, decryption competes with the CPU
  expert compute and made decode slower.
- The mirror is not verified at startup. It must be byte-identical to the model shards.
- About 40 % of each decode token is spent waiting on the disks and most of the rest on CPU expert compute,
  so more RAM (a higher cache hit rate) is still the biggest lever for decode speed.

## Credits and license

- [llama.cpp](https://github.com/ggml-org/llama.cpp) by the ggml authors (MIT)
- The glm5next branch by [Unsloth](https://github.com/unslothai/llama.cpp)
- The MoE streaming core: [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) by [@freedomljc](https://github.com/freedomljc)

The patch is a derivative of llama.cpp and is released under the same MIT license (see [LICENSE](LICENSE), unchanged from llama.cpp).
