> [!NOTE]
> **This branch: Flash-Next (qwen4exp) MTP speculative decoding.** Adds working
> NextN/MTP speculative decode for Qwen3.8-Flash-Next models plus QSA/PLE decode
> performance paths and server checkpoint/MTP-rollback fixes. See
> [FLASHNEXT_MTP.md](./FLASHNEXT_MTP.md) for features, measured numbers, and usage, and the
> [full testing &amp; methodology documentation](https://noonr48.github.io/llama.cpp/flashnext-mtp/)
> for how the tested model was built and verified. Runs everything stock llama.cpp runs; the
> [tested model](https://huggingface.co/jackasda211233/Qwen3.8-Flash-Next-Uncensored-IQ4_NL) is on Hugging Face.

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
qwenllama: Streamed MoE Prefill for Qwen Flash-Next

This fork adds a streamed-expert prefill path for Qwen Flash-Next / "qwen4exp" models in "llama.cpp".

The goal is simple: when MoE expert weights do not fit in VRAM and would normally be evaluated from system RAM on the CPU, temporarily stream those weights into GPU memory and execute the expert GEMMs on the GPU instead.

On the target multi-GPU system, this moved large cold-prefill throughput into the roughly 400 tokens/sec class.

The recent work consists of two closely related changes:

1. Stream host-resident MoE expert weights through temporary VRAM arenas and run their GEMMs on GPU.
2. Reduce QSA prefill memory pressure so larger micro-batches can fit and amortize the expert-transfer cost.

The problem

Qwen Flash-Next is a sparse Mixture-of-Experts model.

When the model is too large to keep all expert tensors in VRAM, llama.cpp can leave some expert weights in system RAM. The resulting execution path can make the CPU responsible for those MoE operations.

For large prefills, that becomes a major bottleneck:

host-resident experts
        |
        v
   CPU MoE GEMMs
        |
        v
   GPU continues

The GPUs may have substantial unused compute capacity while the CPU and host memory subsystem determine overall prefill speed.

The streamed-expert path changes that.

Streamed expert prefill

With streamed experts enabled, system RAM remains the authoritative storage for offloaded expert weights, but the expert computation itself is moved back to CUDA.

system RAM
   expert weights
        |
        | PCIe H2D
        v
temporary VRAM arena
        |
        v
GPU expert GEMMs

The model still does not need enough VRAM to hold every expert layer permanently.

Instead, the runtime allocates a small ring of temporary GPU buffers called expert arena slots.

For example, with three slots:

GPU arena:

slot 0 -> layer N expert image
slot 1 -> layer N+1 expert image
slot 2 -> layer N+2 expert image

Once layer "N" finishes using slot 0, that slot becomes available for layer "N+3".

Conceptually:

time ------------------------------------------------------------>

GPU:
        compute L0      compute L1      compute L2      compute L3
           |               |               |               |

PCIe:
        copy L1         copy L2         copy L3         copy L4

Transfers and computation are therefore pipelined rather than serialized.

Why whole expert layers are streamed

The current implementation streams the complete expert image for an MoE layer:

ffn_up_exps
ffn_gate_exps
ffn_down_exps

It does not currently fetch individual routed experts on demand.

That is intentional.

For a large prefill micro-batch, hundreds or thousands of tokens collectively route to most or essentially all experts in the layer. Fine-grained per-expert streaming therefore provides much less benefit than it would during single-token decoding.

The relevant cost becomes approximately:

one full expert-layer transfer
------------------------------
tokens in the micro-batch

A larger physical micro-batch therefore strongly improves amortization.

For example:

one layer transfer / 256 tokens

vs.

one layer transfer / 2048 tokens

The second case amortizes approximately the same expert-weight traffic across eight times as many tokens.

This path is intended for large prefills, not token-by-token generation.

VRAM arena ring

Each CUDA device receives a configurable number of arena slots.

The planner assigns streamed layers to slots using a ring:

slot = streamed_layer_ordinal % number_of_slots

With three slots:

L0 -> slot 0
L1 -> slot 1
L2 -> slot 2
L3 -> slot 0
L4 -> slot 1
L5 -> slot 2
...

The first set of slots is filled before graph execution begins.

When a layer finishes, its post-fence releases the corresponding slot and queues the next layer image that will use it.

The arena therefore only needs enough VRAM for a small number of layer images rather than the complete set of offloaded experts.

The slot size is determined by the largest eligible streamed layer.

Asynchronous CUDA pipeline

The implementation uses separate compute and transfer machinery:

CUDA compute stream
    normal model execution

CUDA copy stream
    host -> device expert transfers

host worker thread
    staging and transfer scheduling

Each arena slot tracks two important CUDA events:

ev_filled
    the required expert image is ready in the slot

ev_used
    GPU compute has finished reading the slot

Before an expert operation runs, a pre-fence orders the compute stream behind the corresponding fill.

After the expert operation finishes, a post-fence marks the slot reusable and schedules the next fill.

This prevents a new expert image from overwriting an arena slot while a CUDA kernel is still reading it.

The implementation also maintains explicit release credits for reused slots rather than relying only on CUDA event state.

If an expert fill fails, the engine is poisoned and execution aborts before a partially updated expert tensor can be consumed.

Pinned host staging

Expert tensors normally originate in pageable system memory.

The streamer can stage them through pinned host memory:

pageable model memory
        |
        | CPU memcpy
        v
pinned staging buffer
        |
        | async PCIe DMA
        v
GPU arena

Large tensor families are divided into chunks.

This permits the CPU to stage the next chunk while CUDA transfers the previous chunk:

CPU staging:   [ chunk 1 ][ chunk 2 ][ chunk 3 ]
PCIe DMA:                 [ chunk 1 ][ chunk 2 ][ chunk 3 ]

Host copies are also parallelized across multiple CPU threads for sufficiently large transfers.

Pinned staging is enabled by default for the streamed-expert path.

QSA prefill memory fixes

Moving MoE GEMMs back to the GPU only helps if sufficiently large micro-batches can fit in VRAM.

QSA prefill was another source of substantial peak memory usage.

Previously, the query-side QSA score graph could be constructed across the entire physical micro-batch at once:

context blocks x all query tokens

At large contexts and large micro-batches this can become a very large intermediate tensor.

The new code supports query chunking through:

QWEN4EXP_QSA_QUERY_CHUNK

Instead of one monolithic query-side calculation:

all queries
    |
    v
huge QSA score graph

the work can be split:

queries   0..255
queries 256..511
queries 512..767
...

Each chunk is expanded into the graph before the next large score matrix is constructed.

This reduces peak graph memory and makes larger prefill micro-batches practical.

The QSA input path also reuses mask and bias slice identities so equivalent slices are not repeatedly uploaded as distinct graph inputs.

These changes are primarily memory optimizations, but they are important to throughput because streamed-expert efficiency depends heavily on large micro-batches.

Combined effect

The performance path is therefore:

                    OLD

host-resident MoE experts
          |
          v
      CPU GEMMs
          |
          v
 CPU / memory bottleneck
          |
          v
      slow prefill

versus:

                    NEW

host-resident MoE experts
          |
          v
pinned host staging
          |
          | async H2D
          v
rotating VRAM arena <-------------------+
          |                             |
          v                             |
     GPU MoE GEMMs                      |
          |                             |
          +---- release slot -----------+

while QSA is separately chunked to keep peak VRAM usage under control.

In practice:

QSA memory reduction
        |
        v
larger physical ubatch fits
        |
        v
expert transfer amortized over more tokens
        |
        v
GPU expert GEMMs stay productive
        |
        v
~400 tok/s-class large-prefill throughput
on the target system

Runtime options

Enable expert streaming:

--fn-stream-experts

Number of arena slots per GPU:

--fn-stream-slots N

The default is 3.

Two or more slots allow expert transfers to overlap execution.

Minimum micro-batch size that uses streaming:

--fn-stream-min-tokens N

The default is 1024 tokens.

Smaller micro-batches stay on the normal path.

Enable or disable pinned staging:

--fn-stream-pin
--no-fn-stream-pin

Set a per-GPU arena VRAM ceiling:

--fn-stream-budget-mib N

"0" selects the budget automatically from available VRAM.

The automatic path reserves additional memory for the rest of the graph rather than consuming all reported free VRAM.

Multi-GPU arena placement

The arena does not necessarily have to live on the same GPU as the rest of a layer.

Three placement modes are available.

Layer-local

--fn-stream-gpu-mode layer

Each expert arena lives on the device that owns the corresponding layer.

This is the default.

Primary GPU

--fn-stream-gpu-mode primary
--fn-stream-arena-device N

Streamed experts can be concentrated on one GPU.

The expert GEMM then executes on that GPU.

Instead of moving the entire expert image between GPUs, the smaller activation tensors cross the device boundary.

Weighted split

--fn-stream-gpu-mode split
--fn-stream-gpu-split W0,W1,...

Streamed expert layers are distributed across CUDA devices according to byte weights.

For example:

--fn-stream-gpu-split 4,1

targets roughly four times as much streamed expert traffic at device 0 as device 1.

QSA query chunking

The QSA prefill query chunk can be limited with:

QWEN4EXP_QSA_QUERY_CHUNK=256

The appropriate value depends on context size, physical micro-batch size, and available VRAM.

This is primarily a memory-control mechanism rather than a direct arithmetic optimization.

MTP micro-batch limit

The same recent patch also adds:

LLAMA_MTP_UBATCH=N

which allows the MTP context to use a lower physical micro-batch limit than the target context when needed for memory control.

This is separate from the streamed-expert speedup itself.

Eligibility

A layer is currently streamable when its expert tensors satisfy the implementation's supported layout.

In particular, the current path expects:

- separate up, gate, and down expert tensor families,
- all three families to be present,
- host-resident expert storage,
- contiguous expert tensors,
- no unsupported per-expert bias tensors,
- no unsupported per-expert scale tensors,
- no fused gate/up expert representation,
- and a CUDA device associated with the layer.

Experts already resident on the GPU are left alone.

When streaming cannot be applied, the code falls back to the existing path rather than constructing a partial arena configuration.

Current limitation

The streamer operates at whole-MoE-layer granularity.

An arena slot must therefore be large enough to hold the largest complete streamed expert layer:

up experts
+ gate experts
+ down experts
+ backend alignment / padding

This can still require a substantial amount of temporary VRAM.

A future per-expert or grouped-expert streamer could reduce the minimum arena size and potentially reduce PCIe traffic for smaller batches, but the present design is deliberately optimized around large prefills where nearly all experts are active across the batch.

Telemetry

The streamer reports per-micro-batch statistics including:

expert fills
H2D bytes
staging bytes
pre/post fence counts
slot wait time
host staging time
copy submission time
errors
number of devices
achieved H2D bandwidth

This makes it possible to distinguish whether a run is limited by:

PCIe bandwidth
host staging bandwidth
arena starvation
GPU expert compute
or other model work

rather than inferring the bottleneck from total tokens/sec alone.

Summary

The recent prefill optimization does not make the model smaller and does not permanently move the offloaded experts into VRAM.

It changes how offloaded MoE computation is executed.

Instead of:

system RAM weights -> CPU expert computation

it uses:

system RAM weights
        ->
asynchronous streamed layer image
        ->
temporary VRAM arena
        ->
GPU expert computation

with transfers pipelined against GPU execution.

At the same time, QSA prefill is chunked to reduce peak memory so the large physical micro-batches needed to amortize those transfers can actually fit.

That combination is what moved the target system into the approximately 400 tok/s prefill range..md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
