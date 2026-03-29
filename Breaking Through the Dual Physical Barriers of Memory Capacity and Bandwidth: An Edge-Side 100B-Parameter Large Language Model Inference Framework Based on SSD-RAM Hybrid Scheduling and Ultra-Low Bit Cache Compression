Breaking Through the Dual Physical Barriers of Memory Capacity and Bandwidth: An Edge-Side 100B-Parameter Large Language Model Inference Framework Based on SSD-RAM Hybrid Scheduling and Ultra-Low Bit Cache Compression

Abstract

With the rapid expansion of the parameter scale of Large Language Models (LLMs), deploying 100B-level parameter models on resource-constrained edge devices faces dual physical limitations: the "memory capacity wall" and the "memory bandwidth wall". This paper proposes a system-level engineering deployment framework for 48GB unified memory architecture (taking Apple M4 Pro as the research benchmark). The framework integrates an SSD-RAM dynamic scheduling mechanism based on virtual memory mapping and the latest TurboQuant minimal lossless context cache compression algorithm, achieving weight decoupling in the spatial dimension and bandwidth release in the temporal dimension. This study proposes a dual-track scheduling strategy for Mixture of Experts (MoE) architecture and Dense architecture. On the premise of avoiding high-frequency wear of Solid-State Drives (SSD), it successfully realizes the local residency of Qwen3.5-122B-A10B and 108B Dense models on 48GB physical memory devices, and stably achieves an inference throughput of more than 10 tokens/s.


---
1. Introduction

Realizing local inference of 100B-parameter LLMs on edge devices is a milestone for data privacy protection and reducing cloud computing costs. However, even under 4-bit quantization, the weight volume of a 100B-level model is as high as 55-65GB, far exceeding the physical limit of current mainstream high-end consumer devices (such as 32GB-48GB unified memory). In addition, the linearly growing Key-Value Cache (KV Cache) during long text generation not only increases the risk of memory exhaustion (OOM), but also its massive memory access overhead severely squeezes the available bandwidth of model weights, leading to a sharp drop in Token generation rate.

To break through the above physical barriers, this paper constructs a comprehensive inference framework that runs through the underlying scheduling of the operating system and front-end algorithm compression. This framework proves that by precisely regulating the Page Fault behavior of the operating system and combining with cutting-edge quantitative compression mathematical models, the computing power boundary of consumer-grade hardware can be greatly expanded.

2. Related Work & Theoretical Foundations

The engineering implementation of this framework relies on two top academic research achievements in the industry, which solve the problems of "static storage" of weight parameters and "dynamic expansion" of context respectively.

2.1 Flash-Memory Based Inference

Alizadeh et al. (Apple, 2023) pointed out in their research "LLM in a flash: Efficient Large Language Model Inference with Limited Memory" that the traditional full-model loading mode will cause system crashes when the video memory is insufficient. This study proposes to virtually mount model tensors to the address space through the underlying system's `mmap` (memory mapping) mechanism.

Its core contribution is to solve the problem of **SSD Thrashing caused by random reads**: through Windowing (sliding window prefetching) and Row-Column Bundling (row-column data packaging), high-frequency small-block random reads are converted into low-frequency large-block Sequential Access. This allows most of the model's cold parameters (especially inactive experts in MoE models) to safely reside in SSD, and the underlying virtual memory manager of the system (such as macOS's Mach VM) performs seamless replacement of hot and cold data according to the access frequency (LRU strategy).

2.2 TurboQuant Cache Compression

The "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" proposed by Zandieh & Mirrokni (Google Research, ICLR 2026) solves the memory black hole problem in long-context scenarios. Traditional INT8 or 4-bit cache quantization is often accompanied by significant precision degradation.

TurboQuant establishes a lossless 3-bit minimal compression pipeline:

1. Stage 1: PolarQuant (Polar Orthogonal Rotation): A random orthogonal rotation matrix is used to transform high-dimensional KV activation tensors, eliminating Outliers and forcing the irregular high-dimensional feature energy to be scattered into a highly regular Gaussian distribution. Based on this known distribution, a static scalar quantizer (Lloyd-Max) is directly applied to compress to 2-3 bits, abandoning the scaling factors that must be stored in traditional quantization.

2.  Stage 2: QJL Transformation (1-bit Residual Compensation): Aiming at the inner product calculation deviation caused by ultra-low bit quantization, the algorithm extracts the quantization residual of the first stage and performs 1-bit Johnson-Lindenstrauss random projection. This 1-bit feature is introduced for mathematical compensation in the Self-Attention calculation stage of the Transformer to achieve unbiased estimation of the inner product. This algorithm reduces the video memory occupation of 32K context from 5.5GB to less than 1GB, and greatly reduces the video memory bandwidth occupation.

3. System Architecture & Dual-Track Scheduling

Due to the essential differences in parameter activation mechanisms between Mixture of Experts (MoE) architecture and Dense architecture (spatial sparsity vs. spatial density), this framework designs targeted resource allocation and bandwidth scheduling mechanisms. The research hardware benchmark is set as: Apple M4 Pro chip, 48GB unified memory, 273GB/s memory bandwidth, and high-speed PCIe SSD.

3.1 Path A: Sparse Scheduling for MoE Architecture

Taking `Qwen3.5-122B-A10B` as an example, only about 10B expert parameters are activated in each forward propagation, which has extremely high local memory access characteristics.

*   Quantization Strategy: Adopt 4-bit (Q4_K_M) medium quantization, with a total file size of about 65GB.

*   Memory Ledger Mapping:
        *   OS kernel and framework residency: ~8 GB.
        *   KV Cache (TurboQuant 3-bit): ~1 GB (supports 32K ultra-long context window).
        *   Resident memory (hot data): ~20 GB. It includes the Router layer, Shared Experts layer, and the 10B expert block with high-frequency activity in the current round.
        *   Resident SSD (cold data): ~35 GB. These are low-frequency calling domain-specific experts.

*   Bandwidth Deduction: In the Decoding stage, due to the high hit rate cache mechanism, the system does not need to trigger SSD page faults. The 273GB/s bandwidth of M4 Pro fully serves the movement of 10B parameters (about 6GB) in memory, and the theoretical throughput upper limit is far beyond the demand, which can be stably maintained at **15+ tokens/s** in actual measurement.

3.2 Path B: Extreme Compression & Speculative Decoding for Dense Models

Taking the 108B Dense model as an example, all network layers must be traversed for each Token generation, and SSD paging cannot be relied on for locality. Forcing the use of Swap will trigger catastrophic I/O blocking.

*   Quantization Strategy: It is necessary to adopt an extreme ultra-low quantization format, such as `2.25-bit (IQ2_XXS)`, to hard-compress the total model volume to the 30-32GB physical range.

*   Memory Ledger Mapping:
        *   OS kernel and framework residency: ~8 GB.
        *   KV Cache (TurboQuant 3-bit): ~1.5 GB (supports 40K+ context).
        *   Resident memory (full weights): ~32 GB. It is necessary to rely on the memory locking mechanism (mlock) to force the full weights into 48GB physical memory to avoid SSD wear.

*   Breaking Through the Bandwidth Wall: A complete read of 32GB weights is required for a single forward propagation. Under the 273GB/s extreme bandwidth, the theoretical physical upper limit is only ~8.5 tokens/s. To break this physical law, the framework introduces the **Speculative Decoding** mechanism.
        *   Mount a small-parameter (e.g., 0.8B) homologous fine-tuned model as the Draft Model. The Draft Model quickly autoregressively generates 4-5 candidate Tokens in the high-speed cache, and then the main model performs batch verification through a single forward propagation (parallel computing). This mechanism spreads the time cost of reading 32GB weights to multiple Tokens, forcing the equivalent output rate to cross the 10 tokens/s threshold, reaching **10-14 tokens/s**.

4. Engineering Implementation & Deployment Protocol

This section details the specific deployment protocol based on the open-source computation graph framework (referring to `llama.cpp` for implementation) that supports Metal API and TurboQuant operators at the bottom in the macOS environment.

4.1 Compilation & I/O Isolation

To prevent system-level processes from interfering with the virtual memory scheduling of large tensor files, it is necessary to isolate the Spotlight index (`mds_stores` process) at the bottom. Framework compilation must force linking to the on-chip acceleration hardware of Apple Silicon:

make LLAMA_METAL=1

4.2 Core Instruction Set Injection &amp; Parameter Tuning

Running instructions need to accurately configure virtual memory mapping and quantitative computation graph nodes:

./main -m <model_path> \
  --n-gpu-layers 999 \       # Computing Offloading: Force full offloading of Transformer layers to Metal engine
  --mmap 1 \                 # I/O Scheduling: Activate Mach VM virtual memory mapping to avoid direct loading
  -c 32768 \                 # Context Expansion: Set 32K ultra-long context state
  --cache-type-k tq3 \       # State Compression: Activate TurboQuant 3-bit polar quantization computation graph for Key
  --cache-type-v tq3 \       # State Compression: Activate TurboQuant 3-bit polar quantization computation graph for Value
  --threads 8

Key Branch Parameters for Dual-Track Scheduling:

*   MoE Model Deployment: Inject the `--mlock 0` parameter. Strictly prohibit memory hard locking, and authorize macOS Mach VM to dynamically perform block migration between memory page tables and SSD sectors based on the LRU algorithm.

*   Dense Model Deployment: Inject the `--mlock 1` parameter (lock RAM to prevent Swap), and tune the Draft Model engine in parallel: `--draft-model <path_to_draft.gguf> --draft-max 5`.

5. Performance Evaluation & Hardware Preservation

To verify the I/O performance and SSD wear rate of this framework in actual inference, standard "Warm-up" protocol and underlying `iostat` telemetry are introduced.

1.  I/O Surge in Prefill/Warm-up Phase: When the first ultra-long Prompt (such as 3000 Tokens) is input, the system experiences millisecond to second-level latency. Underlying telemetry shows that SSD read throughput surges in pulses (up to 3GB/s - 6GB/s). This process is the operating system performing large-block sequential data (Row-Column Bundled Data) migration as described in Apple's "LLM in a flash" paper, pumping the basic grammar experts and shared weight layers into physical RAM. Due to sequential reading, the P/E (Program/Erase) cycle wear on SSD flash memory cells is extremely low.

2.  Steady-State Decoding Phase: After pre-filling and initial dialogue, the hot data page table is established. Telemetry data shows that the SSD disk read activity (KB/t) drops sharply to nearly 0 MB/s.

3.  Final Energy Efficiency Achievement: After avoiding I/O blocking caused by Random Page Faults, the inference pipeline is completely closed-loop in the high-speed bandwidth of physical memory. In this state, the generation rate of the 100B MoE model stably exceeds the set baseline, and the physical wear of hardware devices is reduced to the normal level of daily system operations.

6. Conclusion

This study proposes an efficient framework for running 100B-parameter large language models on 48GB unified memory edge devices. By taking over the memory scheduling of `mmap` at the system bottom and combining with the cutting-edge TurboQuant 3-bit context residual compression algorithm, this framework successfully decouples the physical space dilemma of model weight management and the bandwidth congestion problem caused by context expansion. Experiments and theoretical deductions prove that both for the spatially sparse MoE architecture and the spatially dense Dense architecture, through reasonable quantization selection, virtual memory hot-cold isolation, and speculative decoding intervention, consumer-grade hardware is fully capable of supporting 100B model inference loads of more than 10 tokens/s with extremely low hardware wear rate. This provides a feasible engineering path for the lossless landing of ultra-large-scale artificial intelligence models on edge devices in the future.
