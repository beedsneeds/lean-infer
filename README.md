# lean-infer

Inference engine with

- Continuous Batching and Paged Attention
- Optimizations: CUDA Graphs
- Observability stack (Prometheus + Grafana)

Developed and benchmarked on an NVIDIA L4 (GCP) running the Deep Learning VM stack, using a saturated, offline batch.

## Getting Started

Run the engine standalone with a synthetic batch. Turn on observability for the interesting stuff.

```
uv run leaninfer
```

## Observability

Beyond what GCP's Deep Learning VM stack provides, you'll need

```bash
# Docker Compose to run the metrics stack
sudo apt install docker.io docker-compose-v2

# (Optional) If you need GPU metrics,
# 1. configure the nvidia runtime so dcgm-exporter can see the GPU
sudo nvidia-ctk runtime configure --runtime=docker
# 2. Copy over the `.env.example` with no changes. Just makes docker compose commands cleaner.
cp .env.example .env
# 3. load the nvidia runtime
sudo systemctl restart docker
sudo usermod -aG docker $USER
# Then reconnect to the instance after usermod
```

Start Prometheus, Grafana, and dcgm-exporter.

```
docker compose up -d
```

On your computer, set up a tunnel to access the dashboards locally: Grafana at localhost:3000, Prometheus at localhost:9090

```
# This is the only thing you run locally, everything else should be on L4
gcloud compute ssh --zone <ZONE> <INSTANCE> \
 --project <PROJECT> \
 -- -N -L 3000:localhost:3000 -L 9090:localhost:9090 -L 9400:localhost:9400
```

Run the engine and check `localhost:3000` and `localhost:9090/targets`

```
uv run leaninfer
```

## Design and Challenges

#### 10x Faster Decode

Once I built observability and continuous batching, I saw both decode and prefill were much worse than the roofline numbers. Since decode was ~97% of wall-clock time and 8× off its bandwidth floor (~20 ms), I prioritized that.

SDPA had silently fallen back to the MATH backend (`bmm`, `mul`, `softmax` kernels instead of one cuDNN fused kernel). I had been prototyping the model in fp32 on cpu but to use fused kernels, I didn't realize I needed to swap to bf16/fp16 when I switched over to a cloud GPU.

<p align="center">
<img src="assets/phase2_trace.png" width="450">
</p>

Once cuDNN SDPA was working, attention got fast enough that it became host-bound with the GPU being mostly idle. CUDA Graphs was built for this regime.

![Comparison](assets/phase2.5_comparison.png)

**Results (in bf16)**: A 9.8x faster decode step and improved decode throughput and wall clock time. Note: the change to bf16 halved the bandwidth floor (~10 ms).

|                          | before    | after       |     |
| ------------------------ | --------- | ----------- | --- |
| decode step              | 165 ms    | 16.9 ms     |     |
| prefill step             | 40 ms     | 40 ms       |     |
| decode throughput        | 188 tok/s | 1,470 tok/s |     |
| TPOT                     | 170 ms    | 21.8 ms     |     |
| 1024 requests wall clock | 23 min    | 3.0 min     |     |
|                          |           |             |     |

---

#### PagedAttention

I changed the workload since prefill had been below the L4's ridge point (~400 token prompt length) so far, making it just another memory-bound workload. Both prompt length and max output tokens were now uniformly distributed in `[512, 1024]` (which explains the periodicity of prefill throughput), and I also bumped batch size to 64 requests.

< add the stats of old vs new regime>

> Note: The traditional characterization is that prefill is compute-bound while decode is memory-bound. I learned later this doesn't always hold true...\* _cue chunked prefill_\*. Until I build that/unified token budget, I'll retain the same synthetic workload since prefill will mostly be below the ridge point if using a real dataset like ShareGPT with my current prefill at B=1.

<p align="center">
<img src="assets/pre phase 3 change prompt 1 compilation.png">
</p>

KV cache utilization (~60%) could be improved, so I swapped to paged kv cache while keeping total kv memory the same (64 slots \* 2048 slot_length now became 256 block_size \* 512 num_blocks). sing Flash Attentions flash_attn_with_kvcache constrained my block size options since it needed multiples of 256. I didn't have to think about the read path but paid for it in more internal fragmentation compared to, say, vLLM's default of 16.
