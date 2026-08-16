# lean-infer

Inference engine with

- Continuous Batching and Paged Attention
- Optimizations: CUDA Graphs
- Observability: Prometheus + Grafana

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

SDPA had silently fallen back to the MATH backend (`bmm`, `mul`, `softmax` kernels instead of one cuDNN fused kernel). I had been prototyping the model in fp32 on cpu but after I switched over to a cloud GPU, I didn't realize I needed to swap to bf16/fp16 to use fused kernels.

<p align="center">
<img src="assets/phase2_trace.png" width="450">
</p>

Once cuDNN SDPA was working, attention got fast enough that it became host-bound with the GPU being mostly idle. CUDA Graphs was built for this regime, so I implemented that next.

![Comparison](assets/phase2.5_comparison.png)

**Results (in bf16)**: A 9.8x faster decode step and improved decode throughput and wall clock time. Note: the change to bf16 halved the bandwidth floor (~10 ms), so we aren't quite comparing apples to apples.

|                          | before    | after       |     |
| ------------------------ | --------- | ----------- | --- |
| decode step              | 165 ms    | 16.9 ms     |     |
| prefill step             | 40 ms     | 40 ms       |     |
| decode throughput        | 188 tok/s | 1,470 tok/s |     |
| TPOT                     | 170 ms    | 21.8 ms     |     |
| 1024 requests wall clock | 23 min    | 3.0 min     |     |
|                          |           |             |     |

---

#### PagedAttention and Workload Gaps

I changed the workload since prefill had been below the L4's ridge point (~400 token prompt length) so far, making it just another memory-bound workload. Both prompt length and max output tokens were now uniformly distributed in `[512, 1024]` (which explains the periodicity of prefill throughput in the panel below), and I also bumped batch size to 64 requests.

< add the stats of old vs new regime>
floor N=128 (old) N=768 (new)
compute 1.28 ms 8.12 ms
memory 4.02 ms 4.27 ms
binding memory compute
TODO: need some proofs for this data

> Note: The traditional characterization is that prefill is compute-bound while decode is memory-bound. I learned later this doesn't always hold true...\* _cue chunked prefill_\*. Until I build that alongside a unified token budget, I'll retain the same synthetic workload since prefill (set at B=1) will mostly be below the ridge point if using a real dataset like ShareGPT.

<p align="center">
<img src="assets/pre phase 3 change prompt 1 compilation.png">
</p>

KV cache utilization (~60%) is the next target for improvement, which is accomplished with paged KV cache. Keeping total kv memory constant, the previous 64 slots \* 2048 slot_length now became 512 num_blocks \* 256 block_size. Rather than hand-write a paged KV kernel, I chose to use Flash Attention's flash_attn_with_kvcache. However, this constrained my block size options since FA needed multiples of 256. I didn't have to think about the KV read path but paid for it in more internal fragmentation compared to, say, vLLM's default of 16. I also bumped batch size to 128 to accommodate more requests; 128 is quite high though and I always pay a small CUDA Graph cost in capturing launches over all 128 requests, real or idle.

<p align="center">
<img src="assets/phase 3 uniform distribution batch compilation.png">
</p>

The panels show "numbers go up" but there's a ton of nuance here:

- Honesty where honesty is due: The improvements to throughput (prefill; Panel 2 yellow) and prefill time (Panel 4) are almost completely attributed to FlashAttention.
- The improvements to KV cache utilization (Panel 1), throughput (decode; Panel 2 green) and concurrency (Panel 3) are attributed to the Paged KV cache I built. This represents a particular ideal setup: where Goodput is 100% of the work done, i.e. no request or token ever generated by the engine is ever thrown away (no preemptions). The obvious cost for 100\% goodput is lower concurrency since I have to allow some blocks remain idle until they are eventually filled

I chose to demonstrate a 100% Goodput baseline by reserving every running request's full expected generation length. I forced this because my workload is predictable (uniform in [512, 1024]) and garbage (vocab IDs randomly generated), so a request will almost never emit a stop token and will exhaust its token budget. Any utilization knob I turn in this regime is just cherry picked to be flattering. Real workloads have an unbounded generation length (capped by an engine's max_new_token policy) that the scheduler can't reliably predict.

Future work is to accommodate for that with an admission-backoff mechanism when some metric (number of preemptions or goodput) reaches a tolerance limit.
