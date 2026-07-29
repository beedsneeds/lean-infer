# lean-infer

WIP inference engine. Currently having continuous batching running with observability plugged in.

Requires an NVIDIA GPU. Developed on a GCP L4 running the Deep Learning VM stack image.

Run the engine standalone with a synthetic batch. Prints tok/s, no other metrics tracked.

```
uv run leaninfer
```

## Observability

Beyond what GCP's Deep Learning VM stack provides, you'll need

```bash
# Docker Compose to run the metrics stack
sudo apt install docker.io docker-compose-v2
# To configure the nvidia runtime so dcgm-exporter can see the GPU
sudo nvidia-ctk runtime configure --runtime=docker
# To load the nvidia runtime; reconnect to the instance after usermod
sudo systemctl restart docker
sudo usermod -aG docker $USER
```

Copy over the `.env.example` with no changes. Just makes docker compose commands cleaner.
Skip if you don't want dcgm-exporter scraping GPU metrics

```
cp .env.example .env
```

Start Prometheus, Grafana, and dcgm-exporter. Dry run with `config` to catch any YAML typos.

```
docker compose config >/dev/null && echo OK
docker compose up -d
```

On your computer, set up a tunnel to access the dashboards locally: Grafana at localhost:3000, Prometheus at localhost:9090

```
# This is the only thing you run locally, everything else should be on L4
gcloud compute ssh --zone <ZONE> <INSTANCE> \
 --project <PROJECT> \
 -- -N -L 3000:localhost:3000 -L 9090:localhost:9090 -L 9400:localhost:9400
```

Run the engine and see `localhost:3000` and `localhost:9090/targets`

```
uv run leaninfer
```
