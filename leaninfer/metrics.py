from prometheus_client import Counter, Gauge, Histogram

# Buckets
_LATENCY = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_PER_TOKEN = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
_STEP = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


# Throughput
PROMPT_TOKENS: Counter = Counter(
    "leaninfer_prompt_tokens",
    "Prompt tokens consumed by prefill",
)
OUTPUT_TOKENS: Counter = Counter(
    "leaninfer_output_tokens",
    "Tokens produced by decode",
)

# Latency
TTFT: Histogram = Histogram(
    "leaninfer_ttft_seconds",
    "Time between arrival to first token. Under a saturated batch this is dominated by queue "
    "position, not engine speed — so queue_delay",
    buckets=_LATENCY,
)
# We don't really need this. Can be inferred by above/below
QUEUE_DELAY: Histogram = Histogram(
    "leaninfer_queue_delay_seconds",
    "Time between arrival to admission. TTFT minus this is the engine's own prefill cost",
    buckets=_LATENCY,
)
PREFILL: Histogram = Histogram(
    "leaninfer_prefill_seconds",
    "Engine prefill cost per request: TTFT minus queue delay, measured at first token.",
    buckets=_LATENCY,
)
TPOT: Histogram = Histogram(
    "leaninfer_tpot_seconds",
    "Mean inter-token latency per request",
    buckets=_PER_TOKEN,
)
STEP_DURATION: Histogram = Histogram(
    "leaninfer_step_duration_seconds",
    "Wall time of one engine step.",
    ["phase"],  # "prefill" | "decode"
    buckets=_STEP,
)

# Delete once you check variance
OUTPUT_LEN: Histogram = Histogram(
    "leaninfer_output_len_tokens",
    "Tokens generated per request before stop or budget exhaustion.",
    buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256),
)


# Occupancy
RUNNING: Gauge = Gauge(
    "leaninfer_requests_running",
    "Requests currently holding a slot.",
)
WAITING: Gauge = Gauge(
    "leaninfer_requests_waiting",
    "Requests admitted to the queue but not yet holding a slot.",
)
KV_TOKENS: Gauge = Gauge(
    "leaninfer_kv_cache_tokens",
    "Token positions currently occupied across all slots.",
)
KV_CAPACITY: Gauge = Gauge(
    "leaninfer_kv_cache_capacity_tokens",
    "Total addressable token positions (n_slots * max_len). Set once per run.",
)
SLOTS_CAPACITY: Gauge = Gauge(
    "leaninfer_slots_capacity",
    "Total request slots (n_slots). Set once per run.",
)

