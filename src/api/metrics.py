from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Model inference latency"
)

def metrics():
    return Response(generate_latest(), media_type="text/plain")
