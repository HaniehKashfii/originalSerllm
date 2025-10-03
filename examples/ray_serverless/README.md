# Plain Ray-based LLM Inference on a Serverless Platform

This example shows how to run a **plain Ray Serve application** for large language
model inference on a serverless platform. Unlike the default ServerlessLLM flow,
this setup does **not** rely on the optimized checkpoint loading path – models are
loaded directly from the Hugging Face Hub cache when a replica starts. The focus
here is demonstrating how to combine Ray's autoscaling primitives with a minimal
FastAPI interface for on-demand inference.

The application code lives in [`ray_serve_app.py`](./ray_serve_app.py) and can be
deployed with `serve run`. Autoscaling is configured so that replicas can scale
all the way down to zero when there is no traffic, keeping GPU time usage to the
bare minimum.

## Prerequisites

- Python 3.10 or newer
- [Ray 2.10+](https://docs.ray.io/en/latest/) with the Serve extras installed
- PyTorch and Transformers (any build that supports the target hardware)
- Optional: at least one NVIDIA GPU for accelerated inference

Install the dependencies into a clean environment:

```bash
pip install "ray[serve]>=2.10.0" torch transformers fastapi uvicorn
```

If you plan to run the workload on a remote Ray cluster or a managed serverless
platform, make sure the dependencies are available on every node that will host
Serve replicas.

## 1. Start a Ray Cluster

For a quick local setup, launch a Ray head node:

```bash
ray start --head --dashboard-host=0.0.0.0 --num-cpus=4
```

If you have additional worker nodes or GPU machines, connect them to the head
with:

```bash
ray start --address='ray://<HEAD_IP>:10001'
```

> **Note:** For real serverless deployments (e.g., KubeRay, Anyscale), use the
> platform's provisioning workflow and make sure the Serve controller can reach
the `ray start --head` instance.

## 2. Configure the Deployment

The Serve deployment reads configuration from environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_ID` | `facebook/opt-125m` | Hugging Face model identifier to load. |
| `MAX_NEW_TOKENS` | `256` | Maximum tokens generated per request. |
| `TEMPERATURE` | `0.7` | Sampling temperature. Set to `0` for greedy decoding. |
| `TOP_P` | `0.9` | Nucleus sampling `top_p`. |
| `NUM_GPUS_PER_REPLICA` | `0.0` | GPUs requested by each replica. Set to `1` on GPU nodes. |
| `NUM_CPUS_PER_REPLICA` | `1.0` | CPUs requested per replica. |
| `MIN_REPLICAS` | `0` | Minimum number of active replicas. |
| `MAX_REPLICAS` | `2` | Maximum number of replicas Serve can scale out to. |
| `TARGET_ONGOING_REQUESTS` | `1.0` | Target ongoing requests per replica used by autoscaling. |
| `RAY_SERVE_AUTOSCALER_IDLE_TIMEOUT_S` | `60` | (Ray setting) Seconds of idleness before scaling a replica down. |

Export any overrides before launching the deployment. For example, to use a GPU
worker with a larger model:

```bash
export MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
export NUM_GPUS_PER_REPLICA=1
export MAX_REPLICAS=3
export RAY_SERVE_AUTOSCALER_IDLE_TIMEOUT_S=120
```

## 3. Deploy the Application

With the cluster running and the environment configured, deploy the Serve graph:

```bash
serve run examples/ray_serverless/ray_serve_app:deployment_graph
```

The command packages the deployment definition, uploads it to the Ray cluster,
and waits for the first replica to become ready. Because `MIN_REPLICAS` defaults
to zero, Serve will create a replica on demand when the first request arrives.

> The first request for a new replica downloads the model from the Hugging Face
> Hub (or reads from the local cache), which can take a few minutes for large
> checkpoints. Subsequent activations reuse the cached weights.

## 4. Send Inference Requests

The deployment exposes two endpoints:

- `GET /healthz` – health probe returning the model ID.
- `POST /generate` – accepts a JSON payload with either a `prompt` or `prompts`
  field.

Example request:

```bash
curl http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "Write a haiku about Ray Serve.",
        "generation_kwargs": {
            "max_new_tokens": 64,
            "temperature": 0.6,
            "top_p": 0.95
        }
      }'
```

The response looks like:

```json
{
  "model": "facebook/opt-125m",
  "prompts": ["Write a haiku about Ray Serve."],
  "completions": ["Scheduling waves flow\nAutoscaling moonlit pods\nServerless whispers"],
  "generation_kwargs": {
    "max_new_tokens": 64,
    "temperature": 0.6,
    "top_p": 0.95
  }
}
```

When there is no traffic, replicas scale down automatically after the idle
timeout. The next request will spin up a fresh replica and reload the checkpoint
straight from the Hugging Face cache.

## 5. Observability and Scaling Behavior

- Inspect autoscaling decisions with `ray dashboard` or `ray list actors`.
- Adjust `TARGET_ONGOING_REQUESTS` to increase or decrease parallelism per
  replica.
- Tune `MAX_REPLICAS` to limit the maximum hardware footprint.

## 6. Tear Down

Shut down the Serve application and Ray cluster when you are done:

```bash
serve delete RayServeLLM
ray stop
```

Because this walkthrough avoids the ServerlessLLM store, it is a good reference
for benchmarking raw Ray Serve behavior or running on platforms where attaching
an external checkpoint store is not possible.
