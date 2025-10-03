---
sidebar_position: 4
---

# Plain Ray Serve Deployment (No Store Optimizations)

This guide walks through deploying a plain large-language-model inference service
on Ray Serve without ServerlessLLM's accelerated checkpoint loader. It pairs with
the [Ray serverless example](../../../examples/ray_serverless/README.md), which
contains the deployment script and step-by-step instructions.

## When to Use This Setup

- You want to benchmark baseline Ray Serve performance before enabling
  ServerlessLLM optimizations.
- Your platform cannot mount the ServerlessLLM store or you prefer to rely on the
  Hugging Face cache built into `transformers`.
- You need a minimal reference for running Ray Serve with autoscaling to zero.

## Overview

The example exposes a FastAPI app via Ray Serve with the following traits:

- **Model loading** uses `transformers.AutoModelForCausalLM.from_pretrained`
  directly, so the first activation downloads weights from the Hugging Face Hub
  (or a pre-populated cache).
- **Autoscaling** is configured via Serve's `autoscaling_config`. Replicas scale
  between zero and `MAX_REPLICAS` depending on request concurrency.
- **Serverless-friendly**: Environment variables allow you to request GPU or CPU
  resources per replica and to tune the autoscaler targets.

## Prerequisites

Install Ray Serve, PyTorch, and Transformers. For example:

```bash
pip install "ray[serve]>=2.10.0" torch transformers fastapi uvicorn
```

Ensure the dependencies are present on all Ray nodes that might host Serve
replicas.

## Deployment Steps

1. **Launch a Ray cluster.** For local tests, run:
   ```bash
   ray start --head --dashboard-host=0.0.0.0 --num-cpus=4
   ```
   Connect additional nodes with `ray start --address='ray://<HEAD_IP>:10001'` or
   follow your managed platform's instructions.

2. **Configure the deployment.** Override environment variables such as
   `MODEL_ID`, `NUM_GPUS_PER_REPLICA`, and `MAX_REPLICAS` to match your hardware
   and workload targets. The full list is documented in the
   [example README](../../../examples/ray_serverless/README.md#2-configure-the-deployment).

3. **Deploy the Serve graph.** From the repository root run:
   ```bash
   serve run examples/ray_serverless/ray_serve_app:deployment_graph
   ```
   Serve uploads the deployment to the cluster and spins up replicas on demand.

4. **Send requests.** POST to `/generate` with a `prompt` or `prompts` field, or
   call `/healthz` to verify the deployment is live. Refer to the example README
   for `curl` samples.

5. **Monitor and scale.** Use the Ray Dashboard or `ray list actors` to observe
   replica creation. Tune `TARGET_ONGOING_REQUESTS` and `MAX_REPLICAS` for
   desired throughput.

6. **Clean up.** When finished, remove the deployment with
   `serve delete RayServeLLM` and stop the cluster via `ray stop`.

Because this configuration intentionally bypasses the ServerlessLLM store, it is
ideal for comparing against the optimized pipeline or for environments where a
shared checkpoint store cannot be attached.
