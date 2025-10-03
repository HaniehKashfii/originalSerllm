# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
"""Minimal Ray Serve application for running LLM inference in a serverless way.

This example demonstrates how to deploy a plain LLM workload on a Ray cluster
without relying on the ServerlessLLM checkpoint store optimizations. It uses
Ray Serve's autoscaling capabilities to scale replicas up and down based on the
incoming request volume.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

import torch
from fastapi import FastAPI, Request
from ray import serve
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _normalize_completions(prompts: Iterable[str], outputs: List[str]) -> List[str]:
    normalized: List[str] = []
    for prompt, text in zip(prompts, outputs):
        if text.startswith(prompt):
            normalized.append(text[len(prompt) :].lstrip())
        else:
            normalized.append(text)
    return normalized


MIN_REPLICAS = _env_int("MIN_REPLICAS", 0)
MAX_REPLICAS = _env_int("MAX_REPLICAS", 2)
TARGET_ONGOING_REQUESTS = _env_float(
    "TARGET_ONGOING_REQUESTS", 1.0
)
NUM_GPUS_PER_REPLICA = _env_float("NUM_GPUS_PER_REPLICA", 0.0)
NUM_CPUS_PER_REPLICA = _env_float("NUM_CPUS_PER_REPLICA", 1.0)


@serve.deployment(
    autoscaling_config={
        "min_replicas": MIN_REPLICAS,
        "initial_replicas": MIN_REPLICAS,
        "max_replicas": max(MIN_REPLICAS, MAX_REPLICAS),
        "target_num_ongoing_requests_per_replica": max(TARGET_ONGOING_REQUESTS, 0.1),
    },
    ray_actor_options={
        "num_gpus": NUM_GPUS_PER_REPLICA,
        "num_cpus": NUM_CPUS_PER_REPLICA,
    },
)
@serve.ingress(app)
class RayServeLLM:
    """A minimal Ray Serve deployment hosting a Hugging Face model."""

    def __init__(self) -> None:
        self.model_id = os.getenv("MODEL_ID", "facebook/opt-125m")
        self.max_new_tokens = _env_int("MAX_NEW_TOKENS", 256)
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()

    @app.get("/healthz")
    async def health(self) -> Dict[str, str]:
        return {"status": "ok", "model": self.model_id}

    @app.post("/generate")
    async def generate(self, request: Request) -> Dict[str, Any]:
        payload = await request.json()
        prompts: Iterable[str]
        if "prompts" in payload:
            prompts = payload["prompts"]
        elif "prompt" in payload:
            prompts = [payload["prompt"]]
        else:
            return {"error": "Request payload must include 'prompt' or 'prompts'."}

        prompts = [str(p) for p in prompts]
        generation_kwargs = payload.get("generation_kwargs", {})
        max_new_tokens = int(
            generation_kwargs.get("max_new_tokens", self.max_new_tokens)
        )
        temperature = float(generation_kwargs.get("temperature", self.temperature))
        top_p = float(generation_kwargs.get("top_p", self.top_p))

        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        )
        if self.device == "cuda":
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.inference_mode():
            generated = self.model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )
        completions = _normalize_completions(prompts, decoded)
        return {
            "model": self.model_id,
            "prompts": list(prompts),
            "completions": completions,
            "generation_kwargs": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }


deployment_graph = RayServeLLM.bind()

