"""
modal_app.py

Purpose:
    Serves a fine-tuned Llama 3.2 3B model (pulled from a private Hugging Face
    Hub repo) as a secured HTTPS inference endpoint on Modal, called from a
    Botpress HTTP Action node.

    Auth: a shared-secret header (X-Auth-Token) is enforced on all routes
    except /health. See botpress_integration.md for the Botpress-side config.

Framework decision (asgi_app vs fastapi_endpoint vs the deprecated web_endpoint):
    We use `@modal.asgi_app()` wrapping a full FastAPI application, NOT
    `@modal.fastapi_endpoint()` and NOT the deprecated `@modal.web_endpoint()`.
    Reasoning:
      - `fastapi_endpoint` (and legacy `web_endpoint`) are designed for a
        single simple route. We need shared-secret auth middleware applied
        globally, a custom exception handler for structured error JSON, and
        room for a `/health` route -- that's exactly the "define your own
        FastAPI app" use case Modal's own docs say to use `asgi_app` for.
      - `asgi_app` gives us full control over middleware, exception handlers,
        and OpenAPI docs while still running on Modal's autoscaled GPU
        containers with lifecycle hooks (`@modal.enter`).
    Net: asgi_app is the correct production choice here.

Lifecycle decision (@modal.build is deliberately NOT used):
    The task spec asked for `@modal.build` + `@modal.enter`. As of Modal 1.0,
    `@modal.build` is deprecated (see Modal's migration guide). The current
    recommended pattern for caching large downloaded assets (model weights)
    is a `modal.Volume` mounted at the Hugging Face cache path, combined with
    `@modal.enter` to load weights into GPU memory once per container.
    This is a deliberate, documented deviation from the literal spec in favor
    of the currently-supported, non-deprecated approach — using @modal.build
    today would print deprecation warnings and is on a path to removal.

REPLACE checklist (search this file for "# REPLACE"):
    1. MODEL_REPO_ID          - your private HF repo, e.g. "your-org/llama-3.2-3b-ft"
    2. MODEL_SUBFOLDER        - subfolder within the repo holding the weights, if any
    3. HF_SECRET_NAME         - name of the Modal secret holding HF_TOKEN
    4. AUTH_SECRET_NAME       - name of the Modal secret holding SHARED_SECRET_TOKEN
    5. GPU_TYPE               - GPU tier (defaults to "T4" for low/bursty traffic)
    6. APP_NAME / VOLUME_NAME - naming, change if it collides with existing apps
"""

import json
import logging
import os
import time
from typing import Optional

import modal
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Modal App, Image, Volume, Secrets
# --------------------------------------------------------------------------

APP_NAME = "llama32-3b-chatbot"  # REPLACE: change if this collides with an existing Modal app
VOLUME_NAME = "llama32-3b-hf-cache"  # REPLACE: persistent volume name for the HF cache

# REPLACE: your private HF Hub repo containing the fine-tuned Llama 3.2 3B weights
MODEL_REPO_ID = "Snow79703/Hummingbird-assistant-llm"
# REPLACE if your folder name ever changes: the fine-tuned weights live in a
# subfolder of the repo (not the repo root), so from_pretrained needs to be
# told where to look.
MODEL_SUBFOLDER = "stage3-merged"

# REPLACE: exact Modal secret name. Create it BEFORE deploying, e.g.:
#   modal secret create huggingface-secret HF_TOKEN=hf_xxx...
HF_SECRET_NAME = "huggingface-secret"

# REPLACE: exact Modal secret name for the Botpress shared-secret header. Create
# it BEFORE deploying, e.g.:
#   modal secret create botpress-shared-secret SHARED_SECRET_TOKEN=<a long random string>
AUTH_SECRET_NAME = "botpress-shared-secret"

# REPLACE: GPU tier. "T4" chosen because: (a) traffic is low/bursty, so we want
# the cheapest GPU that still fits the model comfortably in bf16, and (b) T4 is
# available on Modal's free/starter credit tier. A Llama 3.2 3B model in bf16
# needs ~6-7GB of weights + KV cache headroom, which fits well inside T4's 16GB.
# Upgrade to "L4" or "A10G" if you need lower per-request latency, or "A100"
# for high-throughput production traffic.
GPU_TYPE = "T4"

app = modal.App(APP_NAME)

# Persistent volume for the Hugging Face cache. Using a Volume (instead of the
# deprecated @modal.build pattern) means weights are downloaded once and
# survive across image rebuilds -- you don't re-download multi-GB weights
# every time you tweak an unrelated dependency in the Image.
hf_cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Container image. Versions are pinned for stability -- floating versions on
# a GPU inference stack (torch / transformers / accelerate / bitsandbytes)
# is a common source of silent breakage on redeploy.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.46.0",
        "accelerate==0.34.2",
        # bitsandbytes is installed for optional future quantization (e.g. if
        # you later switch to 8-bit/4-bit loading). It is NOT used in the
        # bf16 code path below and can be dropped if you want a smaller image.
        "bitsandbytes==0.44.1",
        "sentencepiece==0.2.0",
        "safetensors==0.4.5",
        "huggingface_hub==0.25.2",
        "hf_transfer==0.1.8",  # fast, parallel HF downloads
        "fastapi[standard]==0.115.0",
        "pydantic==2.9.2",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": "/cache",  # matches the Volume mount path below
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

# --------------------------------------------------------------------------
# Pydantic request/response schemas
# --------------------------------------------------------------------------


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000, description="User prompt / message")
    max_new_tokens: int = Field(256, ge=1, le=1024, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling top-p")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    system_prompt: Optional[str] = Field(
        None, max_length=4000, description="Optional system/instruction prompt"
    )


class GenerationResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: str


# --------------------------------------------------------------------------
# Structured logging setup
# --------------------------------------------------------------------------


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("llama32_inference")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _log_event(logger: logging.Logger, level: str, **fields) -> None:
    """Emit a single-line JSON log record so it's easy to grep/parse in
    Modal's log viewer or any downstream log aggregator."""
    record = {"timestamp": time.time(), "level": level, **fields}
    getattr(logger, level)(json.dumps(record))


# --------------------------------------------------------------------------
# Model serving class
# --------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/cache": hf_cache_volume},
    secrets=[
        modal.Secret.from_name(HF_SECRET_NAME),
        modal.Secret.from_name(AUTH_SECRET_NAME),
    ],
    # Low/bursty traffic profile: scale to zero when idle to avoid paying for
    # an always-on GPU, accept a cold start on the first request after idle.
    min_containers=0,
    # Cap concurrent containers so a traffic burst can't runaway your bill;
    # raise this if you outgrow "low/bursty".
    max_containers=3,
    # How long an idle container stays warm before Modal shuts it down.
    # 2 minutes balances cold-start avoidance against paying for idle GPU time
    # on a low-traffic bot.
    scaledown_window=120,
    # Generous container-level timeout to comfortably cover model load +
    # a worst-case long generation without killing the container mid-request.
    timeout=600,
)
@modal.concurrent(max_inputs=1)  # serialize requests per container: HF `generate()`
# on a single model instance is not safely reentrant for concurrent calls, and
# GPU compute would serialize anyway. Bursts are handled by max_containers above,
# not by stacking concurrent requests onto one container.
class Llama32Inference:
    @modal.enter()
    def load_model(self):
        """Runs once per container start. Loads tokenizer + model onto the
        GPU and keeps them resident in memory (self.*) for all subsequent
        requests handled by this warm container -- no re-download, no
        re-initialization per request."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.logger = _get_logger()
        start = time.time()

        hf_token = os.environ["HF_TOKEN"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPO_ID, subfolder=MODEL_SUBFOLDER, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO_ID,
            subfolder=MODEL_SUBFOLDER,
            token=hf_token,
            torch_dtype=torch.bfloat16,  # full-precision inference per user requirement
            device_map="auto",
        )
        self.model.eval()

        if not self.tokenizer.chat_template:
            _log_event(
                self.logger,
                "warning",
                event="no_chat_template",
                detail="Tokenizer has no chat_template; requests will use plain-text "
                "prompt concatenation instead. Verify this matches your fine-tuning format.",
            )

        load_seconds = round(time.time() - start, 2)
        _log_event(
            self.logger,
            "info",
            event="model_loaded",
            model=MODEL_REPO_ID,
            load_seconds=load_seconds,
        )

    def _generate(self, req: GenerationRequest) -> GenerationResponse:
        """Runs the actual forward pass. Kept as a plain method (not a Modal
        endpoint) so it can be safely called from a thread pool inside the
        ASGI app without going through another Modal RPC hop."""
        import torch

        start = time.time()

        if self.tokenizer.chat_template:
            messages = []
            if req.system_prompt:
                messages.append({"role": "system", "content": req.system_prompt})
            messages.append({"role": "user", "content": req.prompt})
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            # This fine-tuned checkpoint's tokenizer has no chat_template
            # configured. Falling back to an Alpaca-style Instruction/Response
            # template, matching the "instruction"/"response" field names in
            # the training data. VERIFY this against your actual training
            # script's formatting_func / prompt_template -- if your trainer
            # used a different wrapping (different header text, extra
            # newlines, etc.), replace this string to match exactly. Getting
            # this wrong won't crash anything, but will noticeably degrade
            # response quality since the model is sensitive to the exact
            # format it was trained on.
            if req.system_prompt:
                prompt_text = (
                    f"{req.system_prompt}\n\n"
                    f"### Instruction:\n{req.prompt}\n\n### Response:\n"
                )
            else:
                prompt_text = f"### Instruction:\n{req.prompt}\n\n### Response:\n"
            input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(
                self.model.device
            )

        prompt_tokens = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=req.max_new_tokens,
                temperature=max(req.temperature, 1e-4),  # 0.0 breaks HF sampling
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                do_sample=req.temperature > 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        completion_ids = output_ids[0][prompt_tokens:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # Safety net: if the model wasn't trained with a hard stop token for
        # this format, it can sometimes keep generating past the answer and
        # hallucinate a new "### Instruction:" turn. Trim anything from that
        # marker onward. Harmless no-op if using apply_chat_template instead.
        if "### Instruction:" in completion_text:
            completion_text = completion_text.split("### Instruction:")[0]
        latency_ms = round((time.time() - start) * 1000, 1)

        return GenerationResponse(
            response=completion_text.strip(),
            model=MODEL_REPO_ID,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_ids.shape[-1]),
            latency_ms=latency_ms,
        )

    @modal.asgi_app()
    def web(self):
        """Exposes the model over HTTP as a full FastAPI app (see the
        framework-decision note at the top of this file for why asgi_app was
        chosen over fastapi_endpoint/web_endpoint)."""
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import JSONResponse
        import torch

        web_app = FastAPI(
            title="Llama 3.2 3B Fine-Tuned Inference API",
            version="1.0.0",
            docs_url="/docs",
        )

        # --- Auth: only requests carrying the correct shared-secret header
        # are allowed through. This is the Modal-side half of the
        # Botpress<->Modal shared-secret scheme (see botpress_integration.md).
        # The secret is read fresh from os.environ on every request rather
        # than cached on self -- avoids a request/instance timing issue seen
        # earlier where a cached attribute wasn't reliably visible yet.
        @web_app.middleware("http")
        async def enforce_shared_secret(request: Request, call_next):
            if request.url.path == "/health":
                # Allow unauthenticated health checks for warm-up pings/monitoring.
                return await call_next(request)
            provided = request.headers.get("x-auth-token")
            expected = os.environ["SHARED_SECRET_TOKEN"]
            if provided != expected:
                _log_event(
                    self.logger, "warning", event="auth_rejected", path=request.url.path
                )
                return JSONResponse(
                    status_code=401,
                    content=ErrorResponse(
                        error="unauthorized", detail="Missing or invalid X-Auth-Token header"
                    ).model_dump(),
                )
            return await call_next(request)

        @web_app.exception_handler(Exception)
        async def unhandled_exception_handler(request: Request, exc: Exception):
            _log_event(
                self.logger,
                "error",
                event="unhandled_exception",
                path=request.url.path,
                error=str(exc),
            )
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="internal_error", detail="The model server encountered an error."
                ).model_dump(),
            )

        @web_app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_REPO_ID}

        @web_app.post(
            "/generate",
            response_model=GenerationResponse,
            responses={401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
        )
        async def generate(req: GenerationRequest):
            request_start = time.time()
            _log_event(
                self.logger,
                "info",
                event="request_received",
                prompt_length=len(req.prompt),
                max_new_tokens=req.max_new_tokens,
            )
            try:
                # Offload the blocking (GPU-bound, synchronous) generation call
                # to a thread so the ASGI event loop stays responsive, e.g. for
                # concurrent /health checks.
                result = await run_in_threadpool(self._generate, req)
            except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
                _log_event(
                    self.logger, "error", event="cuda_oom", error=str(exc)
                )
                raise HTTPException(
                    status_code=503,
                    detail="Model server is out of GPU memory; try a shorter prompt "
                    "or fewer max_new_tokens.",
                )
            except Exception as exc:
                _log_event(
                    self.logger, "error", event="generation_failed", error=str(exc)
                )
                raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

            total_latency_ms = round((time.time() - request_start) * 1000, 1)
            _log_event(
                self.logger,
                "info",
                event="request_completed",
                prompt_length=len(req.prompt),
                completion_tokens=result.completion_tokens,
                model_latency_ms=result.latency_ms,
                total_latency_ms=total_latency_ms,
            )
            return result

        return web_app


# --------------------------------------------------------------------------
# Deployment
# --------------------------------------------------------------------------
#
# 1. One-time setup (run locally, once):
#
#    modal secret create huggingface-secret HF_TOKEN=hf_your_private_repo_token
#    modal secret create botpress-shared-secret SHARED_SECRET_TOKEN=$(openssl rand -hex 32)
#
#    (Names must exactly match HF_SECRET_NAME / AUTH_SECRET_NAME above.)
#    Save the SHARED_SECRET_TOKEN value somewhere -- you'll need it for the
#    Botpress HTTP Action node's X-Auth-Token header. To view it again:
#    `modal secret list` shows names only, not values; if lost, delete and
#    recreate the secret with a value you write down yourself.
#
# 2. Deploy:
#
#    modal deploy modal_app.py
#
# 3. Retrieve the public endpoint URL:
#    The `modal deploy` command prints the URL directly, e.g.:
#      https://<your-workspace>--llama32-3b-chatbot-llama32inference-web.modal.run
#    You can also find it any time in the Modal dashboard under
#    Apps -> llama32-3b-chatbot -> web, or programmatically:
#
#    modal app list        # confirm the app name
#    python -c "import modal; f = modal.Cls.from_name('llama32-3b-chatbot', 'Llama32Inference'); \
#                print(f().web.get_web_url())"
#
#    The full inference endpoint is: <that base URL>/generate