
import os
import re
import logging
from typing import Any

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask



GPU_3070_URL = os.getenv(
    "GPU_3070_URL",
    "http://192.168.1.100:8080"
)

GPU_1050_URL = os.getenv(
    "GPU_1050_URL",
    "http://192.168.1.101:8080"
)

ROUTER_HOST = "0.0.0.0"
ROUTER_PORT = 8000

MODEL_4B = "gemma-4b"
MODEL_E2B = "gemma-e2b"

# Tune these after observing real workloads.
CONTEXT_MEDIUM = 2500
CONTEXT_LONG = 5000
COMPLEXITY_THRESHOLD = 3

TIMEOUT = 600.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-router")

app = FastAPI(title="Local LLM Intelligent Router")

client = httpx.AsyncClient(
    timeout=httpx.Timeout(TIMEOUT, connect=10.0)
)



COMPLEX_PATTERNS = [
    r"\b(debug|debugging|troubleshoot)\b",
    r"\b(architecture|system design|distributed system)\b",
    r"\b(optimize|optimization|performance bottleneck)\b",
    r"\b(prove|proof|derive|derivation)\b",
    r"\b(step.by.step|multi.step|reason through)\b",
    r"\b(compare and contrast|evaluate critically)\b",
    r"\b(algorithm|data structure|complexity analysis)\b",
    r"\b(concurrency|parallelism|race condition)\b",
    r"\b(mathematical reasoning|calculus|linear algebra)\b",
    r"\b(implement|refactor|review this code)\b",
    r"\b(root cause|in.depth analysis)\b",
]

SIMPLE_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you)\b",
    r"\b(what is|who is|when is)\b",
    r"\b(define|meaning of|translate)\b",
    r"\b(convert \d+|calculate \d+\s*[\+\-*/])",
]

def get_text(content: Any) -> str:
    """Extract text from standard or multimodal message content."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))

        return " ".join(parts)

    return ""


def extract_messages(messages: list) -> tuple[str, str]:
    """Return complete conversation and latest user prompt."""

    all_text = []
    latest_user = ""

    for message in messages:
        content = get_text(message.get("content", ""))
        all_text.append(content)

        if message.get("role") == "user":
            latest_user = content

    return "\n".join(all_text), latest_user


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate.
    Actual token counts depend on model tokenizer.
    """

    return max(1, len(text) // 4)


def classify_prompt(messages: list) -> dict:
    full_context, latest_prompt = extract_messages(messages)

    context_tokens = estimate_tokens(full_context)
    prompt = latest_prompt.lower()

    score = 0
    reasons = []


    if context_tokens >= CONTEXT_LONG:
        score += 5
        reasons.append("very long context")

    elif context_tokens >= CONTEXT_MEDIUM:
        score += 3
        reasons.append("long context")


    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, prompt):
            score += 3
            reasons.append("complex task: " + pattern)
            break

    # Multiple instructions / questions
    if len(re.findall(r"[?]", latest_prompt)) >= 3:
        score += 2
        reasons.append("multiple questions")

    # Code-heavy prompts
    if (
        "```" in latest_prompt
        or len(re.findall(r"\b(def |class |import |function )", latest_prompt)) >= 3
    ):
        score += 3
        reasons.append("code-heavy prompt")

    # Mathematical expressions
    if len(re.findall(r"[=<>]", latest_prompt)) >= 4:
        score += 2
        reasons.append("mathematical expressions")



    simple = any(
        re.search(pattern, prompt)
        for pattern in SIMPLE_PATTERNS
    )

    if simple and context_tokens < 1000 and score == 0:
        reasons.append("simple direct query")

    # ----------------------------------------------
    # FINAL DECISION
    # ----------------------------------------------

    if score >= COMPLEXITY_THRESHOLD:
        selected = "3070"
        model = MODEL_4B
        endpoint = GPU_3070_URL
    else:
        selected = "1050ti"
        model = MODEL_E2B
        endpoint = GPU_1050_URL

    return {
        "gpu": selected,
        "model": model,
        "endpoint": endpoint,
        "score": score,
        "context_tokens_estimated": context_tokens,
        "reasons": reasons or ["low complexity"]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    messages = payload.get("messages")

    if not isinstance(messages, list) or not messages:
        raise HTTPException(400, "messages must be a non-empty list")

    # Evaluate complexity
    decision = classify_prompt(messages)

    endpoint = decision["endpoint"]
    model = decision["model"]

    logger.info(
        "Routing to %s | score=%s | context=%s | reasons=%s",
        decision["gpu"],
        decision["score"],
        decision["context_tokens_estimated"],
        decision["reasons"]
    )

    payload["model"] = model

    target = endpoint.rstrip("/") + "/v1/chat/completions"

    try:
        upstream_request = client.build_request(
            "POST",
            target,
            json=payload
        )

        upstream_response = await client.send(
            upstream_request,
            stream=payload.get("stream", False)
        )

        if upstream_response.status_code >= 400:
            error_body = await upstream_response.aread()
            await upstream_response.aclose()

            return JSONResponse(
                status_code=upstream_response.status_code,
                content={
                    "error": "Upstream model error",
                    "gpu": decision["gpu"],
                    "details": error_body.decode(
                        "utf-8", errors="replace"
                    )
                }
            )

        headers = {
            k: v for k, v in upstream_response.headers.items()
            if k.lower() not in {
                "content-length",
                "transfer-encoding",
                "connection",
                "content-encoding"
            }
        }

        headers["X-Router-GPU"] = decision["gpu"]
        headers["X-Router-Score"] = str(decision["score"])

        if payload.get("stream", False):
            return StreamingResponse(
                upstream_response.aiter_raw(),
                status_code=upstream_response.status_code,
                headers=headers,
                background=BackgroundTask(upstream_response.aclose),
                media_type=upstream_response.headers.get(
                    "content-type", "text/event-stream"
                )
            )

        body = await upstream_response.aread()
        await upstream_response.aclose()

        return JSONResponse(
            content=httpx.Response(
                status_code=upstream_response.status_code,
                content=body
            ).json(),
            status_code=upstream_response.status_code,
            headers=headers
        )

    except httpx.RequestError as exc:
        logger.exception("Model connection failed")

        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach {decision['gpu']}: {str(exc)}"
        )



@app.post("/route-preview")
async def route_preview(request: Request):

    payload = await request.json()
    messages = payload.get("messages", [])

    if not messages:
        raise HTTPException(400, "No messages supplied")

    return classify_prompt(messages)


@app.get("/health")
async def health():
    return {
        "status": "running",
        "models": {
            "3070": GPU_3070_URL,
            "1050ti": GPU_1050_URL
        }
    }


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()
