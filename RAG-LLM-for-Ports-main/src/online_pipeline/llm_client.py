# src/online_pipeline/llm_client.py
"""
Centralized LLM client for the entire project.

All modules should import from here instead of creating their own
OpenAI clients or building API URLs from env vars.

Usage:
    from online_pipeline.llm_client import get_openai_client, get_model_name, llm_chat

    # Option A: use the shared OpenAI SDK client
    client = get_openai_client()
    resp = client.chat.completions.create(model=get_model_name(), ...)

    # Option B: one-liner convenience
    text = llm_chat(messages=[...], temperature=0.3)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger("online_pipeline.llm_client")

load_dotenv()

# ---------------------------------------------------------------------------
# Singleton configuration (loaded once from .env)
# ---------------------------------------------------------------------------

_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
_MODEL_NAME: str = os.getenv("OPENAI_MODEL", "qwen3.5-35b-a3b")

_openai_client: Optional[OpenAI] = None


def get_api_key() -> str:
    return _API_KEY


def get_base_url() -> str:
    return _BASE_URL


def get_model_name() -> str:
    return _MODEL_NAME


def get_openai_client() -> OpenAI:
    """Return a shared OpenAI SDK client (created once, reused)."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
    return _openai_client


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def llm_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    model: str | None = None,
    timeout: int = 120,
    max_tokens: int | None = None,
) -> str:
    """
    One-liner LLM chat completion using the OpenAI SDK.
    Returns the assistant message text, or empty string on failure.
    """
    client = get_openai_client()
    kwargs: Dict[str, Any] = dict(
        model=model or _MODEL_NAME,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    caller = messages[-1].get("content", "")[:60] if messages else "?"
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0
        text = resp.choices[0].message.content
        # Log token usage if available
        usage = getattr(resp, "usage", None)
        if usage:
            logger.info(
                "LLM_CALL: model=%s %.1fs tokens(in=%s out=%s) prompt=%.50s",
                kwargs.get("model"), elapsed,
                getattr(usage, "prompt_tokens", "?"),
                getattr(usage, "completion_tokens", "?"),
                caller,
            )
        else:
            logger.info(
                "LLM_CALL: model=%s %.1fs resp_len=%d prompt=%.50s",
                kwargs.get("model"), elapsed, len(text or ""), caller,
            )
        return text.strip() if text else ""
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("LLM_CALL FAILED: model=%s %.1fs error=%s prompt=%.50s", kwargs.get("model"), elapsed, e, caller)
        return ""


def llm_chat_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    model: str | None = None,
    timeout: int = 120,
) -> Any:
    """
    LLM chat that expects a JSON response.
    Attempts to parse JSON from the response; returns None on failure.
    """
    text = llm_chat(messages, temperature=temperature, model=model, timeout=timeout)
    if not text:
        return None
    # Try to extract JSON object or array
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def llm_chat_raw_post(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    model: str | None = None,
    timeout: tuple = (10, 120),
) -> str:
    """
    Fallback: raw requests.post to the chat/completions endpoint.
    Useful when OpenAI SDK is not desirable.
    """
    url = _BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or _MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    t0 = time.time()
    caller = messages[-1].get("content", "")[:60] if messages else "?"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        elapsed = time.time() - t0
        if r.status_code != 200:
            logger.error("LLM_RAW_POST: status=%d %.1fs prompt=%.50s", r.status_code, elapsed, caller)
            return ""
        text = r.json()["choices"][0]["message"]["content"].strip()
        logger.info("LLM_RAW_POST: model=%s %.1fs resp_len=%d", payload.get("model"), elapsed, len(text))
        return text
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("LLM_RAW_POST FAILED: %.1fs error=%s", elapsed, e)
        return ""
