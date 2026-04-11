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
_MODEL_NAME: str = os.getenv("OPENAI_MODEL", "qwen3.5-flash")

_openai_client: Optional[OpenAI] = None


def get_api_key() -> str:
    return _API_KEY


def get_base_url() -> str:
    return _BASE_URL


def get_model_name() -> str:
    return _MODEL_NAME


def get_openai_client() -> OpenAI:
    """
    Return a shared OpenAI SDK client (created once, reused).

    IMPORTANT: max_retries=0 because the SDK's default is 2, which triples
    effective timeout (timeout * 3). We handle our own retries at call sites
    where needed. Default timeout is also set here as a global fallback.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=_API_KEY,
            base_url=_BASE_URL,
            max_retries=0,          # critical: prevent 3x timeout multiplier
            timeout=60.0,           # default per-call cap, can be overridden
        )
    return _openai_client


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def llm_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    model: str | None = None,
    timeout: int = 60,
    max_tokens: int | None = None,
    max_retries: int = 1,
) -> str:
    """
    One-liner LLM chat completion using the OpenAI SDK.
    Returns the assistant message text, or empty string on failure.

    Args:
        timeout: Per-attempt timeout in seconds. Actual wall-clock worst case
            is ``timeout * (max_retries + 1)``.
        max_retries: Additional attempts after a failure. Default 1 (so up
            to 2 total attempts). Set to 0 to disable retries.
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
    last_error = None

    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0
            text = resp.choices[0].message.content
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
            last_error = e
            err_str = str(e)[:150]
            # Only retry on transient errors (timeout, connection)
            is_transient = any(
                s in err_str.lower()
                for s in ("timeout", "timed out", "connection", "read", "deadline")
            )
            if attempt < max_retries and is_transient:
                logger.warning(
                    "LLM_CALL retry %d/%d after %.1fs: %s (prompt=%.40s)",
                    attempt + 1, max_retries, elapsed, err_str, caller,
                )
                continue
            logger.error(
                "LLM_CALL FAILED (attempt %d): model=%s %.1fs error=%s prompt=%.50s",
                attempt + 1, kwargs.get("model"), elapsed, err_str, caller,
            )
            return ""

    return ""


def llm_chat_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    model: str | None = None,
    timeout: int = 60,
    max_retries: int = 1,
) -> Any:
    """
    LLM chat that expects a JSON response.
    Attempts to parse JSON from the response; returns None on failure.
    """
    text = llm_chat(
        messages, temperature=temperature, model=model,
        timeout=timeout, max_retries=max_retries,
    )
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


def llm_chat_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    temperature: float = 0.1,
    model: str | None = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    LLM chat with OpenAI-style function/tool calling support.
    Returns the assistant message as a dict with 'content' and optional 'tool_calls'.
    Falls back to llm_chat_json if tool calling is not supported by the endpoint.
    """
    client = get_openai_client()
    kwargs: Dict[str, Any] = dict(
        model=model or _MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        timeout=timeout,
    )
    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0
        msg = resp.choices[0].message
        logger.info("LLM_TOOL_CALL: model=%s %.1fs", kwargs["model"], elapsed)
        result = {"content": msg.content or ""}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ]
        return result
    except Exception as e:
        elapsed = time.time() - t0
        logger.warning(
            "LLM_TOOL_CALL not supported (%.1fs, %s), falling back to llm_chat_json",
            elapsed, e,
        )
        # Fallback: use structured JSON output instead of native tool calling
        fallback_msg = (
            "You have these tools available:\n"
            + json.dumps([t.get("function", t) for t in tools], indent=2)
            + "\n\nRespond with a JSON object containing your plan."
        )
        messages_with_tools = [messages[0]] + [{"role": "system", "content": fallback_msg}] + messages[1:]
        result_json = llm_chat_json(messages_with_tools, temperature=temperature, model=model)
        return {"content": json.dumps(result_json) if result_json else "", "tool_calls": None}


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
