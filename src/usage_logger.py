"""Lightweight OpenAI/litellm usage logger for API cost analysis.

Monkeypatches ``litellm.completion`` and ``litellm.acompletion`` so every LLM
call made during an eval run appends one JSONL record (model + token counts,
including GPT-5.1 reasoning tokens and cached-prompt tokens) to the file named
by the ``PANDO_USAGE_LOG`` environment variable.

No-op unless ``PANDO_USAGE_LOG`` is set, so this is safe to install
unconditionally. Added for the interp-pipeline cost analysis; ``Pando/`` is
gitignored, so this patch is easy to keep or revert.
"""

import json
import os
import threading
import time

_lock = threading.Lock()
_installed = False


def _extract(response):
    rec = {}
    u = getattr(response, "usage", None)
    if u is None:
        return rec
    rec["prompt_tokens"] = getattr(u, "prompt_tokens", None)
    rec["completion_tokens"] = getattr(u, "completion_tokens", None)
    rec["total_tokens"] = getattr(u, "total_tokens", None)
    ctd = getattr(u, "completion_tokens_details", None)
    if ctd is not None:
        rec["reasoning_tokens"] = getattr(ctd, "reasoning_tokens", None)
    ptd = getattr(u, "prompt_tokens_details", None)
    if ptd is not None:
        rec["cached_tokens"] = getattr(ptd, "cached_tokens", None)
    return rec


def _model_of(args, kwargs):
    if "model" in kwargs:
        return kwargs["model"]
    if args:
        return args[0]
    return "?"


def _write(path, model, kind, response, error=None):
    rec = {"ts": time.time(), "model": model, "kind": kind}
    if error is not None:
        rec["error"] = repr(error)
    else:
        try:
            rec.update(_extract(response))
        except Exception as e:  # never let logging break a run
            rec["log_error"] = repr(e)
    with _lock:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def record(model, response):
    """Directly log one call's usage to PANDO_USAGE_LOG (no-op if unset).

    Call this right after a litellm completion/acompletion returns. More robust
    than monkeypatching, since it runs in the same frame as the real call.
    """
    path = os.environ.get("PANDO_USAGE_LOG")
    if not path:
        return
    try:
        _write(path, model, "direct", response)
    except Exception:
        pass


def install():
    """Install the litellm usage-logging wrappers if PANDO_USAGE_LOG is set."""
    global _installed
    path = os.environ.get("PANDO_USAGE_LOG")
    if not path or _installed:
        return
    import litellm

    orig_completion = litellm.completion
    orig_acompletion = litellm.acompletion

    def sync_wrap(*args, **kwargs):
        response = orig_completion(*args, **kwargs)
        _write(path, _model_of(args, kwargs), "sync", response)
        return response

    async def async_wrap(*args, **kwargs):
        response = await orig_acompletion(*args, **kwargs)
        _write(path, _model_of(args, kwargs), "async", response)
        return response

    litellm.completion = sync_wrap
    litellm.acompletion = async_wrap
    _installed = True
    print(f"[usage_logger] installed; logging litellm usage to {path}")
