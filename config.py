"""LLM configuration: Gemini (main), Groq and Ollama (optional), token controls, rate limits."""
import json
import os
import re
import threading
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from crewai import LLM  # noqa: E402

OLLAMA_MODELS = ["llama3.1", "qwen2.5", "mistral"]

# Best choices first. Only models your key can really use are kept.
PREFERRED_GEMINI = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
GEMINI_SKIP = [
    "image", "tts", "live", "audio", "embedding", "veo", "imagen",
    "robotics", "computer-use", "learnlm", "aqa", "gemma", "vision",
]

PREFERRED_GROQ = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "llama-3.1-8b-instant",
]
GROQ_SKIP = ["whisper", "tts", "orpheus", "guard", "safeguard", "embed", "allam"]

# ------------------------------------------------------------------
# Report size profiles (rough guides, check the live counter)
# ------------------------------------------------------------------
REPORT_PROFILES = {
    "Quick": {
        "searches": 4, "researcher_iter": 3, "results": 5, "snippet": 300,
        "words": "600-800", "max_tokens": 3000, "budget": 40000,
        "note": "4 searches, about 600-800 words, roughly 3 LLM requests",
    },
    "Standard": {
        "searches": 6, "researcher_iter": 3, "results": 5, "snippet": 400,
        "words": "1000-1400", "max_tokens": 4000, "budget": 70000,
        "note": "6 searches, about 1000-1400 words, roughly 3 LLM requests",
    },
    "Detailed": {
        "searches": 6, "researcher_iter": 4, "results": 6, "snippet": 500,
        "words": "1600-2200", "max_tokens": 5000, "budget": 120000,
        "note": "6 searches, about 1600-2200 words, roughly 3 LLM requests",
    },
}


# ------------------------------------------------------------------
# Turn off CrewAI's "replay log" database (it crashed with a JSON error
# and we never use replay).
# ------------------------------------------------------------------
def _disable_task_output_storage():
    try:
        from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler
    except Exception:
        return
    TaskOutputStorageHandler.add = lambda self, *a, **k: None
    TaskOutputStorageHandler.update = lambda self, *a, **k: None
    TaskOutputStorageHandler.reset = lambda self, *a, **k: None
    TaskOutputStorageHandler.load = lambda self, *a, **k: []


_disable_task_output_storage()


# ------------------------------------------------------------------
# Keys and model discovery
# ------------------------------------------------------------------
def get_key(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if provider == "groq":
        return os.getenv("GROQ_API_KEY") or ""
    return ""


def has_key(provider: str) -> bool:
    return bool(get_key(provider))


def fetch_gemini_models(api_key: str) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=100&key={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.load(resp)
    out = []
    for m in data.get("models", []):
        name = (m.get("name") or "").replace("models/", "")
        low = name.lower()
        if "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        if "gemini" not in low:
            continue
        if any(w in low for w in GEMINI_SKIP):
            continue
        out.append(name)
    return out


def fetch_groq_models(api_key: str) -> list:
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.load(resp)
    ids = [m["id"] for m in data.get("data", [])]
    return [i for i in ids if not any(w in i.lower() for w in GROQ_SKIP)]


_MODEL_CACHE = {}


def _order(preferred: list, available: list) -> list:
    ordered = [m for m in preferred if m in available]
    ordered += [m for m in sorted(available) if m not in ordered]
    return ordered


def get_models(provider: str) -> list:
    if provider == "ollama":
        return OLLAMA_MODELS
    if provider in _MODEL_CACHE:
        return _MODEL_CACHE[provider]
    preferred = PREFERRED_GEMINI if provider == "gemini" else PREFERRED_GROQ
    key = get_key(provider)
    if key:
        try:
            available = fetch_gemini_models(key) if provider == "gemini" else fetch_groq_models(key)
            if available:
                _MODEL_CACHE[provider] = _order(preferred, available)
                return _MODEL_CACHE[provider]
        except Exception:
            pass
    return preferred  # not cached, so it is retried next time


GEMINI_MODELS = get_models("gemini")
GROQ_MODELS = get_models("groq")


def pick_strong(pool: list):
    """First full 'flash' model (not lite). Used for the Writer only."""
    for m in pool:
        if "flash" in m and "lite" not in m:
            return m
    return None


def assign_models(provider: str, selected: str, pool: list, use_strong: bool) -> dict:
    plan = {"researcher": selected, "writer": selected, "editor": selected}
    if provider == "gemini" and use_strong and "lite" in selected:
        strong = pick_strong(pool)
        if strong:
            plan["writer"] = strong
    return plan


# ------------------------------------------------------------------
# Token counter + hard budget
# ------------------------------------------------------------------
class TokenBudgetExceeded(Exception):
    """Raised when a run passes the token limit you set."""


TOKEN_STATS = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
TOKEN_BUDGET = {"limit": 0}
RATE_HOOK = {"fn": None}  # the app sets this to show "waiting..." messages
_lock = threading.Lock()


def reset_token_stats():
    with _lock:
        for k in TOKEN_STATS:
            TOKEN_STATS[k] = 0


def set_token_stats(prompt: int, completion: int, total: int, calls: int):
    with _lock:
        TOKEN_STATS.update(prompt=prompt, completion=completion, total=total, calls=calls)


def set_token_budget(limit: int):
    TOKEN_BUDGET["limit"] = int(limit)


def _usage_callback(kwargs, completion_response, start_time, end_time):
    """LiteLLM calls this after every successful LLM call."""
    try:
        usage = getattr(completion_response, "usage", None)
        if usage is None and isinstance(completion_response, dict):
            usage = completion_response.get("usage")
        if usage is None:
            return

        def pick(name):
            if isinstance(usage, dict):
                return usage.get(name, 0) or 0
            return getattr(usage, name, 0) or 0

        with _lock:
            TOKEN_STATS["prompt"] += pick("prompt_tokens")
            TOKEN_STATS["completion"] += pick("completion_tokens")
            TOKEN_STATS["total"] += pick("total_tokens")
            TOKEN_STATS["calls"] += 1
    except Exception:
        pass


def _strip_unsupported(obj):
    if isinstance(obj, list):
        return [_strip_unsupported(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_unsupported(v) for k, v in obj.items() if k != "cache_breakpoint"}
    return obj


def _parse_wait_seconds(msg: str, default: float = 20.0) -> float:
    """Read 'try again in 19.5s' / '6m12.5s' / '850ms' / 'retry in 46s' / retryDelay '46s'."""
    m = re.search(r"(?:try again|retry) in\s*(?:(\d+)m)?\s*([0-9.]+)\s*(ms|s)\b", msg, re.I)
    if m:
        minutes = int(m.group(1)) if m.group(1) else 0
        value = float(m.group(2))
        secs = value / 1000 if m.group(3).lower() == "ms" else value
        return minutes * 60 + secs
    m = re.search(r"retryDelay\W+([0-9.]+)s", msg)
    if m:
        return float(m.group(1))
    return default


# Errors where waiting a minute will not help (daily quota used up, or no free quota)
NO_RETRY = ["perday", "per day", "per_day", "(tpd)", "tokens per day", "limit: 0"]


def _patch_litellm():
    try:
        import litellm
    except ImportError:
        return
    litellm.num_retries = 0  # we do our own retries below

    callbacks = list(litellm.success_callback or [])
    if _usage_callback not in callbacks:
        callbacks.append(_usage_callback)
        litellm.success_callback = callbacks

    if getattr(litellm.completion, "_patched", False):
        return
    original = litellm.completion

    def completion(*args, **kwargs):
        if kwargs.get("messages"):
            kwargs["messages"] = _strip_unsupported(kwargs["messages"])

        for attempt in range(10):
            limit = TOKEN_BUDGET["limit"]
            if limit and TOKEN_STATS["total"] >= limit:
                raise TokenBudgetExceeded(
                    f"Token limit reached: used {TOKEN_STATS['total']:,} of {limit:,} tokens. "
                    "Pick a smaller report size, turn off the editor, or raise the limit."
                )
            try:
                return original(*args, **kwargs)
            except TokenBudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                low = msg.lower()
                if any(k in low for k in NO_RETRY):
                    raise
                is_rate = isinstance(e, litellm.RateLimitError) or any(
                    k in low for k in ["rate_limit", "rate limit", "resource_exhausted", "429"]
                )
                is_busy = any(k in low for k in ["overloaded", "503", "unavailable"])
                if not (is_rate or is_busy):
                    raise
                wait = min(_parse_wait_seconds(msg, 20.0 if is_rate else 8.0) + 2, 70)
                hook = RATE_HOOK["fn"]
                if hook:
                    hook(wait, attempt + 1)
                time.sleep(wait)
        return original(*args, **kwargs)

    completion._patched = True
    litellm.completion = completion


_patch_litellm()


# ------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------
def get_llm(provider: str, model: str = None, temperature: float = 0.3, max_tokens: int = 3000) -> LLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower().strip()

    if provider == "gemini":
        key = get_key("gemini")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Create a free key at https://aistudio.google.com/apikey "
                "and add GEMINI_API_KEY=... to your .env file."
            )
        models = get_models("gemini")
        model = model if model in models else models[0]
        # Gemini models "think" before answering, and thinking counts toward the limit,
        # so give them a bigger output budget. Temperature is left at Gemini's own default.
        return LLM(
            model=f"gemini/{model}",
            api_key=key,
            max_tokens=max(max_tokens, 8192),
        )

    if provider == "groq":
        key = get_key("groq")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is missing. Get a free key at https://console.groq.com/keys "
                "and add GROQ_API_KEY=... to your .env file."
            )
        models = get_models("groq")
        model = model if model in models else models[0]
        return LLM(
            model=f"groq/{model}",
            api_key=key,
            temperature=temperature,
            max_tokens=min(max_tokens, 3500),  # keeps requests inside Groq's per-minute limit
        )

    if provider == "ollama":
        model = model or os.getenv("OLLAMA_MODEL", OLLAMA_MODELS[0])
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return LLM(
            model=f"ollama/{model}",
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unknown provider '{provider}'. Use 'gemini', 'groq' or 'ollama'.")