"""
Ollama wrapper — local LLM access for GaitSense agents.

Skill contract:
  corpus tier:  PUBLIC + DERIVED-OK only (caller is responsible for not
                passing PRIVATE content into prompts sent to this wrapper)
  execution:    local
  model target: configurable — default gaitsense-dispatch (fine-tuned Qwen2.5-0.5B)
                fallback: qwen2.5:0.5b (zero-shot) if gaitsense-dispatch unavailable
                override via OLLAMA_MODEL env var

Inference path (gaitsense-dispatch and any GGUF-backed model):
  Must use /api/generate with raw=True and explicit Qwen chat template tokens.
  /api/chat auto-template detection does not correctly apply the fine-tuned
  model's system prompt, causing base-model freeform output instead of JSON.

Usage (as library):
  from hybrid.wrapper.ollama_client import LocalLLM
  llm = LocalLLM()
  response = llm.chat("parse this table and return JSON: ...")
  tokens   = list(llm.stream("summarise these loss values: ..."))

Usage (as CLI smoke test):
  python hybrid/wrapper/ollama_client.py
"""

import json
import os
from typing import Generator, Optional

import requests

# ── defaults — override via env vars ─────────────────────────────────────────
DEFAULT_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gaitsense-dispatch")

# connect timeout short (server is local); read timeout long (generation)
TIMEOUT = (5, 600)

# Models that require raw Qwen template via /api/generate (not /api/chat)
_RAW_QWEN_MODELS = {"gaitsense-dispatch", "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"}


def _qwen_prompt(system: Optional[str], user: str) -> str:
    """Format a prompt using the Qwen chat template for raw /api/generate calls."""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


class LocalLLM:
    """
    Thin wrapper around the Ollama generate/chat endpoints.
    Agents use this to send PUBLIC or DERIVED-OK content to the local model.

    Corpus contract:
      - Never pass PRIVATE content (formulas, raw signals, patient data) into prompt
      - Input tier: PUBLIC or DERIVED-OK only
      - Output tier: DERIVED-OK (model outputs are treated as derived summaries)

    Routing:
      - gaitsense-dispatch and Qwen family: /api/generate + raw=True + Qwen tokens
      - All other models: /api/chat (standard message array)
    """

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST):
        self.model       = model
        self.host        = host
        self.chat_url    = f"{host}/api/chat"
        self.gen_url     = f"{host}/api/generate"
        self._use_raw    = any(m in model for m in _RAW_QWEN_MODELS)

    def _post_raw(self, system: Optional[str], user: str, stream: bool) -> requests.Response:
        payload = {
            "model":  self.model,
            "prompt": _qwen_prompt(system, user),
            "raw":    True,
            "stream": stream,
            "options": {"temperature": 0.0, "stop": ["<|im_end|>"]},
        }
        return requests.post(self.gen_url, json=payload, stream=stream, timeout=TIMEOUT)

    def _post_chat(self, system: Optional[str], user: str, stream: bool) -> requests.Response:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = {"model": self.model, "messages": messages, "stream": stream}
        return requests.post(self.chat_url, json=payload, stream=stream, timeout=TIMEOUT)

    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Single-turn chat. Blocks until complete response received.
        Returns full response string.
        """
        if self._use_raw:
            resp = self._post_raw(system, prompt, stream=True)
            resp.raise_for_status()
            chunks = []
            for raw in resp.iter_lines():
                if not raw:
                    continue
                chunk = json.loads(raw)
                chunks.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break
            return "".join(chunks)

        resp = self._post_chat(system, prompt, stream=True)
        resp.raise_for_status()
        chunks = []
        for raw in resp.iter_lines():
            if not raw:
                continue
            chunk = json.loads(raw)
            chunks.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                break
        return "".join(chunks)

    def stream(self, prompt: str, system: Optional[str] = None) -> Generator[str, None, None]:
        """
        Streaming chat. Yields tokens as they arrive.
        Use when you want to print progress for long tasks.
        """
        if self._use_raw:
            resp = self._post_raw(system, prompt, stream=True)
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                chunk = json.loads(raw)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
            return

        resp = self._post_chat(system, prompt, stream=True)
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            chunk = json.loads(raw)
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break

    def parse_to_json(self, content: str, schema_hint: str = "") -> dict:
        """
        Ask the local model to parse unstructured DERIVED-OK content into JSON.
        Used by uart-reader and train-sum to convert table output into
        structured dicts safe for forwarding to plot-orchestrator.
        """
        system = (
            "You are a data parser. Extract the requested fields and return "
            "valid JSON only. No explanation. No markdown fences."
        )
        hint = f"\nExpected fields: {schema_hint}" if schema_hint else ""
        prompt = f"Parse this into JSON:{hint}\n\n{content}"
        raw = self.chat(prompt, system=system)
        # strip fences if model wraps output anyway
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])
        return json.loads(raw.strip())

    def summarise(self, content: str, max_words: int = 80) -> str:
        """
        Produce a short DERIVED-OK summary of table or metric output.
        Used by plot-orchestrator to condense evidence blocks before forwarding.
        """
        system = (
            f"Summarise in under {max_words} words. "
            "State only facts present in the input. No interpretation."
        )
        return self.chat(content, system=system)

    def is_available(self) -> bool:
        """Check Ollama server is reachable and model is loaded."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=(3, 5))
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False


# ── CLI smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = LocalLLM()
    print(f"[ollama_client] host={llm.host}  model={llm.model}")

    if not llm.is_available():
        print(f"[ollama_client] FAIL — model '{llm.model}' not available")
        print("[ollama_client] run: ollama pull qwen2.5:0.5b  OR  ollama create gaitsense-dispatch -f hybrid/training/Modelfile")
        raise SystemExit(1)

    print("[ollama_client] model available — running smoke test...")

    # test 1: short parse task (PUBLIC/DERIVED-OK content only)
    test_table = """
    acc_z peak raw:      18.43 m/s²
    acc_z peak filtered: 14.21 m/s²
    gyr_y peak:          142.7 dps
    steps detected:      20
    """
    print("\n[test 1] parse_to_json:")
    result = llm.parse_to_json(
        test_table,
        schema_hint="az_peak_raw, az_peak_filt, gy_peak, steps_detected"
    )
    print(json.dumps(result, indent=2))

    # test 2: stream a short summary
    print("\n[test 2] stream summarise:")
    for token in llm.stream("Summarise in one sentence: val_ode=43.7, val_vel=0.18, val_phase=0.05, best_epoch=312"):
        print(token, end="", flush=True)
    print()

    print("\n[ollama_client] smoke test PASS")
