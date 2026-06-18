"""
utils/model_client.py — Inference backend.
DIRECT_MODE=0 : cloud only
DIRECT_MODE=1 : local model only
DIRECT_MODE=2 : cloud primary, local fallback
"""
from __future__ import annotations

import json, os, re, sys, time
from pathlib import Path
from enum import Enum
from typing import Iterator

import httpx

from config.settings import (
    API_URL, API_KEY, CONTEXT_SIZE, MAX_TOKENS, MODEL_PATH,
    TEMPERATURE, TOP_K, TOP_P, MIN_P, REPEAT_PENALTY,
    VERBOSE, N_THREADS, N_THREADS_BATCH, BATCH_SIZE, N_GPU_LAYERS,
    USE_MLOCK, FLASH_ATTN, KV_CACHE_TYPE, DEFRAG_THOLD, MMPROJ_PATH,
)

CLOUD_MODEL = os.environ.get("CLOUD_MODEL", "gemini-2.5-pro-exp-03-25")

_IMG_RE = re.compile(r'\[IMG:(data:[^\]]+)\]')

class _TS(Enum):
    NORMAL   = 0
    THINKING = 1

class ThinkingStreamParser:
    """
    Splits a streaming token sequence into visible text and think content.

    Emits: (text, is_think=True)  — inside <think>…</think>   → show dimmed
            (text, is_think=False) — outside think blocks       → show normally

    Suppresses entirely:
      • <think> / </think> tags
      • <tool_use>…</tool_use>   (EVE's native tool call format)
      • <tool_call>…</tool_call> (alternate format)
      • <|channel>…<channel|>   (Gemma-style)

    Fixed vs original:
      1. <tool_use> added to suppressed set (was only <tool_call> — caused raw
         tool XML to print verbatim to terminal)
      2. _sup_prev_state: suppress inside <think> restores THINKING state on
         close, so the </think> that follows is handled correctly
      3. Think content buffered and flushed at word boundaries, not char-by-char
         (huge reduction in sys.stdout.write() calls on CPU inference)
      4. No premature emit of partial tag bytes
    """
    _THINK_OPEN     = "<think>"
    _THINK_CLOSE    = "</think>"
    _SUPPRESS_OPENS = ("<tool_use>", "<tool_call>", "<|channel>")
    _SUPPRESS_CLOSE = ("</tool_use>", "</tool_call>", "<channel|>")

    def __init__(self):
        self._state          = _TS.NORMAL
        self._buf            = ""
        self._think          = ""
        self._suppress       = False
        self._sup_prev_state = _TS.NORMAL

    def feed(self, chunk: str) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        i = 0
        while i < len(chunk):
            ch = chunk[i]; i += 1

            if self._suppress:
                self._buf += ch
                for ct in self._SUPPRESS_CLOSE:
                    if self._buf.endswith(ct):
                        self._buf = ""
                        self._suppress = False
                        self._state = self._sup_prev_state
                        break
                if len(self._buf) > 64:
                    self._buf = self._buf[-32:]
                continue

            if self._state == _TS.NORMAL:
                self._buf += ch
                buf = self._buf

                if self._THINK_OPEN.startswith(buf):
                    if buf == self._THINK_OPEN:
                        self._state = _TS.THINKING
                        self._buf = ""
                    continue

                matched_sup = False
                for ot in self._SUPPRESS_OPENS:
                    if ot.startswith(buf):
                        if buf == ot:
                            self._sup_prev_state = self._state
                            self._suppress = True
                            self._buf = ""
                        matched_sup = True
                        break
                if matched_sup:
                    continue

                if buf:
                    out.append((buf, False))
                    self._buf = ""

            else:  # THINKING
                self._buf += ch
                buf = self._buf

                if self._THINK_CLOSE.startswith(buf):
                    if buf == self._THINK_CLOSE:
                        if self._think:
                            out.append((self._think, True))
                            self._think = ""
                        self._state = _TS.NORMAL
                        self._buf = ""
                    continue

                matched_sup = False
                for ot in self._SUPPRESS_OPENS:
                    if ot.startswith(buf):
                        if buf == ot:
                            if self._think:
                                out.append((self._think, True))
                                self._think = ""
                            self._sup_prev_state = self._state
                            self._suppress = True
                            self._buf = ""
                        matched_sup = True
                        break
                if matched_sup:
                    continue

                self._think += buf
                self._buf = ""
                if self._think and self._think[-1] in " \n\t,.:;!?":
                    out.append((self._think, True))
                    self._think = ""

        return out

    def flush(self) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        if self._think:
            out.append((self._think, True))
            self._think = ""
        if self._buf:
            out.append((self._buf, self._state == _TS.THINKING))
            self._buf = ""
        return out

    @property
    def in_thinking(self) -> bool:
        return self._state == _TS.THINKING


class ModelClient:
    def __init__(self, base_url: str = "", direct: int = 1,
                 model_path: str = MODEL_PATH, mmproj_path: str = ""):
        self._direct      = direct   # 0=cloud, 1=local, 2=hybrid
        self._base_url    = base_url
        self._model_path  = model_path
        self._mmproj_path = mmproj_path
        self._llm         = None
        self._cloud_last  = False   # True if last request used cloud
        if direct >= 1:
            self._load_local()

    def health(self) -> bool:
        if self._direct >= 1:
            return self._llm is not None
        return True

    @property
    def _vision_enabled(self) -> bool:
        if self._direct >= 1 and self._llm:
            return getattr(self._llm, "chat_handler", None) is not None or bool(
                getattr(self._llm, "clip_model_path", None)
            )
        return bool(self._mmproj_path or MMPROJ_PATH)

    def complete(
        self,
        messages: list[dict],
        system:    str  = "",
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        stop: list[str] | None = None,
        stream: bool = True,
    ) -> str | Iterator[str]:
        # Hybrid mode: try cloud first, fallback to local
        if self._direct == 2:
            try:
                result = self._complete_cloud(messages, system, max_tokens, temperature, stop, stream)
                self._cloud_last = True
                return result
            except Exception as e:
                if VERBOSE:
                    print(f"[hybrid] Cloud failed ({e}), falling back to local model…", file=sys.stderr)
                return self._complete_direct(messages, system, max_tokens, temperature, stop, stream)
        elif self._direct == 0:
            return self._complete_cloud(messages, system, max_tokens, temperature, stop, stream)
        else:
            return self._complete_direct(messages, system, max_tokens, temperature, stop, stream)

    # ── Local model ────────────────────────────────────────────────────────
    def _load_local(self) -> None:
            try:
                from llama_cpp import Llama
            except ImportError:
                print("[model] llama-cpp-python not installed.\n"
                      "  pip install llama-cpp-python", file=sys.stderr)
                return

            cpu  = os.cpu_count() or 4
            nt   = N_THREADS or max(1, cpu // 2)
            ntb  = N_THREADS_BATCH or cpu
            ngl  = N_GPU_LAYERS

            if VERBOSE:
                print(f"[model] Loading {self._model_path}\n"
                      f"  threads={nt} batch_threads={ntb} gpu_layers={ngl} ctx={CONTEXT_SIZE}",
                      file=sys.stderr)

            # Only proven, stable kwargs — nothing experimental
            base_kwargs = dict(
                model_path      = self._model_path,
                n_ctx           = CONTEXT_SIZE,
                n_threads       = nt,
                n_threads_batch = ntb,
                n_batch         = BATCH_SIZE,
                n_gpu_layers    = ngl,
                use_mmap        = True,           # keep mmap on (fastest)
                use_mlock       = USE_MLOCK,
                verbose         = False,
                defrag_thold    = DEFRAG_THOLD,
            )

            mmproj = self._mmproj_path or MMPROJ_PATH
            if mmproj and Path(mmproj).exists():
                try:
                    from llama_cpp.llama_chat_format import Gemma3ChatHandler
                    base_kwargs["chat_handler"] = Gemma3ChatHandler(clip_model_path=mmproj, verbose=VERBOSE)
                except Exception:
                    try:
                        from llama_cpp.llama_chat_format import Llava16ChatHandler
                        base_kwargs["chat_handler"] = Llava16ChatHandler(clip_model_path=mmproj, verbose=VERBOSE)
                    except Exception:
                        base_kwargs["clip_model_path"] = mmproj

            if KV_CACHE_TYPE > 0:
                base_kwargs["type_k"] = KV_CACHE_TYPE
                base_kwargs["type_v"] = KV_CACHE_TYPE

            def _try_load(**extra):
                return Llama(**base_kwargs, **extra)

            if FLASH_ATTN:
                try:
                    self._llm = _try_load(flash_attn=True)
                    return
                except Exception as e:
                    print(f"[model] flash_attn failed ({e}), retrying without…", file=sys.stderr)

            try:
                self._llm = _try_load()
            except Exception as e_kv:
                print(f"[model] KV quant failed ({e_kv}), retrying with plain f16 KV…", file=sys.stderr)
                bare = {k: v for k, v in base_kwargs.items()
                        if k not in ("type_k", "type_v", "defrag_thold")}
                try:
                    self._llm = Llama(**bare)
                except Exception as e_bare:
                    print(f"[model] Load failed: {e_bare}", file=sys.stderr)
                    self._llm = None


    def _build_messages(self, messages: list[dict], system: str) -> list[dict]:
        out = []
        if system:
            out.append({"role": "system", "content": system})
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "[IMG:" in content:
                parts: list[dict] = []
                last = 0
                for m in _IMG_RE.finditer(content):
                    txt = content[last:m.start()].strip()
                    if txt:
                        parts.append({"type": "text", "text": txt})
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": m.group(1)},
                    })
                    last = m.end()
                tail = content[last:].strip()
                if tail:
                    parts.append({"type": "text", "text": tail})
                if not any(p["type"] == "text" for p in parts):
                    parts.append({"type": "text", "text": "Describe this image."})
                out.append({**msg, "content": parts})
            else:
                out.append(msg)
        return out

    # ── Helper to suppress stderr during local model calls ─────────────────
    @staticmethod
    def _suppress_stderr():
        """Return (saved_fd, devnull_fd) so caller can restore later.
        Returns None if VERBOSE is True (no suppression needed)."""
        if VERBOSE:
            return None
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_fd   = os.dup(2)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
            return saved_fd
        except OSError:
            return None

    @staticmethod
    def _restore_stderr(saved_fd):
        if saved_fd is not None:
            try:
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
            except OSError:
                pass

    def _complete_direct(
        self, messages, system, max_tokens, temperature, stop, stream
    ) -> str | Iterator[str]:
        if not self._llm:
            raise RuntimeError("Local model not loaded.")
        msgs  = self._build_messages(messages, system)
        stops = stop or []
        mt = max_tokens if max_tokens > 0 else -1
        if stream:
            return self._stream_direct(msgs, mt, temperature, stops)
        else:
            saved = self._suppress_stderr()
            try:
                resp = self._llm.create_chat_completion(
                    messages       = msgs,
                    max_tokens     = mt,
                    temperature    = temperature,
                    top_k          = TOP_K,
                    top_p          = TOP_P,
                    min_p          = MIN_P,
                    repeat_penalty = REPEAT_PENALTY,
                    stop           = stops,
                    stream         = False,
                )
            finally:
                self._restore_stderr(saved)
            return resp["choices"][0]["message"]["content"] or ""

    def _stream_direct(self, msgs, max_tokens, temperature, stops) -> Iterator[str]:
        saved = self._suppress_stderr()
        try:
            gen = self._llm.create_chat_completion(
                messages       = msgs,
                max_tokens     = max_tokens,
                temperature    = temperature,
                top_k          = TOP_K,
                top_p          = TOP_P,
                min_p          = MIN_P,
                repeat_penalty = REPEAT_PENALTY,
                stop           = stops,
                stream         = True,
            )
        finally:
            self._restore_stderr(saved)
        for chunk in gen:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                yield delta

    # ── Cloud API ──────────────────────────────────────────────────────────
    def _complete_cloud(
        self, messages, system, max_tokens, temperature, stop, stream
    ):
        msgs = self._build_messages(messages, system)
        payload = {
            "model":       CLOUD_MODEL,
            "messages":    msgs,
            "max_tokens":  max_tokens if max_tokens > 0 else 4096,
            "temperature": temperature,
            "stream":      False,
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                resp = httpx.post(
                    API_URL, json=payload, headers=headers,
                    timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"] or ""
                elif resp.status_code in (429, 413):
                    wait = min(2 ** attempt, 8)
                    if VERBOSE:
                        print(f"[cloud] rate limited, retrying in {wait}s…", file=sys.stderr)
                    time.sleep(wait)
                elif resp.status_code in (401, 403):
                    raise RuntimeError(f"Cloud authentication failed: {resp.text[:100]}")
                else:
                    raise RuntimeError(f"Cloud returned {resp.status_code}: {resp.text[:200]}")
            except RuntimeError:
                raise
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Cloud connection error: {e}")
                time.sleep(1)

        raise RuntimeError("Cloud rate limit exceeded after retries")