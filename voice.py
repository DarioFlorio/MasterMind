"""
voice.py — Speech I/O for MasterMind.

Speech-to-Text backends (in priority order):
  1. faster-whisper  — local, fast, high quality (recommended)
  2. whisper         — original OpenAI Whisper (slower fallback)

Text-to-Speech backends (selected via TTS_BACKEND in .env):
  kokoro    — local, natural-sounding, no internet needed  (default)
  piper     — local, fast, good quality, single binary
  edge      — Microsoft neural TTS, excellent quality, needs internet
  elevenlabs— best possible quality, needs API key + internet
  pyttsx3   — offline, OS voices, robotic but zero dependencies

Configuration (.env):
  TTS_BACKEND=kokoro          # kokoro | piper | edge | elevenlabs | pyttsx3
  TTS_VOICE=af_heart          # voice name / ID (backend-specific)
  TTS_SPEED=1.0               # playback speed multiplier
  ELEVENLABS_API_KEY=...      # required only for elevenlabs backend
  EDGE_TTS_VOICE=en-GB-SoniaNeural  # for edge backend
  PIPER_BINARY=piper          # path to piper binary
  PIPER_MODEL=...             # path to .onnx voice model
  WHISPER_MODEL=base          # tiny|base|small|medium|large
  VAD_SILENCE_MS=700          # ms of silence before utterance ends
  VAD_THRESHOLD=0.01          # mic energy threshold (0.0-1.0)
"""
from __future__ import annotations

import io
import os
import queue
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

# ── Config from env ───────────────────────────────────────────────────────────

TTS_BACKEND    = os.getenv("TTS_BACKEND", "kokoro").lower().strip()
TTS_VOICE      = os.getenv("TTS_VOICE", "")
TTS_SPEED      = float(os.getenv("TTS_SPEED", "1.0"))
WHISPER_MODEL  = os.getenv("WHISPER_MODEL", "base")
VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "700"))
VAD_THRESHOLD  = float(os.getenv("VAD_THRESHOLD", "0.01"))
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY", "")
EDGE_VOICE     = os.getenv("EDGE_TTS_VOICE", "en-GB-SoniaNeural")
PIPER_BINARY   = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL    = os.getenv("PIPER_MODEL", "")

SAMPLE_RATE   = 16000
CHANNELS      = 1
CHUNK_FRAMES  = 512   # ~32ms per chunk at 16kHz

# ── ANSI (reuse main.py palette) ─────────────────────────────────────────────
_C     = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
RESET  = "\033[0m"      if _C else ""
DIM    = "\033[2m"      if _C else ""
GREEN  = "\033[92m"     if _C else ""
YELLOW = "\033[93m"     if _C else ""
CYAN   = "\033[96m"     if _C else ""
RED    = "\033[91m"     if _C else ""
BOLD   = "\033[1m"      if _C else ""
MIC    = "🎙"
SPEAK  = "🔊"


# ═══════════════════════════════════════════════════════════════════════════════
# Microphone capture with Voice Activity Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _rms(data: bytes) -> float:
    """Root-mean-square energy of a chunk of 16-bit PCM."""
    import struct
    count = len(data) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", data)
    return (sum(s * s for s in samples) / count) ** 0.5 / 32768.0


def listen() -> Optional[bytes]:
    """
    Record from the microphone until VAD_SILENCE_MS of silence.
    Returns raw 16-bit PCM bytes at SAMPLE_RATE, or None on error.

    Uses sounddevice for cross-platform mic capture.
    VAD is simple energy-based (fast, no model needed).
    """
    try:
        import sounddevice as sd
    except ImportError:
        print(f"{RED}sounddevice not installed. Run: pip install sounddevice{RESET}")
        return None

    silence_chunks = int((VAD_SILENCE_MS / 1000) * SAMPLE_RATE / CHUNK_FRAMES)
    max_record_s   = 60  # hard cap

    frames: list[bytes] = []
    silent_count  = 0
    speaking      = False
    total_chunks  = 0
    max_chunks    = int(max_record_s * SAMPLE_RATE / CHUNK_FRAMES)

    print(f"\n  {MIC}  {GREEN}Listening…{RESET}  {DIM}(speak now, pause to finish){RESET}",
          flush=True)

    q: queue.Queue = queue.Queue()

    def _callback(indata, frame_count, time_info, status):
        q.put(bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_FRAMES,
            dtype="int16",
            channels=CHANNELS,
            callback=_callback,
        ):
            while total_chunks < max_chunks:
                try:
                    chunk = q.get(timeout=2.0)
                except queue.Empty:
                    break

                total_chunks += 1
                energy = _rms(chunk)

                if energy > VAD_THRESHOLD:
                    speaking = True
                    silent_count = 0
                    frames.append(chunk)
                elif speaking:
                    frames.append(chunk)
                    silent_count += 1
                    if silent_count >= silence_chunks:
                        break  # done speaking

    except Exception as e:
        print(f"{RED}Mic error: {e}{RESET}")
        return None

    if not frames or not speaking:
        print(f"  {DIM}(nothing heard){RESET}")
        return None

    print(f"  {DIM}Captured {len(frames) * CHUNK_FRAMES / SAMPLE_RATE:.1f}s of audio{RESET}")
    return b"".join(frames)


# ═══════════════════════════════════════════════════════════════════════════════
# Speech-to-Text
# ═══════════════════════════════════════════════════════════════════════════════

def _pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw PCM in a WAV container (in memory)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def transcribe(pcm: bytes) -> Optional[str]:
    """
    Transcribe PCM audio to text.
    Tries faster-whisper first, falls back to original whisper.
    """
    wav_bytes = _pcm_to_wav_bytes(pcm)

    # ── faster-whisper (preferred) ────────────────────────────────────────────
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        try:
            segments, _ = model.transcribe(tmp, beam_size=5, language="en")
            text = " ".join(s.text for s in segments).strip()
        finally:
            Path(tmp).unlink(missing_ok=True)
        return text or None

    except ImportError:
        pass
    except Exception as e:
        print(f"{YELLOW}faster-whisper error: {e}{RESET}")

    # ── original whisper (fallback) ───────────────────────────────────────────
    try:
        import whisper
        import numpy as np
        model = whisper.load_model(WHISPER_MODEL)
        # whisper wants float32 numpy array
        import struct
        count = len(pcm) // 2
        samples = struct.unpack(f"<{count}h", pcm)
        audio = np.array(samples, dtype=np.float32) / 32768.0
        result = model.transcribe(audio, language="en")
        text = result.get("text", "").strip()
        return text or None

    except ImportError:
        print(f"{RED}No whisper library found. Install: pip install faster-whisper{RESET}")
        return None
    except Exception as e:
        print(f"{RED}Whisper error: {e}{RESET}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Text-to-Speech  (multiple backends)
# ═══════════════════════════════════════════════════════════════════════════════

def _clean_for_tts(text: str) -> str:
    """Strip markdown, code blocks, and tool noise before speaking."""
    import re
    # Remove code fences
    text = re.sub(r"```[\s\S]*?```", " [code block] ", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Remove markdown bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove headings markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove tool/thinking markers
    text = re.sub(r"<tool_use>.*?</tool_use>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[.*?\]", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _play_wav(wav_path: str) -> None:
    """Play a WAV file cross-platform."""
    try:
        import sounddevice as sd
        import soundfile as sf
        data, sr = sf.read(wav_path, dtype="float32")
        sd.play(data, sr)
        sd.wait()
        return
    except ImportError:
        pass
    # Fallback: system player
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
    elif sys.platform == "darwin":
        os.system(f"afplay {wav_path!r}")
    else:
        for player in ("aplay", "paplay", "ffplay -nodisp -autoexit"):
            if os.system(f"{player} {wav_path!r} > /dev/null 2>&1") == 0:
                break


def _play_bytes(audio_bytes: bytes, fmt: str = "wav") -> None:
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        _play_wav(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Kokoro backend ────────────────────────────────────────────────────────────

def _speak_kokoro(text: str) -> None:
    """
    Kokoro TTS — local, high quality neural voice.
    pip install kokoro soundfile
    Voice options: af_heart, af_bella, af_nicole, am_adam, am_michael,
                   bf_emma, bf_isabella, bm_george, bm_lewis
    """
    try:
        from kokoro import KPipeline
    except ImportError:
        raise RuntimeError(
            "Kokoro not installed. Run: pip install kokoro soundfile\n"
            "Voice models download automatically on first use."
        )

    voice = TTS_VOICE or "af_heart"
    # lang code: 'a' = American English, 'b' = British English
    lang = "b" if voice.startswith("b") else "a"

    pipeline = KPipeline(lang_code=lang)
    import soundfile as sf
    import sounddevice as sd
    import numpy as np

    # Stream sentence-by-sentence so audio starts before full text is ready
    for _, _, audio in pipeline(text, voice=voice, speed=TTS_SPEED, split_pattern=r"[.!?]+"):
        if audio is not None and len(audio) > 0:
            sd.play(audio, 24000)
            sd.wait()


# ── Piper backend ─────────────────────────────────────────────────────────────

def _speak_piper(text: str) -> None:
    """
    Piper TTS — local binary, fast, good quality.
    Download from: https://github.com/rhasspy/piper/releases
    Set PIPER_BINARY and PIPER_MODEL in .env
    """
    import subprocess
    binary = PIPER_BINARY
    model  = PIPER_MODEL

    if not model:
        raise RuntimeError(
            "PIPER_MODEL not set in .env.\n"
            "Download a voice from https://github.com/rhasspy/piper/blob/master/VOICES.md\n"
            "Example: PIPER_MODEL=/path/to/en_US-lessac-medium.onnx"
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name

    try:
        proc = subprocess.run(
            [binary, "--model", model, "--output_file", tmp, "--length_scale",
             str(1.0 / max(TTS_SPEED, 0.1))],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Piper error: {proc.stderr.decode()[:200]}")
        _play_wav(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Edge-TTS backend ──────────────────────────────────────────────────────────

def _speak_edge(text: str) -> None:
    """
    Microsoft Edge TTS — cloud, excellent quality, free for personal use.
    pip install edge-tts
    Requires internet. Voice: set EDGE_TTS_VOICE in .env.
    Run `edge-tts --list-voices` to see all available voices.
    """
    try:
        import edge_tts
        import asyncio
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

    voice = TTS_VOICE or EDGE_VOICE

    async def _synth():
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        communicate = edge_tts.Communicate(text, voice, rate=f"{int((TTS_SPEED-1)*100):+d}%")
        await communicate.save(tmp)
        return tmp

    tmp = asyncio.run(_synth())
    try:
        # Convert mp3 → wav for playback (or play directly)
        try:
            import soundfile as sf
            import sounddevice as sd
            data, sr = sf.read(tmp, dtype="float32")
            sd.play(data, sr); sd.wait()
        except Exception:
            # Fallback: system player that can handle mp3
            if sys.platform == "win32":
                os.startfile(tmp)
                time.sleep(3)
            elif sys.platform == "darwin":
                os.system(f"afplay {tmp!r}")
            else:
                os.system(f"mpg123 -q {tmp!r} 2>/dev/null || ffplay -nodisp -autoexit {tmp!r} > /dev/null 2>&1")
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── ElevenLabs backend ────────────────────────────────────────────────────────

def _speak_elevenlabs(text: str) -> None:
    """
    ElevenLabs TTS — best quality, requires API key.
    pip install elevenlabs
    Set ELEVENLABS_API_KEY in .env.
    Set TTS_VOICE to a voice ID or name (e.g. 'Rachel', 'Adam').
    """
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import play, Voice, VoiceSettings
    except ImportError:
        raise RuntimeError("elevenlabs not installed. Run: pip install elevenlabs")

    if not ELEVENLABS_KEY:
        raise RuntimeError(
            "ELEVENLABS_API_KEY not set in .env.\n"
            "Get a free key at https://elevenlabs.io"
        )

    client = ElevenLabs(api_key=ELEVENLABS_KEY)
    voice  = TTS_VOICE or "Rachel"

    audio = client.generate(
        text=text,
        voice=Voice(
            voice_id=voice,
            settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
            ),
        ),
        model="eleven_multilingual_v2",
    )
    play(audio)


# ── pyttsx3 fallback backend ──────────────────────────────────────────────────

def _speak_pyttsx3(text: str) -> None:
    """
    pyttsx3 — zero dependency OS TTS. Works offline.
    Sounds robotic but always available.
    pip install pyttsx3
    """
    try:
        import pyttsx3
    except ImportError:
        raise RuntimeError("pyttsx3 not installed. Run: pip install pyttsx3")

    engine = pyttsx3.init()
    engine.setProperty("rate", int(engine.getProperty("rate") * TTS_SPEED))
    if TTS_VOICE:
        voices = engine.getProperty("voices")
        for v in voices:
            if TTS_VOICE.lower() in v.name.lower() or TTS_VOICE == v.id:
                engine.setProperty("voice", v.id)
                break
    engine.say(text)
    engine.runAndWait()


# ── Public speak() dispatcher ─────────────────────────────────────────────────

_BACKENDS = {
    "kokoro":     _speak_kokoro,
    "piper":      _speak_piper,
    "edge":       _speak_edge,
    "elevenlabs": _speak_elevenlabs,
    "pyttsx3":    _speak_pyttsx3,
}


def speak(text: str, backend: str = TTS_BACKEND) -> None:
    """
    Speak text aloud using the configured TTS backend.
    Strips markdown/code automatically.
    Falls back through backends if the primary fails.
    """
    clean = _clean_for_tts(text)
    if not clean:
        return

    # Sentence-cap for very long responses (speak first ~500 chars, then rest)
    MAX_CHARS = 1200
    if len(clean) > MAX_CHARS:
        # Find a sentence boundary
        cut = clean.rfind(". ", 0, MAX_CHARS)
        if cut == -1:
            cut = MAX_CHARS
        clean = clean[: cut + 1]

    fn = _BACKENDS.get(backend)
    if fn is None:
        print(f"{YELLOW}Unknown TTS backend '{backend}'. Using pyttsx3.{RESET}")
        fn = _speak_pyttsx3

    print(f"  {SPEAK}  {DIM}speaking via {backend}…{RESET}", flush=True)
    try:
        fn(clean)
    except RuntimeError as e:
        print(f"{YELLOW}[TTS] {e}{RESET}")
        if backend != "pyttsx3":
            print(f"{DIM}Falling back to pyttsx3…{RESET}")
            try:
                _speak_pyttsx3(clean)
            except Exception as e2:
                print(f"{RED}[TTS] pyttsx3 also failed: {e2}{RESET}")
    except Exception as e:
        print(f"{RED}[TTS] Unexpected error: {e}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# VoiceSession — stateful wrapper used by main.py
# ═══════════════════════════════════════════════════════════════════════════════

class VoiceSession:
    """
    Manages voice mode state. Created once; toggled on/off via enable()/disable().

    Usage in REPL:
        vs = VoiceSession()
        vs.enable()          # /voice on
        vs.disable()         # /voice off
        text = vs.get_input()   # mic if enabled, input() if not
        vs.maybe_speak(reply)   # speaks if enabled
    """

    def __init__(self):
        self.active  = False
        self.backend = TTS_BACKEND

    def enable(self, backend: Optional[str] = None) -> None:
        self.active  = True
        if backend:
            self.backend = backend.lower().strip()
        self._check_deps()
        print(
            f"\n  {MIC}  {GREEN}{BOLD}Voice mode ON{RESET}  "
            f"{DIM}(STT: whisper/{WHISPER_MODEL}  TTS: {self.backend}){RESET}\n"
            f"  {DIM}Say your message after the prompt. Pause to send. /voice off to stop.{RESET}"
        )

    def disable(self) -> None:
        self.active = False
        print(f"\n  {DIM}Voice mode OFF — back to keyboard.{RESET}")

    def toggle(self) -> None:
        if self.active:
            self.disable()
        else:
            self.enable()

    def _check_deps(self) -> None:
        """Warn if required packages are missing, with install hints."""
        missing = []
        try:
            import sounddevice
        except (ImportError, OSError):
            missing.append("sounddevice")
        try:
            import faster_whisper
        except ImportError:
            try:
                import whisper
            except ImportError:
                missing.append("faster-whisper")

        if missing:
            pkgs = " ".join(missing)
            print(
                f"{YELLOW}  Missing packages: {pkgs}\n"
                f"  Install: pip install {pkgs}{RESET}"
            )

    def get_input(self, fallback_prompt: str = "") -> Optional[str]:
        """
        Get user input: mic transcription if active, keyboard otherwise.
        Returns None if nothing was captured (user should re-prompt).
        """
        if not self.active:
            try:
                return input(fallback_prompt).strip() or None
            except (EOFError, KeyboardInterrupt):
                return None

        pcm = listen()
        if pcm is None:
            return None

        print(f"  {DIM}Transcribing…{RESET}", end="", flush=True)
        t0 = time.time()
        text = transcribe(pcm)
        elapsed = time.time() - t0

        if not text:
            print(f"\r  {YELLOW}Could not transcribe audio.{RESET}      ")
            return None

        print(f"\r  {CYAN}{BOLD}You:{RESET} {text}  {DIM}({elapsed:.1f}s){RESET}")
        return text

    def maybe_speak(self, text: str) -> None:
        """Speak text if voice mode is active."""
        if self.active and text:
            speak(text, backend=self.backend)

    def set_backend(self, name: str) -> None:
        if name not in _BACKENDS:
            print(f"{RED}Unknown backend '{name}'. Options: {', '.join(_BACKENDS)}{RESET}")
            return
        self.backend = name
        print(f"  {DIM}TTS backend → {name}{RESET}")

    @staticmethod
    def list_backends() -> str:
        lines = [
            f"  {BOLD}Available TTS backends:{RESET}",
            f"  {CYAN}kokoro{RESET}      — local, natural-sounding neural voice (recommended)",
            f"  {CYAN}piper{RESET}       — local binary, fast, good quality",
            f"  {CYAN}edge{RESET}        — Microsoft neural TTS, excellent (needs internet)",
            f"  {CYAN}elevenlabs{RESET}  — best quality, cloneable voices (needs API key)",
            f"  {CYAN}pyttsx3{RESET}     — offline OS voice, always available (robotic)",
            f"",
            f"  Set default in .env: TTS_BACKEND=kokoro",
            f"  Switch live:         /voice backend edge",
        ]
        return "\n".join(lines)
