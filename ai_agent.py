#!/usr/bin/env python3
"""
AI Agent - CPU-optimized assistant for 8GB RAM
Uses llama-cpp-python with quantized GGUF models
"""

import yaml
import os
import json
from pathlib import Path
from llama_cpp import Llama
from datetime import datetime


class AgentMemory:
    def __init__(self, config_path: str = "agent_config.yaml"):
        self.config = self._load_config(config_path)
        self.persist_file = str(Path(config_path).with_suffix(".json"))
        self.memory = self._load_memory()
        self._model = None  # lazy-loaded once

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

        # Pull model path from .env / environment (MODEL_PATH wins over yaml name)
        env_model = os.environ.get("MODEL_PATH", "").strip()
        if env_model:
            cfg.setdefault("model", {})["path"] = env_model.replace("\\", "/")

        # Defaults
        cfg.setdefault("model", {})
        cfg["model"].setdefault("path", "model.gguf")
        cfg["model"].setdefault("context_window", 4096)
        cfg["model"].setdefault("max_tokens", 512)
        cfg["hardware"] = {"cpu_only": True, "ram_limit_gb": 8}
        return cfg

    def _load_memory(self) -> dict:
        if os.path.exists(self.persist_file):
            try:
                with open(self.persist_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"user_preferences": {}, "session_history": [], "system_facts": {}}

    def _save_memory(self):
        try:
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"[warn] Cannot save memory: {e}")

    def _get_model(self) -> Llama:
        if self._model is not None:
            return self._model

        model_path = self.config["model"]["path"]
        n_ctx = int(self.config["model"]["context_window"])

        # CPU-only: n_gpu_layers=0, thread count from env or sensible default
        n_threads = int(os.environ.get("N_THREADS", 0)) or os.cpu_count() or 4

        print(f"[ai_agent] Loading model: {model_path}")
        print(f"[ai_agent] CPU-only | ctx={n_ctx} | threads={n_threads}")

        self._model = Llama(
            model_path=model_path,   # FIX: was model_name= (wrong kwarg)
            n_ctx=n_ctx,
            n_gpu_layers=0,          # FIX: force CPU
            n_threads=n_threads,
            use_mlock=False,         # keep under 8 GB
            verbose=False,
        )
        print("[ai_agent] Model loaded.")
        return self._model

    def _generate_completion(self, prompt: str, system_memory: dict = None) -> str:
        model = self._get_model()
        max_tokens = int(self.config["model"]["max_tokens"])

        mem_snippet = json.dumps(system_memory or self.memory, indent=2)
        full_prompt = (
            "You are a helpful AI assistant. Use the memory below when relevant.\n\n"
            f"MEMORY:\n{mem_snippet}\n\n"
            f"USER: {prompt}\n\nASSISTANT:"
        )

        try:
            result = model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=float(os.environ.get("TEMPERATURE", 0.7)),
                top_p=float(os.environ.get("TOP_P", 0.95)),
                stop=["USER:", "\nUSER"],
                echo=False,
            )
            # FIX: was response["completion"] — correct key is choices[0]["text"]
            return result["choices"][0]["text"].strip()
        except Exception as e:
            print(f"[ai_agent] Generation error: {e}")
            return "Sorry, I ran into an error generating a response."

    def chat(self, message: str, user_id: str = "default") -> str:
        timestamp = datetime.now().isoformat()

        system_memory = {
            "user_preferences": self.memory["user_preferences"],
            # Only send last 10 turns to keep prompt small / under 8 GB
            "session_history": self.memory["session_history"][-10:],
            "system_facts": self.memory["system_facts"],
        }

        response = self._generate_completion(message, system_memory)

        self.memory["session_history"].append({
            "timestamp": timestamp,
            "user_id": user_id,
            "message": message,
            "response": response,
        })
        self._save_memory()
        return response

    def add_preference(self, key: str, value: str):
        self.memory["user_preferences"][key] = value
        self._save_memory()

    def get_pref(self, key: str):
        return self.memory["user_preferences"].get(key)

    def update_preference(self, key: str, value: str):
        self.memory["user_preferences"][key] = value
        self._save_memory()

    def get_session_history(self, limit: int = 100):
        return self.memory["session_history"][-limit:]


def main():
    # Load .env if present (simple key=value parser, no extra deps)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    config_path = str(Path(__file__).parent / "agent_config.yaml")
    agent = AgentMemory(config_path)

    print("\n" + "=" * 50)
    print("AI AGENT READY  (CPU | 8 GB)")
    print("=" * 50)
    print("Type your message and press Enter.")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "stop"}:
                print("Agent: Goodbye!")
                break
            print("Agent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(f"{response}\n")
    except KeyboardInterrupt:
        print("\n[ai_agent] Interrupted. Bye.")


if __name__ == "__main__":
    main()