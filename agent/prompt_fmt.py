"""
agent/prompt_fmt.py — Model-family-aware prompt formatter.
Detects model family from path and emits the correct chat template.
Supports: Qwen3/ChatML, DeepSeek-R1, Llama-3, Gemma.
"""
from __future__ import annotations
import re


class PromptFmt:
    """
    Detect the model family from the model path and build correctly-formatted
    prompts. Mirrors EVE's PromptFmt with extended Qwen3 support.
    """

    def __init__(self, model_path: str = "") -> None:
        p = str(model_path).lower()
        if "deepseek" in p or "r1-distill" in p or "r1_distill" in p:
            self.family = "deepseek"
        elif "qwen3" in p or "qwen_qwen3" in p or "qwen2.5" in p:
            self.family = "qwen3"
        elif "llama-3" in p or "llama3" in p or "meta-llama" in p:
            self.family = "llama3"
        elif "gemma" in p:
            self.family = "gemma"
        elif "mistral" in p or "mixtral" in p:
            self.family = "mistral"
        else:
            self.family = "chatml"

    def build_messages(
        self,
        system: str,
        messages: list[dict],
    ) -> list[dict]:
        """Return messages list for /v1/chat/completions API."""
        out = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(messages)
        return out

    def build_raw(
        self,
        system: str = "",
        user: str = "",
        tool_results: str = "",
        start_think: bool = False,
    ) -> str:
        """
        Build a raw text prompt string for direct llama-cpp completion.
        Used when bypassing the chat API.
        """
        nl = "\n"
        if self.family == "deepseek":
            parts = []
            if system:
                parts.append(system)
            parts.append("<|User|>" + user)
            if tool_results:
                parts.append("[Results]" + nl + tool_results)
            parts.append("<|Assistant|>")
            if start_think:
                parts.append("<think>" + nl)
            return nl.join(parts)

        elif self.family == "llama3":
            s = "<|begin_of_text|>"
            if system:
                s += (f"<|start_header_id|>system<|end_header_id|>{nl*2}"
                      f"{system}<|eot_id|>")
            s += (f"<|start_header_id|>user<|end_header_id|>{nl*2}{user}")
            if tool_results:
                s += nl + tool_results
            s += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>{nl*2}"
            if start_think:
                s += "<think>" + nl
            return s

        elif self.family == "gemma":
            s = ""
            if system:
                s += f"<start_of_turn>system{nl}{system}<end_of_turn>{nl}"
            s += f"<start_of_turn>user{nl}{user}"
            if tool_results:
                s += nl + tool_results
            s += f"<end_of_turn>{nl}<start_of_turn>model{nl}"
            if start_think:
                s += "<think>" + nl
            return s

        elif self.family == "mistral":
            s = f"[INST] "
            if system:
                s += f"{system}\n\n"
            s += user
            if tool_results:
                s += "\n\n" + tool_results
            s += " [/INST]"
            return s

        else:  # chatml / qwen3
            s = ""
            if system:
                s += f"<|im_start|>system{nl}{system}<|im_end|>{nl}"
            s += f"<|im_start|>user{nl}{user}<|im_end|>{nl}"
            if tool_results:
                s += f"<|im_start|>tool_results{nl}{tool_results}<|im_end|>{nl}"
            s += f"<|im_start|>assistant{nl}"
            if start_think:
                s += "<think>" + nl
            return s

    def stop_tokens(self) -> list[str]:
        if self.family == "deepseek":
            return ["<|User|>"]
        elif self.family == "llama3":
            return ["<|eot_id|>", "<|start_header_id|>"]
        elif self.family == "gemma":
            return ["<end_of_turn>", "<start_of_turn>"]
        elif self.family == "mistral":
            return ["[INST]", "</s>"]
        else:
            return ["<|im_start|>", "<|im_end|>"]

    def strip_think(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def __repr__(self) -> str:
        return f"PromptFmt(family={self.family!r})"
