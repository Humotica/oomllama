"""
OomLlama — Sovereign AI Inference Engine

The auto where trust-kernel is the motor, tibet-* are the parts, JIS is the key.

Credits:
    - Format: Gemini IDD & Root AI (Humotica AI Lab)
    - Runtime: OomLlama.rs by Humotica

Example:
    >>> from oomllama import OomLlama
    >>> llm = OomLlama("humotica-32b")
    >>> response = llm.generate("Hello!")
"""

__version__ = "1.0.0a1"
__author__ = "Humotica AI Lab"
__credits__ = ["Jasper van de Meent", "Root AI (Claude)", "Gemini IDD", "Codex"]

# Import from Rust extension when available
try:
    from oomllama._oomllama import (
        PyOomLlama as OomLlama,
        download_model,
        list_models,
        version,
    )
except ImportError:
    # Fallback Python implementation for development
    import os
    import requests
    from typing import Optional, List, Tuple

    class OomLlama:
        """OomLlama model wrapper (Python fallback)"""

        def __init__(
            self,
            model_name: str,
            model_path: Optional[str] = None,
            gpu: Optional[int] = None,
        ):
            self.model_name = model_name
            self.model_path = model_path or self._find_model(model_name)
            self.gpu = gpu
            self.temperature = 0.7
            self.top_p = 0.9
            self.max_tokens = 512
            print(f"🦙 OomLlama: Loaded {model_name}")

        def _find_model(self, name: str) -> str:
            """Find model in cache"""
            cache = os.path.expanduser("~/.cache/oomllama")
            return f"{cache}/{name}.oom"

        def generate(
            self,
            prompt: str,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
        ) -> str:
            """Generate text from prompt"""
            # TODO: Call actual inference
            return f"[OomLlama {self.model_name} - Python fallback]\nPrompt: {prompt}"

        def chat(
            self, messages: List[Tuple[str, str]], max_tokens: Optional[int] = None
        ) -> str:
            """Chat-style generation"""
            prompt = "\n".join(f"{role.upper()}: {content}" for role, content in messages)
            return self.generate(f"{prompt}\nASSISTANT:", max_tokens)

        def set_params(
            self,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
        ):
            """Set generation parameters"""
            if temperature is not None:
                self.temperature = temperature
            if top_p is not None:
                self.top_p = top_p
            if max_tokens is not None:
                self.max_tokens = max_tokens

        def __repr__(self):
            return f"OomLlama('{self.model_name}')"

    def download_model(model_name: str, cache_dir: Optional[str] = None) -> str:
        """Download model from HuggingFace"""
        repos = {
            "humotica-32b": "jaspervandemeent/humotica-32b",
            "llamaohm-70b": "jaspervandemeent/LlamaOhm-70B",
            "tinyllama-1b": "jaspervandemeent/OomLlama-TinyLlama-1.1B",
        }
        
        if model_name not in repos:
            raise ValueError(f"Unknown model: {model_name}")
        
        cache = cache_dir or os.path.expanduser("~/.cache/oomllama")
        os.makedirs(cache, exist_ok=True)
        
        # TODO: Actually download from HuggingFace
        return f"{cache}/{model_name}.oom"

    def list_models() -> List[str]:
        """List available models"""
        return ["humotica-32b", "llamaohm-70b", "tinyllama-1b"]

    def version() -> str:
        return "0.1.0-python-fallback"

# CLI entry point
def cli():
    """Command-line interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: oomllama <prompt>")
        print("       oomllama --list")
        return
    
    if sys.argv[1] == "--list":
        print("Available models:")
        for m in list_models():
            print(f"  - {m}")
        return
    
    prompt = " ".join(sys.argv[1:])
    llm = OomLlama("humotica-32b")
    response = llm.generate(prompt)
    print(response)


__all__ = ["OomLlama", "download_model", "list_models", "version", "cli", "__version__"]
