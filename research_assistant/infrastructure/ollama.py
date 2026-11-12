"""Ollama generation adapter."""

import json
import requests
from research_assistant import config


def generate(
    prompt: str,
    system: str = "",
    *,
    temperature: float = 0.0,
    num_predict: int = 1024,
    repeat_penalty: float = 1.1,
) -> str:
    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    options = {
        "temperature": temperature,
        "num_ctx": 4096,
        "num_predict": num_predict,
        "repeat_penalty": repeat_penalty,
    }
    if temperature == 0.0:
        options["seed"] = 42
    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }
    if system:
        payload["system"] = system
    try:
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: `ollama serve`"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Ollama request timed out after 120s (model: {config.OLLAMA_MODEL})."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Ollama request failed ({config.OLLAMA_MODEL}): {exc}"
        ) from exc
    parts = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
            parts.append(chunk.get("response", ""))
            if chunk.get("done"):
                break
        except json.JSONDecodeError:
            continue
    return "".join(parts).strip()
